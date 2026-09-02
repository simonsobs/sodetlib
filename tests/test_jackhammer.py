import atexit
import os
import shutil
import subprocess
import tempfile
import threading
from unittest import mock
from unittest.mock import MagicMock

import pytest
import yaml

# jackhammer.py reads OCS_CONFIG_DIR and parses sys_config.yml at import
# time, so the env var and config file must exist before the import.
_test_config_dir = tempfile.mkdtemp(prefix='jackhammer_test_')
_test_sys_config = {
    'crate_id': 1,
    'shelf_manager': 'shm-smrf-sp01',
    'slot_order': [2, 3, 4],
    'slots': {
        'SLOT[2]': {
            'device_config': '$OCS_CONFIG_DIR/dev_cfg.yml',
            'pysmurf_config': '$OCS_CONFIG_DIR/pysmurf.cfg',
            'stream_port': 4536,
        },
        'SLOT[3]': {
            'device_config': '$OCS_CONFIG_DIR/dev_cfg.yml',
            'pysmurf_config': '$OCS_CONFIG_DIR/pysmurf.cfg',
            'stream_port': 4537,
        },
        'SLOT[4]': {
            'device_config': '$OCS_CONFIG_DIR/dev_cfg.yml',
            'pysmurf_config': '$OCS_CONFIG_DIR/pysmurf.cfg',
            'stream_port': 4538,
        },
    },
}

with open(os.path.join(_test_config_dir, 'sys_config.yml'), 'w') as f:
    yaml.dump(_test_sys_config, f)

os.environ['OCS_CONFIG_DIR'] = _test_config_dir
atexit.register(shutil.rmtree, _test_config_dir, ignore_errors=True)

try:
    from sodetlib.hammers import jackhammer  # noqa: E402
except ImportError as e:
    pytest.skip(f"Missing dependency: {e}", allow_module_level=True)

MODULE = 'sodetlib.hammers.jackhammer'


def _ping_side_effect(failing_slots, exc_type=subprocess.TimeoutExpired):
    """Emulate carrier ping results after a crate reboot.

    hammer() calls ``subprocess.run(['ping_carrier', ip], timeout=...)``
    for each slot after rebooting carriers.  This side_effect intercepts
    those calls and raises for slots in ``failing_slots``, while letting
    all other subprocess.run calls (e.g. ``docker stop``) succeed.

    The slot number is extracted from the carrier IP, which follows the
    convention ``10.0.<crate_id>.<slot + 100>``.

    Emulated failure modes:
      - TimeoutExpired: carrier never came back after reboot (default)
      - Any other exc_type: unexpected error during ping (e.g. OSError
        when the ping_carrier binary is missing)
    """
    def side_effect(cmd, **kwargs):
        if isinstance(cmd, list) and cmd and cmd[0] == 'ping_carrier':
            ip = cmd[1]
            slot = int(ip.split('.')[-1]) - 100
            if slot in failing_slots:
                if exc_type is subprocess.TimeoutExpired:
                    raise subprocess.TimeoutExpired(cmd, kwargs.get('timeout', 120))
                raise exc_type(f"ping failed for {ip}")
        return MagicMock(returncode=0)
    return side_effect


def _server_side_effect(failing_slots, exc_type=TimeoutError):
    """Emulate server connection results after streamer dockers start.

    hammer() calls ``check_server_connection(port, retry=True,
    timeout=...)`` for each slot to wait for the pysmurf server
    to become reachable.  This side_effect raises for slots in
    ``failing_slots`` and returns True for the rest.

    Emulated failure modes:
      - TimeoutError: server never became reachable within the timeout
        (default)
      - Any other exc_type: unexpected error during the EPICS check
        (e.g. a RuntimeError from the underlying caget call)
    """
    def side_effect(port, retry=False, timeout=180):
        slot = (int(port) - 9000) // 3
        if slot in failing_slots:
            if exc_type is TimeoutError:
                raise TimeoutError(
                    f"Timed out after {timeout}s waiting for server "
                    f"connection to {epics_server}"
                )
            raise exc_type(f"Server connection error for slot {slot}")
        return True
    return side_effect


def _util_run_side_effect(failing_slots=None, raising_slots=None):
    """Emulate pysmurf setup results run inside setup_smurfs threads.

    setup_smurfs() calls ``util_run('python3', args=[..., '-N', slot])``
    in a thread per slot.  This side_effect parses the ``-N <slot>``
    argument to determine which slot is being set up, then either
    succeeds, returns a failing exit code, or raises.

    Emulated failure modes:
      - failing_slots: pysmurf setup ran but exited non-zero (e.g. a
        configuration error detected by the setup script)
      - raising_slots: the util_run call itself raised (e.g.
        FileNotFoundError when the docker binary is missing, or a
        subprocess error)
    """
    failing_slots = failing_slots or set()
    raising_slots = raising_slots or {}

    def side_effect(cmd, args=None, **kwargs):
        args = args or []
        if '-N' in args:
            idx = args.index('-N')
            if idx + 1 < len(args):
                slot = int(args[idx + 1])
                if slot in raising_slots:
                    raise raising_slots[slot]
                if slot in failing_slots:
                    return MagicMock(returncode=1)
        return MagicMock(returncode=0)
    return side_effect


@pytest.fixture
def hammer_mocks():
    """Patches all external dependencies of hammer() with safe defaults.

    Mocks util_run rather than setup_smurfs so that the real threading
    and error-handling code inside setup_smurfs is exercised.
    """
    with mock.patch.multiple(
        MODULE,
        dump_docker_logs=mock.DEFAULT,
        kill_bad_dockers=mock.DEFAULT,
        start_services=mock.DEFAULT,
        start_sync_dockers=mock.DEFAULT,
        controller_cmd=mock.DEFAULT,
        run_on_shelf_manager=mock.DEFAULT,
        setup_fans=mock.DEFAULT,
        check_server_connection=mock.DEFAULT,
        util_run=mock.DEFAULT,
        subprocess=mock.DEFAULT,
        time=mock.DEFAULT,
    ) as mocks:
        mocks['subprocess'].run.return_value = MagicMock(returncode=0)
        mocks['subprocess'].TimeoutExpired = subprocess.TimeoutExpired
        mocks['check_server_connection'].return_value = True
        mocks['util_run'].return_value = MagicMock(returncode=0)
        yield mocks


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_all_slots_succeed(hammer_mocks):
    result = jackhammer.hammer(slots=[2, 3, 4], no_dump=True)

    assert result == {'succeeded': [2, 3, 4], 'failed': {}}


def test_default_slots_from_config(hammer_mocks):
    result = jackhammer.hammer(no_dump=True)

    assert result == {'succeeded': [2, 3, 4], 'failed': {}}


# ---------------------------------------------------------------------------
# Single-stage failures
# ---------------------------------------------------------------------------

def test_single_slot_fails_at_ping(hammer_mocks):
    hammer_mocks['subprocess'].run.side_effect = _ping_side_effect({3})

    result = jackhammer.hammer(slots=[2, 3, 4], no_dump=True)

    assert result['succeeded'] == [2, 4]
    assert 'timed out' in result['failed'][3].lower()


def test_single_slot_fails_at_server(hammer_mocks):
    hammer_mocks['check_server_connection'].side_effect = _server_side_effect({3})

    result = jackhammer.hammer(slots=[2, 3, 4], no_dump=True)

    assert result['succeeded'] == [2, 4]
    assert 3 in result['failed']


def test_single_slot_fails_at_setup(hammer_mocks):
    hammer_mocks['util_run'].side_effect = _util_run_side_effect(
        failing_slots={3}
    )

    result = jackhammer.hammer(slots=[2, 3, 4], no_dump=True)

    assert result['succeeded'] == [2, 4]
    assert 'non-zero exit code' in result['failed'][3]


def test_ping_generic_exception(hammer_mocks):
    hammer_mocks['subprocess'].run.side_effect = _ping_side_effect(
        {3}, exc_type=OSError
    )

    result = jackhammer.hammer(slots=[2, 3, 4], no_dump=True)

    assert result['succeeded'] == [2, 4]
    assert 'Carrier ping failed' in result['failed'][3]


def test_server_generic_exception(hammer_mocks):
    hammer_mocks['check_server_connection'].side_effect = _server_side_effect(
        {4}, exc_type=RuntimeError
    )

    result = jackhammer.hammer(slots=[2, 3, 4], no_dump=True)

    assert result['succeeded'] == [2, 3]
    assert 'Server connection failed' in result['failed'][4]


# ---------------------------------------------------------------------------
# Cascading / total failure
# ---------------------------------------------------------------------------

def test_all_slots_fail_at_ping_early_return(hammer_mocks):
    hammer_mocks['subprocess'].run.side_effect = _ping_side_effect({2, 3, 4})

    result = jackhammer.hammer(slots=[2, 3, 4], no_dump=True)

    assert result['succeeded'] == []
    assert set(result['failed'].keys()) == {2, 3, 4}


def test_multiple_failures_at_different_stages(hammer_mocks):
    hammer_mocks['subprocess'].run.side_effect = _ping_side_effect({2})
    hammer_mocks['check_server_connection'].side_effect = _server_side_effect({3})
    hammer_mocks['util_run'].side_effect = _util_run_side_effect(
        raising_slots={4: RuntimeError("docker not found")}
    )

    result = jackhammer.hammer(slots=[2, 3, 4], no_dump=True)

    assert result['succeeded'] == []
    assert set(result['failed'].keys()) == {2, 3, 4}


# ---------------------------------------------------------------------------
# Mode flags
# ---------------------------------------------------------------------------

def test_no_reboot_skips_ping_phase(hammer_mocks):
    result = jackhammer.hammer(slots=[2, 3, 4], no_reboot=True, no_dump=True)

    assert result == {'succeeded': [2, 3, 4], 'failed': {}}
    hammer_mocks['run_on_shelf_manager'].assert_not_called()


def test_skip_setup_mode(hammer_mocks):
    result = jackhammer.hammer(slots=[2, 3, 4], skip_setup=True, no_dump=True)

    assert result == {'succeeded': [2, 3, 4], 'failed': {}}
    hammer_mocks['util_run'].assert_not_called()


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def test_invalid_slot_raises(hammer_mocks):
    with pytest.raises(ValueError, match='not valid'):
        jackhammer.hammer(slots=[99], no_dump=True)


# ---------------------------------------------------------------------------
# Direct setup_smurfs tests
# ---------------------------------------------------------------------------

@mock.patch(f'{MODULE}.util_run')
def test_setup_smurfs_all_succeed(mock_util_run):
    mock_util_run.return_value = MagicMock(returncode=0)

    errors = jackhammer.setup_smurfs([2, 3, 4])

    assert errors == {}
    assert mock_util_run.call_count == 3


@mock.patch(f'{MODULE}.util_run')
def test_setup_smurfs_nonzero_exit(mock_util_run):
    mock_util_run.side_effect = _util_run_side_effect(failing_slots={3})

    errors = jackhammer.setup_smurfs([2, 3, 4])

    assert 2 not in errors
    assert 4 not in errors
    assert 3 in errors
    assert 'non-zero exit code' in errors[3]


@mock.patch(f'{MODULE}.util_run')
def test_setup_smurfs_exception_in_thread(mock_util_run):
    mock_util_run.side_effect = _util_run_side_effect(
        raising_slots={4: FileNotFoundError("docker binary missing")}
    )

    errors = jackhammer.setup_smurfs([2, 3, 4])

    assert 2 not in errors
    assert 3 not in errors
    assert 4 in errors
    assert 'Pysmurf setup raised' in errors[4]
    assert 'docker binary missing' in errors[4]


@mock.patch(f'{MODULE}.util_run')
def test_setup_smurfs_thread_timeout(mock_util_run):
    hang_event = threading.Event()

    def hanging_util_run(cmd, args=None, **kwargs):
        args = args or []
        if '-N' in args:
            idx = args.index('-N')
            if idx + 1 < len(args) and int(args[idx + 1]) == 3:
                hang_event.wait(timeout=5)
                return MagicMock(returncode=0)
        return MagicMock(returncode=0)

    mock_util_run.side_effect = hanging_util_run

    errors = jackhammer.setup_smurfs([2, 3], timeout=0.1)

    hang_event.set()

    assert 2 not in errors
    assert 3 in errors
    assert 'timed out' in errors[3].lower()
