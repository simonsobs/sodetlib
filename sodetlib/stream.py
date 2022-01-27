from sotodlib.io import load_smurf
from sodetlib.util import Registers
import os
import time

try:
    from pysmurf.client.util.pub import set_action
except:
    set_action = lambda : (lambda f : f)


def get_session_files(cfg, session_id, idx=None, stream_id=None):
    base_dir = cfg.sys['g3_dir']
    if stream_id is None:
        stream_id = cfg.sys['slots'][f'SLOT[{cfg.slot}]']['stream_id']
    subdir = os.path.join(base_dir, str(session_id)[:5], stream_id)
    files = sorted([
        os.path.join(subdir, f) for f in os.listdir(subdir)
        if str(session_id) in f
    ])

    if idx is None:
        return files
    elif isinstance(idx, int):
        return files[idx]
    else:  # list of indexes
        return [files[i] for i in idx]


def load_session(cfg, session_id, idx=None, stream_id=None, show_pb=False):
    """
    Loads a stream-session into an axis manager.

    Args
    ----
    cfg : DetConfig object
        DetConfig object
    session_id: int
        Session id corresonding with the stream session you wish to load
    idx: int, list(int), optional
    """
    files = get_session_files(cfg, session_id, idx, stream_id=stream_id)
    return load_smurf.load_file(files, show_pb=show_pb)


@set_action()
def take_g3_data(S, dur, **stream_kw):
    """
    Takes data for some duration

    Args
    ----
    S : SmurfControl
        Pysmurf control object
    dur : float
        Duration to take data over

    Returns
    -------
    session_id : int
        Id used to read back stream data
    """
    stream_g3_on(S, **stream_kw)
    time.sleep(dur)
    sid = stream_g3_off(S, emulator=stream_kw.get('emulator', False))
    return sid


@set_action()
def stream_g3_on(S, make_freq_mask=True, emulator=False, tag='',
                 channel_mask=None, filter_wait_time=2):
    """
    Starts the G3 data-stream. Returns the session-id corresponding with the
    data stream.

    Args
    ----
    S : S
        Pysmurf control object
    make_freq_mask : bool, optional
        Tell pysmurf to write and register the current freq mask
    emulator : bool
        If True, will enable the emulator source data-generator. Defaults to
        False.

    Return
    -------
    session_id : int
        Id used to read back streamed data
    """
    reg = Registers(S)

    reg.pysmurf_action.set(S.pub._action)
    reg.pysmurf_action_timestamp.set(S.pub._action_ts)
    reg.stream_tag.set(tag)

    S.stream_data_on(make_freq_mask=make_freq_mask, channel_mask=channel_mask,
                     filter_wait_time=filter_wait_time)

    if emulator:
        reg.source_enable.set(1)
        S.set_stream_enable(1)

    reg.open_g3stream.set(1)

    # Sometimes it takes a bit for data to propogate through to the
    # streamer
    for _ in range(5):
        sess_id = reg.g3_session_id.get()
        if sess_id != 0:
            break
        time.sleep(0.3)
    return sess_id


@set_action()
def stream_g3_off(S, emulator=False):
    """
    Stops the G3 data-stream. Returns the session-id corresponding with the
    data stream.

    Args
    ----
    S : S
        Pysmurf control object
    emulator : bool
        If True, will enable the emulator source data-generator. Defaults to
        False.

    Return
    -------
    session_id : int
        Id used to read back streamed data
    """
    reg = Registers(S)
    sess_id = reg.g3_session_id.get()

    if emulator:
        reg.source_enable.set(0)

    S.set_stream_enable(0)
    S.stream_data_off()

    reg.open_g3stream.set(0)
    reg.pysmurf_action.set('')
    reg.pysmurf_action_timestamp.set(0)
    reg.stream_tag.set('')

    # Waits until file is closed out before returning
    S.log("Waiting for g3 file to close out")
    while reg.g3_session_id.get() != 0:
        time.sleep(0.5)

    return sess_id

