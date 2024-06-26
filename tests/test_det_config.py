import pytest
from sodetlib.det_config import DetConfig
import os


@pytest.fixture
def cfg():
    (sys, dev, pysmurf) = [
        os.path.join(os.path.dirname(__file__), f'input/{fname}')
        for fname in ['sys_config.yml', 'dev_cfg.yml', 'pysmurf.cfg']
    ]
    return DetConfig(sys_file=sys, dev_file=dev, pysmurf_file=pysmurf)


def test_dev_cfg(cfg):
    cfg.parse_args(args=['-N', '2'])
    cfg.dev.update_band(1, {'dc_att': 3, 'drive': 11})
    cfg.dev.update_bias_group(0, {'bias_high': 11, 'enabled': 1})
    cfg.dev.update_experiment({"amp_50k_Id": 12})
    out = cfg.dump_configs('config', clobber=True)
    sys, dev, pysmurf = out['sys'], out['dev'], out['pysmurf']
    # Testing reloading written config file
    cfg = DetConfig(sys_file=sys, dev_file=dev, pysmurf_file=pysmurf)
    cfg.parse_args(args=['-N', '2'])
    assert (cfg.dev.bands[1]['dc_att'] == 3)


def test_failed_update(cfg):
    cfg.parse_args(args=[])
    with pytest.raises(ValueError):
        cfg.dev.update_experiment({'abcd': 12})


def test_uxm_file(cfg):
    cfg.uxm_file= 'input/uxm.yaml'
    cfg.parse_args(args=['-N', '2'])
    assert (cfg.uxm is not None)


def test_offline_pysmurf_instance(cfg):
    cfg.parse_args(args=[])
    cfg.get_smurf_control(offline=True, dump_configs=True)


def test_online_dump_confg():
    cfg = DetConfig()
    cfg.load_config_files(slot=2)
    cfg.get_smurf_control(dump_configs=False)
    config_dir = 'config_dump'
    cfg.dump_configs(config_dir, clobber=True, dump_rogue_tree=True)
