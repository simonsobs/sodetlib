import pytest
from sodetlib.det_config import DetConfig
import os
import shutil


def setup_module():
    if os.path.exists('config'):
        shutil.rmtree('config')




def test_dev_cfg():
    cfg = DetConfig()
    cfg.load_config_files()
    cfg.dump_configs('config')
    assert (True)