import matplotlib
import argparse
import os
matplotlib.use('Agg')

from sodetlib.det_config import DetConfig
from sodetlib.operations import uxm_setup

if __name__ == "__main__":
    cfg = DetConfig()
    parser = argparse.ArgumentParser()

    parser.add_argument("--bands", type=int, default=None, nargs="+")

    args = cfg.parse_args(parser)
    S = cfg.get_smurf_control(dump_configs=True)

    uxm_setup.uxm_setup(S, cfg, bands=args.bands)
