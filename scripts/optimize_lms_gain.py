import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pysmurf.client
import argparse
import numpy as np

from sodetlib.det_config import DetConfig
from sodetlib.smurf_funcs.optimize_params import optimize_lms_gain


if __name__=='__main__':
    cfg = DetConfig()
    parser = argparse.ArgumentParser()
    parser.add_argument('--band', '-b', type=int, required=True,
                        help='band (must be in range [0,7])')
    parser.add_argument('--BW-target', '-m', type=float, required=True,
                        help='Target readout bandwidth to optimize lms_gain')
    args = cfg.parse_args(parser)
    S = cfg.get_smurf_control(dump_configs=True)
    optimize_lms_gain(S, cfg,args.band, args.BW_target,
                        tunefile=None, reset_rate_khz=None,
                        frac_pp=None, lms_freq=None,
                        meas_time=None, make_plot = True)
