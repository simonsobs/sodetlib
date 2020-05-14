import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pysmurf.client
import argparse
import numpy as np
import pickle as pkl
from scipy import signal
import os

from sodetlib.det_config import DetConfig
from sodetlib.smurf_funcs import optimize_power_per_band


if __name__=='__main__':
    cfg = DetConfig()
    parser = argparse.ArgumentParser()
    parser.add_argument('--band', '-b', type=int, required=True,
                        help='band (must be in range [0,7])')
    parser.add_argument('--meas-time', '-m', type=float,
                        help='Time (sec) to take data for noise PSD. If not'
                             'specified, defaults to 30 sec.')
    parser.add_argument('--fix-drive', action='store_true',
                        help="If specified, not vary drive to find global min.")
    args = cfg.parse_args(parser)
    S = cfg.get_smurf_control(dump_configs=True)
    optimize_power_per_band(S, cfg, args.band, meas_time=args.meas_time,
                            fix_drive=args.fix_drive)
