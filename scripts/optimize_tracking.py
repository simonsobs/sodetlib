from sodetlib.legacy.smurf_funcs.optimize_params import optimize_tracking
from sodetlib.det_config import DetConfig
import numpy as np
import argparse
import pysmurf.client
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


if __name__ == '__main__':
    cfg = DetConfig()
    parser = argparse.ArgumentParser()
    parser.add_argument('--band', '-b', type=int, required=True,
                        help='band (must be in range [0,7])')
    parser.add_argument('--phi0-number', '--Nphi0', type=int, default=5,
                        help='Periods per flux ramp.')
    parser.add_argument('--plot', '-p', action='store_true',
                        help='Make channel tracking plots')
    args = cfg.parse_args(parser)
    S = cfg.get_smurf_control(dump_configs=True)
    optimize_tracking(S, cfg, band=args.band, init_fracpp=None,
                      phi0_number=args.phi0_number,
                      reset_rate_khz=None, lms_gain=None,
                      make_plots=args.plot)
