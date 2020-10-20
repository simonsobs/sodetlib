import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pysmurf.client
import argparse
import numpy as np

from sodetlib.det_config import DetConfig
from sodetlib.smurf_funcs.optimize_params import optimize_tracking


if __name__=='__main__':
    cfg = DetConfig()
    parser = argparse.ArgumentParser()
    parser.add_argument('--band', '-b', type=int, required=True,
                        help='band (must be in range [0,7])')
    parser.add_argument('--phi0-number', type=float, required=True,
                        help='Periods per flux ramp.')
    args = cfg.parse_args(parser)
    S = cfg.get_smurf_control(dump_configs=True)
    optimize_tracking(S, cfg, band= args.band, init_fracpp = 0.2,
						phi0_number = args.phi0_number,
						reset_rate_khz = None, lms_gain = None)

