from sodetlib.smurf_funcs.optimize_params import optimize_tracking
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
    parser.add_argument('--phi0-number', type=int,
                        help='Periods per flux ramp.')
    parser.add_argument('--no-relock', '-n', action='store_false',
                        dest='relock')

    args = cfg.parse_args(parser)
    S = cfg.get_smurf_control(dump_configs=True)
    optimize_tracking(S, cfg, band=args.band, init_fracpp=0.2,
                      phi0_number=args.phi0_number,
                      relock=args.relock)
