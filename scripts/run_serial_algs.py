"""
This is a utility script for the OCS pysmurf controller that runs a combination
of serial grad descent, serial eta scan, and tracking setup for a set of bands.
It uses tracking args based on the device config file, so make sure that
contains the correct parameters before running.
"""

import matplotlib
matplotlib.use('Agg')
import argparse

from sodetlib.det_config import DetConfig
from sodetlib.util import get_tracking_kwargs

if __name__ == '__main__':
    cfg = DetConfig()

    parser = argparse.ArgumentParser()
    parser.add_argument('bands', nargs='+', type=int)
    parser.add_argument("--no-tracking", action='store_true')
    parser.add_argument("--no-grad", action='store_true')
    parser.add_argument("--no-eta", action='store_true')

    parser.add_argument('--frac-pp', '--fpp', type=float)
    parser.add_argument('--lms-freq', type=float)

    args = cfg.parse_args(parser)
    S = cfg.get_smurf_control(dump_configs=True)

    for b in args.bands:
        print(f"Running serial algs for band {b}")
        if not args.no_grad:
            S.run_serial_gradient_descent(b)
        if not args.no_eta:
            S.run_serial_eta_scan(b)
        if not args.no_tracking:
            tk = get_tracking_kwargs(S, cfg, b, kwargs=None)
            if args.frac_pp is not None:
                tk['fraction_full_scale'] = args.frac_pp
            if args.lms_freq is not None:
                tk['lms_freq'] = args.lms_freq
            S.tracking_setup(b, **tk)

