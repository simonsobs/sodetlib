import matplotlib
matplotlib.use('Agg') # I put it in backend
import argparse
import os
import numpy as np

from sodetlib.det_config import DetConfig
from sodetlib.smurf_funcs import find_and_tune_freq


if __name__ == '__main__':
    cfg = DetConfig()

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--bands', nargs='+', type=int,
        help='input bands to tune as ints, separated by spaces. Must be in '
             'range [0,7]. Defaults to tuning all bands.')
    parser.add_argument(
        '--subband', '-s', nargs='*', type=int, help="List of subbands to scan")
    parser.add_argument(
        '--plotname-append', type=str, default='',
        help="Appended to the default plot filename. Default is ''.")
    parser.add_argument(
        '--new-master-assignment', '--ma', type=bool, default=True,
        help='Whether to create a new master assignment file. This file '
             'defines the mapping between resonator frequency and channel '
             'number. Default True.')

    args = cfg.parse_args(parser)
    S = cfg.get_smurf_control(dump_configs=True)

    if not args.bands:
        args.bands = list(range(8))

    find_and_tune_freq(S, cfg, args.bands, subband=args.subband,
                       plotname_append=args.plotname_append,
                       new_master_assignment=args.new_master_assignment)

    # Writes new device config over existing one.
    print(f"Writing new config over {cfg.dev_file}")
    cfg.dev.dump(os.path.abspath(os.path.expandvars(cfg.dev_file)), clobber=True)
