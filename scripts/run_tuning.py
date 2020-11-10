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
        '--new-master-assignment', '--ma', type=bool, default=True,
        help='Whether to create a new master assignment file. This file '
             'defines the mapping between resonator frequency and channel '
             'number. Default True.')
    parser.add_argument('--amp-cut', '-a', type=float, default=0.1,
                        help='The fractional distance from the median value '
                        'to decide whether there is a resonance')

    args = cfg.parse_args(parser)
    S = cfg.get_smurf_control(dump_configs=True)

    if args.bands is None:
        args.bands = []
        bays = S.which_bays()
        if 0 in bays:
            args.bands.extend([0,1,2,3])
        if 1 in bays:
            args.bands.extend([4,5,6,7])

    find_and_tune_freq(S, cfg, args.bands,
                       new_master_assignment=args.new_master_assignment)

    # Writes new device config over existing one.
    print(f"Writing new config over {cfg.dev_file}")
    cfg.dev.dump(os.path.abspath(os.path.expandvars(cfg.dev_file)), clobber=True)
