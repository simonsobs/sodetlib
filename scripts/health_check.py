import matplotlib
import argparse
import os
matplotlib.use('Agg')

from sodetlib.det_config import DetConfig
from sodetlib.smurf_funcs import health_check

if __name__ == '__main__':
    cfg = DetConfig()
    parser = argparse.ArgumentParser()

    # Custom arguments for this script
    parser.add_argument('--bay0', action='store_true',
                        help='Whether or not bay0 is active')
    parser.add_argument('--bay1', action='store_true',
                        help='Whether or not bay1 is active')

    args = cfg.parse_args(parser)
    S = cfg.get_smurf_control(dump_configs=True)

    # Put your script calls here
    health_check(S, cfg, bay0=args.bay0, bay1=args.bay1)

    out_file = os.path.abspath(os.path.expandvars(cfg.dev_file))
    print(f"Writing new device config to {out_file}")
    cfg.dev.dump(out_file, clobber=True)
