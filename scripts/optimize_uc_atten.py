import argparse
import os
import matplotlib
matplotlib.use('Agg')

from sodetlib.det_config import DetConfig
from sodetlib.smurf_funcs.optimize_params import optimize_uc_atten


if __name__ == '__main__':
    cfg = DetConfig()
    parser = argparse.ArgumentParser()
    parser.add_argument('--band', '-b', type=int, required=True,
                        help='band (must be in range [0,7])')
    parser.add_argument('--meas-time', '-m', type=float, default=10.,
                        help='Time (sec) to take data for noise PSD. If not'
                             'specified, defaults to 30 sec.')
    parser.add_argument('--drive', '-d', type=int,
                        help="Drive to optimize uc_atten")
    parser.add_argument('--plot', '-p', action='store_true',
                        help='Make tracking plots')
    parser.add_argument('--setup-notches', '--sc', action='store_true',
                       help='Run setup notches after each atten step')

    args = cfg.parse_args(parser)
    S = cfg.get_smurf_control(dump_configs=True)

    optimize_uc_atten(S, cfg, args.band, meas_time=args.meas_time,
                      run_setup_notches=args.setup_notches)

    cfg_path = os.path.abspath(os.path.expandvars(cfg.dev_file))
    cfg.dev.dump(cfg_path, clobber=True)
