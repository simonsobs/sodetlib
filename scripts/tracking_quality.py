import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import argparse
import numpy as np
from sodetlib.det_config import DetConfig
import sodetlib.smurf_funcs.smurf_ops as op
from sodetlib.util import cprint
from pysmurf.client.util.pub import set_action


if __name__ == '__main__':
    cfg = DetConfig()
    parser = argparse.ArgumentParser()
    parser.add_argument('--bands', '-b', type=int, nargs='+', required=True)
    parser.add_argument('--threshold', '-t', type=float, default=0.9)
    parser.add_argument('--plots', '-p', action='store_true')
    args = cfg.parse_args(parser)
    S = cfg.get_smurf_control(dump_configs=True)

    for b in args.bands:
        rs, f, df, sync = op.tracking_quality(
            S, cfg, b, make_channel_plots=args.plots, r_thresh=args.threshold
        )
        nchans = len(S.which_on(b))
        good_chans = np.where(rs > args.threshold)[0]
        cprint(f"{len(good_chans)} / {nchans} have passed on band {b}",
               True)
        cprint(f"Good chans:\n{good_chans}", True)
