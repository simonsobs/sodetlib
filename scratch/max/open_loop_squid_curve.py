import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pysmurf.client
import argparse
import numpy as np

from sodetlib.det_config import DetConfig
from sodetlib.smurf_funcs import take_squid_open_loop


if __name__=='__main__':
    cfg = DetConfig()
    parser = argparse.ArgumentParser()
    parser.add_argument('--bands', '-b', type=int, nargs = '+',required=True,
                        help='band (must be in range [0,7])')
    parser.add_argument('--wait-time', type=float, default=0.1,
                        help='Time to wait between flux steps in seconds')
    parser.add_argument('--Npts', type=int,default = 3,
                        help='Number of points to average for each point')
    parser.add_argument('--NPhi0s',type=int,default = 4,
                        help = 'Number of periods in your squid curve')
    parser.add_argument('--Nsteps',type=int,default = 500,
                        help='Number of points in your squid curve')
    parser.add_argument('--relock',action='store_false',
                        help = 'Will run relock if True otherwise will not')
    
    args = cfg.parse_args(parser)
    print(args.relock)
    S = cfg.get_smurf_control(dump_configs=True)
    take_squid_open_loop(S=S, cfg=cfg, bands = args.bands,
                         wait_time = args.wait_time, Npts = args.Npts,
                         NPhi0s = args.NPhi0s,Nsteps = args.Nsteps,
                         relock=args.relock, frac_pp=None, lms_freq=None,
                         reset_rate_khz=None,lms_gain=None)
