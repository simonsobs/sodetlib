import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pysmurf.client
import argparse
import numpy as np
import pickle as pkl

from sodetlib.det_config import DetConfig
from sodetlib.smurf_funcs.smurf_ops import res_shift


if __name__=='__main__':
    cfg = DetConfig()
    parser = argparse.ArgumentParser()
    #parser.add_argument('--band', '-b', type=int, required=True,
    #                    help='band (must be in range [0,7])')
    bands = [2]
    parser.add_argument('--setup', action = 'store_true',
                        help='If flag is enabled will relock')
    parser.add_argument('--temp')
    parser.add_argument('--out-file')
    parser.add_argument('--tunefile')
    args = cfg.parse_args(parser)
    temp = args.temp
    cfg.load_config_files(slot = 2)
    S = cfg.get_smurf_control(dump_configs=True, make_logfile=False)
    if args.setup:
        S.load_tune(args.tunefile)
        for band in bands:
            S.relock(band)
    else:
        S.load_tune()
    out_dict = res_shift(S, bands)
    fpath = f'{S.output_dir}/{S.get_timestamp()}_res_shift_dict.pkl'
    pkl.dump(out_dict, open(fpath,'wb'))

    with open(args.out_file, 'a') as fname:
        fname.write(f'T = {temp} mK, filepath : {fpath}\n')

