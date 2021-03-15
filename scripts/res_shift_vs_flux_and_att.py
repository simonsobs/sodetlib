import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pysmurf.client
import argparse
import numpy as np
import pickle as pkl

from sodetlib.det_config import DetConfig
from sodetlib.smurf_funcs.smurf_ops import res_shift_vs_flux_bias, res_shift_vs_uc_att


if __name__=='__main__':
    cfg = DetConfig()
    parser = argparse.ArgumentParser()
    #parser.add_argument('--band', '-b', type=int, required=True,
    #                    help='band (must be in range [0,7])')
    bands = [2]
    parser.add_argument('--tunefile')
    #tunefile = '/data/smurf_data/tune/1607469461_tune.npy'
    parser.add_argument('--frac-pp-steps', nargs='+', type=float)
    #frac_pp_steps = np.linspace(0,2*0.28/5,20)
    parser.add_argument('--uc-atts',nargs = '+',type = int)
    #uc_atts = np.arange(30,-2,-2)
    args = cfg.parse_args(parser)
    tunefile = args.tunefile
    frac_pp_steps = args.frac_pp_steps
    uc_atts = args.uc_atts
    S = cfg.get_smurf_control(dump_configs=True)
    output_dir = S.output_dir
    outdict_att = res_shift_vs_uc_att(S = S, uc_atts = uc_atts, bands = bands,
                                      tunefile = tunefile)
    ctime = S.get_timestamp()
    pkl.dump(outdict_att,open(f'{output_dir}/{ctime}_res_shift_vs_uc_att.pkl','wb'))
    outdict_flux = res_shift_vs_flux_bias(S = S, frac_pp_steps =frac_pp_steps,
                                            bands = bands, tunefile = tunefile)
    
    ctime = S.get_timestamp()
    pkl.dump(outdict_flux,open(f'{output_dir}/{ctime}_res_shift_vs_flux_bias.pkl','wb'))


    
