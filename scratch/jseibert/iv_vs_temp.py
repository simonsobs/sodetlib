# updating IV_vs_temp to work for new sodetlib functions and updated pysmurf functions
# Joe Seibert
# 3/11/21

import matplotlib
matplotlib.use('Agg')

import pysmurf.client
import argparse
import numpy as np
import os
import time
import glob

from sodetlib.det_config  import DetConfig
import numpy as np

import sodetlib.smurf_funcs.optimize_params as op
import sodetlib.util as su
from sodetlib.smurf_funcs.det_ops import take_iv

from scipy.interpolate import interp1d

if __name__=='__main__':
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--out-file')

    parser.add_argument('--band', type=int, default=2)
    parser.add_argument('--bias-group',type=int,default=0)
    parser.add_argument('--chans',type=int,nargs='+',default=None)
    parser.add_argument('--ob-volt',type=float,default=19.9)
    parser.add_argument('--step-volt',type=float,default=0.1)
    parser.add_argument('--temp',type=float)
    
    # Initialize pysmurf object
    cfg = DetConfig()
    cfg.load_config_files(slot=2)
    S = cfg.get_smurf_control()
    
    # Parse args
    args = parser.parse_args()
    
    band = args.band
    bias_group = args.bias_group
    ob_volt = args.ob_volt
    step_size = args.step_volt
    good_resp_chans = args.chans
    temp = args.temp
    

    # Set attenuators to optimal values
    S.set_att_uc(band,8)
    S.set_att_dc(band,8)
    
    # Tune resonators
    S.find_freq(band, tone_power=12, make_plot=True, save_plot=True, show_plot=True)
    S.setup_notches(band,new_master_assignment=True) # should this be which master assignment?
    for _ in range(2):
        S.run_serial_gradient_descent(band)
        S.run_serial_eta_scan(band)    
    tk = su.get_tracking_kwargs(S,cfg,band)
    S.tracking_setup(band,**tk)
    
    if good_resp_chans is None:
        good_resp_chans = S.which_on(band)
    
    for chan in S.which_on(band):
        if chan in good_resp_chans:
            continue
        else:
            S.channel_off(band,chan)
    
    # Take IVs
    iv_info_fp = take_iv(S,bias_groups=[0],bias_high=19.9,bias_step=step_size,cool_wait=120,high_current_mode=False,overbias_voltage=ob_volt,cool_voltage=8.0)
    
    with open(args.out_file, 'a') as fname:
        fname.write(f'T = {temp} mK, outdir : {S.output_dir}, tunefile : {S.tune_file} iv_info : {iv_info_fp}\n')
    