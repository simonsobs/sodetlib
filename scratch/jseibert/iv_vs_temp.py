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

from sodetlib.smurf_funcs import tracking_quality
import sodetlib.smurf_funcs.optimize_params as op
import sodetlib.util as su
from sodetlib.smurf_funcs.det_ops import take_iv

from scipy.interpolate import interp1d

if __name__=='__main__':
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--out-file')

    #parser.add_argument('--band', type=int,nargs='+', default=[0,1,2,3,4,5,6,7])
    #parser.add_argument('--bias-group',type=int,nargs='+',default=[0,1,2,3,4,5,6,7,8,9,10,11])
    parser.add_argument('--chans',type=int,nargs='+',default=None)
    parser.add_argument('--ob-volt',type=float,default=19.9)
    parser.add_argument('--step-volt',type=float,default=0.1)
    parser.add_argument('--temp',type=float)
    
    # Initialize pysmurf object
    cfg = DetConfig()
    cfg.load_config_files(slot=3)
    S = cfg.get_smurf_control()
    
    # Parse args
    args = parser.parse_args()
    
    #bands = args.band
    #bias_group = args.bias_group
    bands = [0,1,2,3,4,5,6,7]
    bias_group = [0,1,2,3,4,5,6,7,8,9,10,11]
    ob_volt = args.ob_volt
    step_size = args.step_volt
    good_resp_chans = args.chans
    temp = args.temp
    
    ucs = [16,18,18,24,18,14,26,26]; dcs = [12,14,14,30,8,30,30,30]

    # Set attenuators to optimal values
    #S.set_att_uc(band,8)
    #S.set_att_dc(band,8)

    tk = {}
    for band in bands:
        S.set_att_uc(band,ucs[band])
        S.set_att_dc(band,dcs[band])
        S.find_freq(band, tone_power=12, make_plot=True, show_plot=False, save_plot=True, amp_cut=0.05,grad_cut = 0.005)
        S.setup_notches(band, tone_power=12, new_master_assignment=True, min_offset = 0.05)
        for _ in range(2):
            S.run_serial_gradient_descent(band)
            S.run_serial_eta_scan(band)
        tk[band] = su.get_tracking_kwargs(S,cfg,band)
        tk[band]['nsamp'] = 2**18
        x = tracking_quality(S, cfg, band, show_plots=True,tracking_kwargs=tk[band])
    # Tune resonators
    #S.find_freq(band, tone_power=12, make_plot=True, save_plot=True, show_plot=True)
    #S.setup_notches(band,new_master_assignment=True) # should this be which master assignment?
    
    
    #for _ in range(2):
        #S.run_serial_gradient_descent(band)
        #S.run_serial_eta_scan(band)    
    #tk = su.get_tracking_kwargs(S,cfg,band)
    #S.tracking_setup(band,**tk)
    
    #if good_resp_chans is None:
       # good_resp_chans = S.which_on(band)
    
    #for chan in S.which_on(band):
        #if chan in good_resp_chans:
            #continue
        #else:
        #S.channel_off(band,chan)
    
    # Take IVs
    iv_info_fp = take_iv(S,bias_groups=bias_group,bias_high=19.9,bias_step=step_size,cool_wait=120,high_current_mode=False,overbias_voltage=ob_volt,cool_voltage=8.0)
    with open(args.out_file, 'a') as fname:
        fname.write(f'T = {temp} mK, outdir: {S.output_dir}, tunefile: {S.tune_file}, iv_info: {iv_info_fp}\n')
    
