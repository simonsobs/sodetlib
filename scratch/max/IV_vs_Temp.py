import matplotlib
matplotlib.use('Agg')

import pysmurf.client
import argparse
import numpy as np
import os
import time
import glob




if __name__=='__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--setup', action='store_true')
    parser.add_argument('--config-file', required=True)
    parser.add_argument('--epics-root', default='smurf_server_s2')

    parser.add_argument('--out-file')

    parser.add_argument('--band', type=int, default=2)
    parser.add_argument('--subband',type=int,nargs = '+')
    parser.add_argument('--bias-group',type=int,default=1)
    parser.add_argument('--tick-resp-chans',type=int,nargs='+',default=None)
    parser.add_argument('--ob-volt',type=float,default=10)
    parser.add_argument('--step-volt',type=float,default=0.05)
    parser.add_argument('--frac-pp',type=float)
    parser.add_argument('--temp',type=float)
    
    args = parser.parse_args()
    
    S = pysmurf.client.SmurfControl(
            epics_root = args.epics_root,
            cfg_file = args.config_file,
            setup = args.setup,make_logfile=False
    )

    print("Plots in directory: ",S.plot_dir)
    print(f"subbands = {args.subband}")
    
    #Initialize the band for our mux chip
    band = args.band
    bias_group = args.bias_group
    ob_amp = args.ob_volt
    step_size = args.step_volt
    frac_pp = args.frac_pp
    sbs_on = args.subband
    good_resp_chans = args.tick_resp_chans
    temp = args.temp
    if good_resp_chans == None:
        good_resp_chans = S.which_on(band)
    
    #S.find_freq(band, subband=sbs_on,drive_power = 12,make_plot=True,show_plot=False,save_plot = True)
    #S.load_tune('/data/smurf_data/tune/1588116270_tune.npy')
    #S.load_tune('/data/smurf_data/tune/1592250732_tune.npy')
    S.load_tune('/data/smurf_data/tune/1592250732_tune.npy')
    S.setup_notches(band,drive=12,new_master_assignment=False)
    for chan in S.which_on(band):
        if chan in good_resp_chans:
            continue
        else:
            S.channel_off(band,chan)
    S.run_serial_gradient_descent(band)
    S.run_serial_eta_scan(band)
    S.tracking_setup(band,reset_rate_khz=4,lms_freq_hz=20000,fraction_full_scale = frac_pp,make_plot=True,show_plot=False,channel=S.which_on(band),nsamp=2**18,feedback_start_frac=0.02,feedback_end_frac=0.94)
    #tunefile = S.tune_file
    plotdir = S.plot_dir
    iv_file = S.slow_iv_all(band = band, channels = S.which_on(band),bias_groups=np.asarray([bias_group]), overbias_voltage=ob_amp,bias_high=11.0, bias_step=step_size,wait_time=1, high_current_mode=False,overbias_wait=15, cool_wait=300,phase_excursion_min=.1,bias_low=3.5) 
    print(iv_file)
    #iv_file = iv_file.replace('_raw_data','')
    

    with open(args.out_file, 'a') as fname:
        fname.write(f'T = {temp} mK, plotdir : {plotdir}, iv_file : {iv_file}\n')

    


