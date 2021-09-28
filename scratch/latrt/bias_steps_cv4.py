import matplotlib
matplotlib.use('Agg')

import pysmurf.client
import argparse
import numpy as np
import os
import time
import glob
from sodetlib.smurf_funcs.smurf_ops import take_g3_data, stream_g3_off, stream_g3_on
from sodetlib.det_config  import DetConfig
import numpy as np
import sodetlib.smurf_funcs.det_ops as do
import sodetlib.analysis.det_analysis as da


if __name__=='__main__':
    parser = argparse.ArgumentParser()

    #parser.add_argument('--band', type=int, required=True)
    #parser.add_argument('--bias-group',type=int,default=0)

    parser.add_argument('--step-size',type=float)

    #parser.add_argument('--bias-high',type=float)
    #parser.add_argument('--bias-low',type=float)
    #parser.add_argument('--bias-step',type=float)
    #parser.add_argument('--overbias-voltage',type=float,default=19.9)
    parser.add_argument('--rfracs', nargs='+', type=float)
    parser.add_argument('--bias-groups', nargs='+', type=int)
    args = parser.parse_args()
    
    # Initialize pysmurf object
    cfg = DetConfig()
    cfg.load_config_files(slot=2)
    S = cfg.get_smurf_control()
    
    # Basic setup info
    #band = args.band
    #bg = args.bias_group
    bgs = args.bias_groups
    rfracs = args.rfracs

    iv_analyze_fp = '/data/smurf_data/20210901/crate1slot2/1630529854/outputs/1630533338_iv_analyze.npy'
    bg_map_fp = '/data/smurf_data/bias_group_maps/1630532841_bg_map.npy'
    # Size of bias step. Recommended is 0.1 pW height at 50% R_n
    step_size = args.step_size #In volts

    #bias_high = args.bias_high
    #bias_low = args.bias_low
    #bias_step = args.bias_step
    #ob_volt = args.overbias_voltage

    ctime = S.get_timestamp()
    out_file = f'/data/sodetlib_data/{ctime}_bias_steps.txt'

    with open(out_file, 'a') as fname:
        fname.write('# Rfrac,Bias_Group,Datafile,Output_Dir,Sample_rate,Start,Stop\n')

    # Starting downsample and filter information
    start_downsample = S.get_downsample_factor()

    # No more downsampling
    S.set_downsample_factor(1)
    fs = S.get_flux_ramp_freq()*1e3/S.get_downsample_factor()

    # Disable downsample filter
    S.set_filter_disable(1)

    # Set things to high current mode, and set bias voltage to zero
    for bg in bgs:
        S.set_tes_bias_bipolar(bg,0.0)

    # Convert step size to high current mode
    step_size /= S.high_low_current_ratio
    
    # Overbias detectors
   # S.overbias_tes(bias_group = bg, tes_bias = bias_high/S.high_low_current_ratio,overbias_wait=5,overbias_voltage=ob_volt,high_current_mode=True)
    for bg in bgs: 
        for rfrac in rfracs:
            chosen_biases_fp = da.bias_points_from_rfrac(S, cfg, iv_analyze_fp, bg_map_fp, rfrac=rfrac,
                                           bias_groups=[bg])

            do.bias_detectors_from_sc(S, chosen_biases_fp, high_current_mode=True)
            time.sleep(180)
            print(S.get_tes_bias_bipolar_array())
           
            # Once temp is stable do some bias steps

            # Set up waveform to pass down bias line
            sig = np.ones(2048)
            cur_bias = S.get_tes_bias_bipolar(bg)
            sig *= cur_bias / (2*S._rtm_slow_dac_bit_to_volt)
            sig[1024:] += step_size / (2*S._rtm_slow_dac_bit_to_volt)
            ts = int(0.75/(6.4e-9 * 2048))
            S.set_rtm_arb_waveform_timer_size(ts, wait_done = True)

            start = S.get_timestamp()
            g3_id = stream_g3_on(S)
            time.sleep(1)
            S.play_tes_bipolar_waveform(bg,sig)
            time.sleep(5)

            # Instead of using the pysmurf stop function, set to the original dc value
            S.set_rtm_arb_waveform_enable(0)
            S.set_tes_bias_bipolar(bg,cur_bias)
            time.sleep(1)
            stream_g3_off(S)
            stop = S.get_timestamp()

            # Save datfile info

            with open(out_file, 'a') as fname:
                fname.write(f'{rfrac},{bg},{g3_id},{S.output_dir},{fs},{start},{stop}\n')
            
            # Sleep for 15 seconds before next step
            time.sleep(15)

    # Turn on downsample filter again and downsampling
    S.set_filter_disable(0)
    S.set_downsample_factor(20)
