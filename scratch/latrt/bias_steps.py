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



if __name__=='__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--band', type=int, required=True)
    parser.add_argument('--bias-group',type=int,default=0)

    parser.add_argument('--step-size',type=float)

    parser.add_argument('--bias-high',type=float)
    parser.add_argument('--bias-low',type=float)
    parser.add_argument('--bias-step',type=float)
    parser.add_argument('--overbias-voltage',type=float,default=19.9)

    args = parser.parse_args()
    
    # Initialize pysmurf object
    cfg = DetConfig()
    cfg.load_config_files(slot=2)
    S = cfg.get_smurf_control()
    
    # Basic setup info
    band = args.band
    bg = args.bias_group

    # Size of bias step. Recommended is 0.1 pW height at 50% R_n
    step_size = args.step_size #In volts

    bias_high = args.bias_high
    bias_low = args.bias_low
    bias_step = args.bias_step
    ob_volt = args.overbias_voltage

    ctime = S.get_timestamp()
    out_file = f'/data/sodetlib_data/{ctime}_bias_steps.txt'

    with open(out_file, 'a') as fname:
        fname.write('# Bias,Datafile,Output_Dir,Sample_rate,Start,Stop\n')

    # Starting downsample and filter information
    start_downsample = S.get_downsample_factor()

    # No more downsampling
    S.set_downsample_factor(1)
    fs = S.get_flux_ramp_freq()*1e3/S.get_downsample_factor()

    # Disable downsample filter
    S.set_filter_disable(1)

    # Set things to high current mode, and set bias voltage to zero
    S.set_tes_bias_high_current(bg)
    S.set_tes_bias_bipolar(bg,0.0)

    # Convert step size to high current mode
    step_size /= S.high_low_current_ratio
    
    # Overbias detectors
    S.overbias_tes(bias_group = bg, tes_bias = bias_high/S.high_low_current_ratio,overbias_wait=5,overbias_voltage=ob_volt,high_current_mode=True)
    
    for b_bias in np.arange(bias_high,bias_low,-bias_step):
        
        # Step the bias down to some value in the transition
        bias_target_low_current_mode = b_bias
        print(bias_target_low_current_mode/S.high_low_current_ratio)
        
        cur_bias = S.get_tes_bias_bipolar(bg) 
        print(cur_bias)
        b_step = np.arange(bias_target_low_current_mode/S.high_low_current_ratio,
                   cur_bias,
            0.1/S.high_low_current_ratio)
        b_step = np.flip(b_step)
        print(b_step)
        for b in b_step:
            S.set_tes_bias_bipolar(bg,b)
            time.sleep(1)
            print(S.get_tes_bias_bipolar(bg))
            time.sleep(10)
        time.sleep(180)
        
        # Check that its set at the bias you commanded it to
        print(S.get_tes_bias_bipolar(bg))
        print(b_bias/S.high_low_current_ratio)

        # If not set it to the right value
        S.set_tes_bias_bipolar(bg,b_bias/S.high_low_current_ratio) 
        print(S.get_tes_bias_bipolar(bg))
        
        # Once temp is stable do some bias steps

        # Set up waveform to pass down bias line
        sig = np.ones(2048)
        cur_bias = S.get_tes_bias_bipolar(bg)
        sig *= cur_bias / (2*S._rtm_slow_dac_bit_to_volt)
        sig[1024:] += step_size / (2*S._rtm_slow_dac_bit_to_volt)
        ts = int(0.75/(6.4e-9 * 2048))
        S.set_rtm_arb_waveform_timer_size(ts, wait_done = True)

        # For right now, we take each set of bias steps twice, but only keep the second one
        # This is a remnant from an old pysmurf bug, but it isn't hurting anything except
        # storage space on the smurf-srv. Can probably take this out now
        for i in range(1):
            start = S.get_timestamp()
            g3_id = S.stream_g3_on(S)
            time.sleep(1)
            S.play_tes_bipolar_waveform(bg,sig)
            time.sleep(5)

            # Instead of using the pysmurf stop function, set to the original dc value
            S.set_rtm_arb_waveform_enable(0)
            S.set_tes_bias_bipolar(bg,cur_bias)
            time.sleep(1)
            S.stream_data_off()
            stop = S.get_timestamp()

        # Save datfile info

        with open(out_file, 'a') as fname:
            fname.write(f'{b_bias},{g3_id},{S.output_dir},{fs},{start},{stop}\n')
        
        # Sleep for 15 seconds before next step
        time.sleep(15)

    # Turn on downsample filter again and downsampling
    S.set_filter_disable(0)
    S.set_downsample_factor(20)
