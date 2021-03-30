import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pysmurf.client
import argparse
import numpy as np
import os
import time
import glob
from scipy import signal
import scipy.optimize as opt
pi = np.pi



if __name__=='__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--setup', action='store_true')
    parser.add_argument('--config-file', required=True)
    parser.add_argument('--epics-root', default='smurf_server_s2')

    parser.add_argument('--out-file')

    parser.add_argument('--band', type=int, required=True)
    parser.add_argument('--bias-group',type=int,default=1)

    parser.add_argument('--step-size',type=float)

    parser.add_argument('--bias-high',type=float)
    parser.add_argument('--bias-low',type=float)
    parser.add_argument('--bias-step',type=float)
    parser.add_argument('--overbias-voltage',type=float,default=19.9)

    args = parser.parse_args()
    
    # Initialize pysmurf object
    S = pysmurf.client.SmurfControl(
            epics_root = args.epics_root,
            cfg_file = args.config_file,
            setup = args.setup,make_logfile=False
    )

    # Basic setup info
    band = args.band
    bg = args.bias_group

    # Size of bias step. Recommended is 0.1 pW height at 50% R_n
    step_size = args.step_size #In volts

    bias_high = args.bias_high
    bias_low = args.bias_low
    bias_step = args.bias_step
    ob_volt = args.overbias_voltage

    # Output information
    out_file = args.out_file
    output_dir = S.output_dir

    with open(out_file, 'a') as fname:
        fname.write('# Bias,Datafile,Output_Dir,Sample_rate')

    # Starting downsample and filter information
    start_downsample = S.get_downsample_factor()
    a = S.get_filter_a(band)
    b = S.get_filter_b(band)

    # No more downsampling
    S.set_downsample_factor(1)
    fs = S.get_flux_ramp_freq()*1e3/S.get_downsample_factor()

    # Disable downsample filter
    S.set_filter_disable(1)

    # Set things to high current mode, and set bias voltage to zero
    S.set_tes_bias_high_current(bg)
    S.set_tes_bias_bipolar(bg,0.0)

    # Overbias detectors
    S.set_tes_bias_bipolar(bg,ob_volt/S.high_low_current_ratio)

    for b_bias in np.arange(bias_high,bias_low,-bias_step):
        
        # Step the bias down to some value in the transition
        bias_target_low_current_mode = b_bias
        print(bias_target_low_current_mode/S.high_low_current_ratio)
        
        cur_bias = S.get_tes_bias_bipolar(bg) 
        print(cur_bias)
        b_step = np.arange(cur_bias,
                bias_target_low_current_mode/S.high_low_current_ratio,
                -0.1/S.high_low_current_ratio)
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
        for i in range(2):
            datfile = S.stream_data_on()
            time.sleep(1)
            S.play_tes_bipolar_waveform(bg,sig)
            time.sleep(5)

            # Instead of using the pysmurf stop function, set to the original dc value
            S.set_rtm_arb_waveform_enable(0)
            S.set_tes_bias_bipolar(bg,cur_bias)
            time.sleep(1)
            S.stream_data_off()

        # Save datfile info

        with open(out_file, 'a') as fname:
            fname.write(f'{b},{datfile},{output_dir},{fs}\n')
        
        # Sleep for 60 seconds before ending
        time.sleep(60)

    # Turn on downsample filter again
    S.set_filter_disable(0)
