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
pi = np.pi

if __name__=='__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--setup', action='store_true')
    parser.add_argument('--config-file', required=True)
    parser.add_argument('--epics-root', default='smurf_server_s2')

    parser.add_argument('--out-file')

    parser.add_argument('--band', type=int, default=2)
    parser.add_argument('--bias-group',type=int,default=1)
    parser.add_argument('--frac-pp',type=float, default = 0.28834)
    parser.add_argument('--temp',type=float, default = .100)
    # parser.add_argument('--noise-dur',type=float,)
    # parser.add_argument('--step-dur',type=float)
    # parser.add_argument('--step-size',type=float)
    # parser.add_argument('--bias-high',type=float)
    # parser.add_argument('--bias-low',type=float)
    # parser.add_argument('--bias-step',type=float)
    # parser.add_argument('--high-current-mode',action='store_true')
    
    args = parser.parse_args()
    
    S = pysmurf.client.SmurfControl(
            epics_root = args.epics_root,
            cfg_file = args.config_file,
            setup = args.setup,make_logfile=False
    )

    #Args to define:
    step_size = args.step_size #In volts
    band = args.band
    reset_rate_khz = S.get_flux_ramp_freq()*1e3
    downsampled = S.get_downsample_factor()
    fs = reset_rate_khz/downsampled
    # noise_dur = args.noise_dur
    # step_dur = args.step_dur
    bias_group = args.bias_group
    temp = args.temp
    nperseg = 2**16
    detrend = 'constant'
    plot_dir = S.plot_dir
    output_dir = S.output_dir

    # if args.high_current_mode:
    # 	bias_high = args.bias_high / S.high_low_current_ratio
    # 	bias_low = args.bias_low / S.high_low_current_ratio
    # 	bias_step = args.bias_step / S.high_low_current_ratio
    # else:
    # 	bias_high = args.bias_high
    # 	bias_low = args.bias_low
    # 	bias_step = args.bias_step

    # bias_points = np.arange(bias_high,bias_low-bias_step,-bias_step)

    bias_points = np.arange(10.5,3.5,-0.25)
    extra_points = np.array([13.5,12.5,11.5,2.5,1.5,0])
    bias_points = np.concatenate((bias_points,extra_points))
    bias_points = np.sort(bias_points)
    bias_points = np.flip(bias_points)

    print('Bias points: ' + str(bias_points))

    # Enable high current mode, and then set both DACs to 0 output
    S.set_tes_bias_high_current(1)
    S.set_rtm_slow_dac_volt(3,0)
    S.set_rtm_slow_dac_volt(4,0)

    # Overbias detectors, taking into account the high current mode factor
    ob_amp_low_current_mode = 13.5
    ob_amp_high_current_mode = ob_amp_low_current_mode / S.high_low_current_ratio

    S.set_rtm_slow_dac_volt(4,ob_amp_high_current_mode)

    for b_bias in bias_points:
        #Step the bias down to some value in the transition
        
        bias_target_low_current_mode = b_bias
        print(bias_target_low_current_mode/S.high_low_current_ratio)
        
        cur_bias = S.get_rtm_slow_dac_volt(4) 
        print(cur_bias)
        b_step = np.arange(cur_bias,
                bias_target_low_current_mode/S.high_low_current_ratio,
                -0.1/S.high_low_current_ratio)
        #print(b_step)
        for b in b_step:
            S.set_rtm_slow_dac_volt(4,b)
            print('Bias set to: ' + str(S.get_rtm_slow_dac_volt(4)))
            time.sleep(5)
        time.sleep(120)

        #If not set it to the right value
        S.set_rtm_slow_dac_volt(4,b_bias/S.high_low_current_ratio) 
        print('Bias set to: ' + str(S.get_rtm_slow_dac_volt(4)))
        
        #Once temp is stable do some bias steps
        step_size = 0.004
        cur_bias = S.get_rtm_slow_dac_volt(4) 
        print('Starting Bias Step Data')
        datfile = S.stream_data_on() 
        noise_dur = 1
        step_dur = 0.2
        n_steps = 15
        time.sleep(noise_dur) 
        for i in range(n_steps):
            S.set_rtm_slow_dac_volt(4,cur_bias+step_size) 
            time.sleep(step_dur) 
            S.set_rtm_slow_dac_volt(4,cur_bias) 
            time.sleep(step_dur)
        time.sleep(1) 
        S.stream_data_off() 
        print('Stopping Bias Step Data')

        out_file = args.out_file
        print(f'Writing to file {out_file}')
        with open(out_file, 'a') as fname:
            fname.write(f'T = {temp} mK, Bias Point: {b}, Datafile: {datfile}, plot_dir: {plot_dir}\n')

