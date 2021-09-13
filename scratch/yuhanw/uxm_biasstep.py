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
from scipy.interpolate import interp1d
import argparse
import time
import csv


parser = argparse.ArgumentParser()


parser.add_argument('--slot',type=int)
parser.add_argument('--temp',type=str)
parser.add_argument('--output_file',type=str)

args = parser.parse_args()
slot_num = args.slot
bath_temp = args.temp
out_fn = args.output_file


cfg = DetConfig()
cfg.load_config_files(slot=slot_num)
S = cfg.get_smurf_control()

for band in [0,1,2,3,4,5,6,7]:
    S.run_serial_gradient_descent(band);
    S.run_serial_eta_scan(band);
    S.set_feedback_enable(band,1) 
    S.tracking_setup(band,reset_rate_khz=cfg.dev.bands[band]['flux_ramp_rate_khz'],fraction_full_scale=cfg.dev.bands[band]['frac_pp'], make_plot=False, save_plot=False, show_plot=False, channel=S.which_on(band), nsamp=2**18, lms_freq_hz=None, meas_lms_freq=True,feedback_start_frac=cfg.dev.bands[band]['feedback_start_frac'],feedback_end_frac=cfg.dev.bands[band]['feedback_end_frac'],lms_gain=cfg.dev.bands[band]['lms_gain'])
    


fieldnames = ['bath_temp','bias_voltage', 'bias_line', 'band', 'data_path','note']
with open(out_fn, 'w', newline = '') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

bias_groups = [0,1,2,3,4,5,6,7,8,9,10,11]
for bias_v in [12,11,10,9,8,7,6,5,4,3,2,1,0]:
    bias_voltage = [bias_v,bias_v,bias_v,bias_v,bias_v,bias_v,bias_v,bias_v,bias_v,bias_v,bias_v,bias_v,0,0,0]
    S.set_tes_bias_bipolar_array(np.array(bias_voltage))
    time.sleep(60)
    for bias_index, bias_g in enumerate(bias_groups):

        step_size = 0.05

        print(S.get_tes_bias_bipolar_array())
        print('equivalent in low current mode:')
        # print(S.get_tes_bias_bipolar_array()*S.high_low_current_ratio)
       
      
        S.set_downsample_factor(1)
        print(S.get_sample_frequency())
        S.set_filter_disable(1)
       
        # Defining wave form
        signal = np.ones(2048)
        
        signal *= bias_v / (2*S._rtm_slow_dac_bit_to_volt) #defining the bias step level (lower step)
        signal[1024:] += step_size / (2*S._rtm_slow_dac_bit_to_volt) #defining the bias step level (upper step)
        ts = int(2/(6.4e-9 * 2048))
        S.set_rtm_arb_waveform_timer_size(ts, wait_done = True)
       
       
        S.play_tes_bipolar_waveform(bias_g,signal)
        datafile_self = S.stream_data_on()
        time.sleep(4)
        S.stream_data_off()
         
        S.set_rtm_arb_waveform_enable(0)
        S.set_filter_disable(0)
        S.set_downsample_factor(20)

        row = {}
        row['bath_temp'] = bath_temp
        row['data_path'] = datafile_self
        row['bias_voltage'] = bias_v
        row['bias_line'] = bias_g
        row['band'] = 'all'
        row['note'] = step_size
        with open(out_fn, 'a', newline = '') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow(row)
    bias_voltage = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    S.set_tes_bias_bipolar_array(np.array(bias_voltage))