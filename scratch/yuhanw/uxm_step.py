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

fieldnames = ['bath_temp','bias_voltage', 'bias_line', 'band', 'data_path','type']
with open(out_fn, 'w', newline = '') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
 
for band in [0,1,2,3,4,5,6,7]:
    S.run_serial_gradient_descent(band);
    S.run_serial_eta_scan(band);
    S.set_feedback_enable(band,1) 
    S.tracking_setup(band,reset_rate_khz=cfg.dev.bands[band]['flux_ramp_rate_khz'],fraction_full_scale=cfg.dev.bands[band]['frac_pp'], make_plot=False, save_plot=False, show_plot=False, channel=S.which_on(band), nsamp=2**18, lms_freq_hz=None, meas_lms_freq=True,feedback_start_frac=cfg.dev.bands[band]['feedback_start_frac'],feedback_end_frac=cfg.dev.bands[band]['feedback_end_frac'],lms_gain=cfg.dev.bands[band]['lms_gain'])
    

bias_groups = [0,1,2,3,4,5,6,7,8,9,10,11]
target_noise_v = np.arange(12,-0.5,-0.5)
S.overbias_tes_all(bias_groups = bias_groups, overbias_wait=1, tes_bias= 12, cool_wait= 60, high_current_mode=False, overbias_voltage= 12)
for bias_index, bias_v in enumerate(target_noise_v):
    S.set_tes_bias_bipolar_array([bias_v,bias_v,bias_v,bias_v,bias_v,bias_v,bias_v,bias_v,bias_v,bias_v,bias_v,bias_v,0,0,0])
    time.sleep(60)
    datafile_self = S.stream_data_on()
    time.sleep(120)
    S.stream_data_off()

    row = {}
    row['data_path'] = datafile_self
    row['bias_voltage'] = bias_v
    row['type'] = 'noise'
    row['bias_line'] = 'all'
    row['band'] = 'all'
    row['bath_temp'] = bath_temp
    with open(out_fn, 'a', newline = '') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow(row)













# for bias_gp in [0,1,2,3,4,5,6,7,8,9,10,11]:
#         row = {}
#         row['bath_temp'] = bath_temp
#         row['bias_line'] = bias_gp
#         row['band'] = 'all'
             
#         print(f'Taking IV on bias line {bias_gp}, all band')
          
 
#         row['data_path'] = S.run_iv(bias_groups = [bias_gp], wait_time=0.001, bias_high=16, bias_low=0, bias_step = 0.025, overbias_voltage=18, cool_wait=150, high_current_mode=False, make_plot=False, save_plot=True, cool_voltage = 8)
     
          
#         with open(out_fn, 'a', newline = '') as csvfile:
#             writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#             writer.writerow(row)
 
 
# row = {}
# row['bath_temp'] = bath_temp
# row['bias_line'] = 'all'
# row['band'] = 'all'
 
# row['data_path'] = S.run_iv(bias_groups = [0,1,2,3,4,5,6,7,8,9,10,11], wait_time=0.001, bias_high=16, bias_low=0, bias_step = 0.025, overbias_voltage=18, cool_wait=300, high_current_mode=False, make_plot=False, save_plot=True, cool_voltage = 8)
 
# with open(out_fn, 'a', newline = '') as csvfile:
#     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#     writer.writerow(row)