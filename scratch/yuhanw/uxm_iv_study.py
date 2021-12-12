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

fieldnames = ['bath_temp', 'bias_line', 'band', 'data_path','datas','coolwait','cool_voltage','overbias','bias_high']
with open(out_fn, 'w', newline = '') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
 
for band in [0,1,2,3,4,5,6,7]:
    S.run_serial_gradient_descent(band);
    S.run_serial_eta_scan(band);
    S.set_feedback_enable(band,1) 
    S.tracking_setup(band,reset_rate_khz=cfg.dev.bands[band]['flux_ramp_rate_khz'],fraction_full_scale=cfg.dev.bands[band]['frac_pp'], make_plot=False, save_plot=False, show_plot=False, channel=S.which_on(band), nsamp=2**18, lms_freq_hz=None, meas_lms_freq=True,feedback_start_frac=cfg.dev.bands[band]['feedback_start_frac'],feedback_end_frac=cfg.dev.bands[band]['feedback_end_frac'],lms_gain=cfg.dev.bands[band]['lms_gain'])
    

bias_gp = 0
cool_wait_time = 30

for bias_high_v in [16,14,12,10,8]:
        row = {}
        row['bath_temp'] = bath_temp
        row['bias_line'] = bias_gp
        row['band'] = 'all'
        row['coolwait'] = cool_wait_time 
        row['cool_voltage'] = 8 
        row['overbias'] = 16
        row['bias_high'] = bias_high_v
        print(f'Taking IV on bias line {bias_gp}, all band')
          
        
        iv_data = S.run_iv(bias_groups = [bias_gp], wait_time=0.001, bias_high=bias_high_v, bias_low=0, bias_step = 0.025, overbias_voltage=16, cool_wait=cool_wait_time, high_current_mode=False, make_plot=False, save_plot=True, cool_voltage = 8)
        dat_file = iv_data[0:-13]+'.npy'
        row['data_path'] = dat_file
          
        with open(out_fn, 'a', newline = '') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow(row)
 
