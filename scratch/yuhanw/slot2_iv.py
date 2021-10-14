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

fieldnames = ['bath_temp', 'bias_line', 'band', 'data_path','datas']
with open(out_fn, 'w', newline = '') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
 
 
for bias_gp in [0,1,2,3,4,5,6,7,8,9,10,11]:
        row = {}
        row['bath_temp'] = bath_temp
        row['bias_line'] = bias_gp
        row['band'] = 'all'
             
        print(f'Taking IV on bias line {bias_gp}, all band')
          
 
        row['data_path'] = S.run_iv(bias_groups = [bias_gp], wait_time=0.001, bias_high=16, bias_low=0, bias_step = 0.025, overbias_voltage=18, cool_wait=10, high_current_mode=False, make_plot=False, save_plot=True, cool_voltage = 8)
     
          
        with open(out_fn, 'a', newline = '') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow(row)
 
 
# row = {}
# row['bath_temp'] = bath_temp
# row['bias_line'] = 'all'
# row['band'] = 'all'
 
# row['data_path'] = S.run_iv(bias_groups = [0,1,2,3,4,5,6,7,8,9,10,11], wait_time=0.001, bias_high=16, bias_low=0, bias_step = 0.025, overbias_voltage=18, cool_wait=300, high_current_mode=False, make_plot=False, save_plot=True, cool_voltage = 8)
 
# with open(out_fn, 'a', newline = '') as csvfile:
#     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#     writer.writerow(row)