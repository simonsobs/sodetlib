'''
Code written in Oct 2021 by Yuhan Wang
to be used through OCS
UFM testing script in Pton
loop around different bias voltage and collect noise in transistion, special for cryocard c02-01 due to resistance loading issue, skipping BL10
'''

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
parser.add_argument('--biashigh',type=int)
parser.add_argument('--temp',type=str)
parser.add_argument('--output_file',type=str)

args = parser.parse_args()
slot_num = args.slot
bath_temp = args.temp
out_fn = args.output_file
bias_high = args.biashigh


cfg = DetConfig()
cfg.load_config_files(slot=slot_num)
S = cfg.get_smurf_control()


fieldnames = ['bath_temp', 'bias_v', 'band', 'data_path','step_size']
with open(out_fn, 'w', newline = '') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()



S.overbias_tes_all(bias_groups = [0,1,2,3,4,5,6,7,8,9,11], overbias_wait=1, tes_bias= 12, cool_wait= 3, high_current_mode=False, overbias_voltage= 8)
time.sleep(300)

step_array = np.arange(bias_high,0,-1)
step_size = 0.01 
for bias_voltage_step in step_array:

    bias_voltage = bias_voltage_step
    S.set_tes_bias_bipolar_array([bias_voltage,bias_voltage,bias_voltage,bias_voltage,bias_voltage,bias_voltage,bias_voltage,bias_voltage,bias_voltage,bias_voltage,0,bias_voltage,0.,0. ,0. ]) 
    time.sleep(120) 

    dat_path = S.stream_data_on()
    for k in [0,1,2,3,4,5,6,7,8,9,10,11]:
        time.sleep(10)
    S.stream_data_off()
    row = {}
    row['bath_temp'] = '100mK'
    row['bias_v'] = bias_voltage_step
    row['band'] = 'all'
    row['data_path'] = dat_path
    row['step_size'] = 0

    with open(out_fn, 'a', newline = '') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow(row)


