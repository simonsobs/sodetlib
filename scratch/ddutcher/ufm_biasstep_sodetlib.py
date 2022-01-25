'''
Code written in Oct 2021 by Yuhan Wang
to be used through OCS
UFM testing script in Pton
loop around different bias voltage and collect biasstep measurement 
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
from sodetlib.smurf_funcs import det_ops
import numpy as np
from scipy.interpolate import interp1d
import argparse
import time
import csv

parser = argparse.ArgumentParser()

parser.add_argument('--slot', type=int)
parser.add_argument('--bg', type=int, nargs='+', default=None)
parser.add_argument('--biashigh', type=float)
parser.add_argument('--biaslow',type=float, default=0)
parser.add_argument('--temp', type=str)
parser.add_argument('--output_file', type=str)

args = parser.parse_args()
if args.bg is None:
    bias_groups = [0,1,2,3,4,5,6,7,8,9,10,11]
else:
    bias_groups = args.bg
slot_num = args.slot
bath_temp = args.temp
out_fn = args.output_file
bias_high = args.biashigh
bias_low = args.biaslow

cfg = DetConfig()
cfg.load_config_files(slot=slot_num)
S = cfg.get_smurf_control()
S.load_tune(cfg.dev.exp['tunefile'])

fieldnames = ['bath_temp', 'bias_v', 'band', 'data_path','step_size']
if not os.path.exists(out_fn):
    with open(out_fn, 'w', newline = '') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

S.overbias_tes_all(
    bias_groups=bias_groups,
    overbias_wait=1,
    tes_bias=20,
    cool_wait=3,
    high_current_mode=False,
    overbias_voltage=12,
)
time.sleep(300)

step_array = np.arange(bias_high, bias_low, -0.5)
#step_size = 0.01 
for bias_voltage_step in step_array:
    bias_array = np.zeros(S._n_bias_groups)
    bias_voltage = bias_voltage_step
    for bg in bias_groups:
        bias_array[bg] = bias_voltage
    S.set_tes_bias_bipolar_array(bias_array) 
    time.sleep(120)

    bsa = det_ops.take_bias_steps(S, cfg)

    row = {}
    row['bath_temp'] = f'{bath_temp}mK'
    row['bias_v'] = bias_voltage_step
    row['band'] = 'all'
    row['data_path'] = bsa.filepath
    row['step_size'] = .05

    with open(out_fn, 'a', newline = '') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow(row)

    #take 20s timestream for noise
    for bg in bias_groups:
        bias_array[bg] = bias_voltage_step
    S.set_tes_bias_bipolar_array(bias_array)
    time.sleep(10)
    dat_path = S.stream_data_on()
    time.sleep(20)
    S.stream_data_off()
    row['data_path'] = dat_path
    row['step_size'] = 0

    with open(out_fn, 'a', newline = '') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow(row)
