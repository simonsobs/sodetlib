'''
takes SC noise and takes IV
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
import sodetlib as sdl
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

S.load_tune(cfg.dev.exp['tunefile'])

for band in [0,1,2,3,4,5,6,7]:
    S.run_serial_gradient_descent(band);
    S.run_serial_eta_scan(band);
    S.set_feedback_enable(band,1) 
    S.tracking_setup(
        band, reset_rate_khz=cfg.dev.bands[band]['flux_ramp_rate_khz'],
        fraction_full_scale=cfg.dev.bands[band]['frac_pp'], make_plot=False,
        save_plot=False, show_plot=False, channel=S.which_on(band),
        nsamp=2**18, lms_freq_hz=None, meas_lms_freq=True,
        feedback_start_frac=cfg.dev.bands[band]['feedback_start_frac'],
        feedback_end_frac=cfg.dev.bands[band]['feedback_end_frac'],
        lms_gain=cfg.dev.bands[band]['lms_gain'],
    )
  
bias_groups = [0,1,2,3,4,5,6,7,8,9,10,11] 
S.set_filter_disable(0)
S.set_rtm_arb_waveform_enable(0)
S.set_downsample_factor(20)
for bias_index, bias_g in enumerate(bias_groups):
    S.set_tes_bias_low_current(bias_g)

bias_v = 0
bias_array = np.zeros(S._n_bias_groups)
for bg in bias_groups:
    bias_array[bg] = bias_v
S.set_tes_bias_bipolar_array(bias_array)
time.sleep(10)
# datafile_self = S.stream_data_on()
# time.sleep(20)
# S.stream_data_off()

#take 30s timestream for noise
sid = sdl.take_g3_data(S, 30)
am = sdl.load_session(cfg.stream_id, sid, base_dir=cfg.sys['g3_dir'])
ctime = int(am.timestamps[0])
noisedict = sdl.noise.get_noise_params(
    am, wl_f_range=(10,30), fit=False, nperseg=1024)
noisedict['sid'] = sid
savename = os.path.join(S.output_dir, f'{ctime}_take_noise.npy')
sdl.validate_and_save(
    savename, noisedict, S=S, cfg=cfg, make_path=False
)

fieldnames = ['bath_temp','bias_voltage', 'bias_line', 'band', 'data_path','type'] 
row = {}
row['data_path'] = savename
row['bias_voltage'] = bias_v
row['type'] = 'noise'
row['bias_line'] = 'all'
row['band'] = 'all'
row['bath_temp'] = bath_temp
with open(out_fn, 'a', newline = '') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writerow(row)

for bias_gp in bias_groups:
    row = {}
    row['bath_temp'] = bath_temp
    row['bias_line'] = bias_gp
    row['band'] = 'all'
    row['bias_voltage'] = 'IV 18 to 0'
    row['type'] = 'IV'
    print(f'Taking IV on bias line {bias_gp}, all band')
      
    row['data_path'] = det_ops.take_iv(
        S,
        cfg,
        bias_groups = [bias_gp],
        wait_time=0.001,
        bias_high=18,
        bias_low=0,
        bias_step = 0.025,
        overbias_voltage=18,
        cool_wait=30,
        high_current_mode=False,
        make_channel_plots=False,
        save_plots=True,
        do_analysis=True,
    )

    with open(out_fn, 'a', newline = '') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow(row)

    time.sleep(30)
