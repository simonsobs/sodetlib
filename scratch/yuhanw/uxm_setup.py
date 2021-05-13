import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pysmurf.client
import argparse
import numpy as np
import pickle as pkl
from scipy import signal
import os

from sodetlib.det_config import DetConfig



bands = [0,1,2,3]
slot_num = 2

cfg = DetConfig()
cfg.load_config_files(slot=slot_num)
S = cfg.get_smurf_control()

print('plotting directory is:')
print(S.plot_dir)

S.all_off()
S.set_rtm_arb_waveform_enable(0)
S.set_filter_disable(0)
S.set_downsample_factor(20)
S.set_mode_dc()

for band in bands:
	print('setting up band {}'.format(band))

	S.set_att_dc(band,cfg.dev.bands[band]['dc_att'])
	print('band {} dc_att {}'.format(band,S.get_att_dc(band)))

	S.set_att_uc(band,cfg.dev.bands[band]['uc_att'])
	print('band {} uc_att {}'.format(band,S.get_att_uc(band)))

	S.amplitude_scale[band] = cfg.dev.bands[band]['drive']
	print('band {} tone power {}'.format(band,S.amplitude_scale[band] ))

	print('estimating phase delay')
	S.estimate_phase_delay(band)
	print('setting synthesis scale')
	# hard coding it for the current fw
	S.set_synthesis_scale(band,1)
	print('running find freq')
	S.find_freq(band,tone_power=cfg.dev.bands[band]['drive'],make_plot=True)
	print('running setup notches')
	S.setup_notches(band,tone_power=cfg.dev.bands[band]['drive'],new_master_assignment=True)
	print('running serial gradient descent and eta scan')
	S.run_serial_gradient_descent(band);
	S.run_serial_eta_scan(band);
	print('running tracking setup')
	S.set_feedback_enable(band,1) 
	S.tracking_setup(band,reset_rate_khz=cfg.dev.bands[band]['flux_ramp_rate_khz'],fraction_full_scale=cfg.dev.bands[band]['frac_pp'], make_plot=False, save_plot=False, show_plot=False, channel=S.which_on(band), nsamp=2**18, lms_freq_hz=None, meas_lms_freq=True,feedback_start_frac=cfg.dev.bands[band]['feedback_start_frac'],feedback_end_frac=cfg.dev.bands[band]['feedback_end_frac'],lms_gain=cfg.dev.bands[band]['lms_gain'])
	print('checking tracking')
	S.check_lock(band,reset_rate_khz=cfg.dev.bands[band]['flux_ramp_rate_khz'],fraction_full_scale=cfg.dev.bands[band]['frac_pp'], lms_freq_hz=None, feedback_start_frac=cfg.dev.bands[band]['feedback_start_frac'],feedback_end_frac=cfg.dev.bands[band]['feedback_end_frac'],lms_gain=cfg.dev.bands[band]['lms_gain'])

print('taking 20s timestream')
S.take_noise_psd(20, nperseg=2**16, save_data=True, make_channel_plot=False, return_noise_params=True)

S.save_tune()    
print('plotting directory is:')
print(S.plot_dir)