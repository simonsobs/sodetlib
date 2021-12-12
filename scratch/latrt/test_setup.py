"""run the specific setup we want for a long term stability test over
thanksgiving break
"""

import matplotlib
matplotlib.use('Agg')

import time
import numpy as np
from sodetlib.det_config import DetConfig

import sodetlib.smurf_funcs.smurf_ops as so
import sodetlib.smurf_funcs.det_ops as do
import sodetlib.analysis.det_analysis as da

from sodetlib.smurf_funcs.det_ops import take_iv
from sodetlib.smurf_funcs.smurf_ops import take_g3_data, stream_g3_off, stream_g3_on

bands = range(4)
bgs_all = range(6)
bgs_to_bias = [0,1,2,3,4,5]

cfg = DetConfig()
cfg.load_config_files(slot=2)
S = cfg.get_smurf_control(dump_configs=True, make_logfile=False, 
			apply_dev_configs=True, load_device_tune=True)
"""" Comment out to where things broke
## Turn off detector biases in case they are on
for b in bgs_all:
    S.set_tes_bias_bipolar(b,0)
print("Biases Off, wait 3 min for cool")
time.sleep(180)

## Reset Tuning on the same channel assignments.
S.load_tune(cfg.dev.exp['tunefile'])
for band in bands:
    band_cfg = cfg.dev.bands[band]
    S.setup_notches(band, tone_power=band_cfg['drive'],
                    new_master_assignment=False)
    for _ in range(3):
        S.run_serial_gradient_descent(band)
        S.run_serial_eta_scan(band)

## WARNING FILE CHANGING!!!!
cfg.dev.update_experiment({'tunefile': S.tune_file})
cfg.dev.dump('/config/device_configs/dev_cfg_s2_cv4.yaml', clobber=True)

## Set up tracking
for band in bands:
    band_cfg = cfg.dev.bands[band]
    S.tracking_setup(band, reset_rate_khz=cfg.dev.bands[band]['flux_ramp_rate_khz'],
                 fraction_full_scale=band_cfg['frac_pp'] , 
                 make_plot=True, save_plot=True, show_plot=False,
                 nsamp=2**18, lms_freq_hz=band_cfg['lms_freq_hz'] , 
                 meas_lms_freq=False, channel = S.which_on(band)[::20],
                 feedback_start_frac=cfg.dev.bands[band]['feedback_start_frac'],
                 feedback_end_frac=cfg.dev.bands[band]['feedback_end_frac'],
                 lms_gain=cfg.dev.bands[band]['lms_gain'])

## measure readout noise after tracking
S.take_noise_psd(20, nperseg=2**16, save_data=True, 
                 make_channel_plot=False, return_noise_params=True, )

## Tickles, because we like these
sfile = do.take_tickle(S, cfg, bgs_all, tickle_freq=5, tickle_voltage=0.005, 
		   high_current=True)
summary, segs, summary_fp = da.analyze_tickle_data(S, sfile, return_full=True, 
				return_segs=True, sc_thresh = 0.01)
bg_map_fp = da.make_bias_group_map(S, tsum_fp = summary_fp)


## IV over all at once
iv_info_fp = do.take_iv(S=S,cfg=cfg,bias_groups=bgs_all,bias_high=19.9,bias_low=0.0,
                        bias_step=0.1,cool_wait=20., high_current_mode=False,
                        do_analysis=True,cool_voltage=10.0,overbias_voltage=19.9, 
                        make_summary_plots=True)
"""
bg_map_fp = '/data/smurf_data/bias_group_maps/1638204825_bg_map.npy'
## Find points we'd like to bias at
chosen_biases_fp = da.bias_points_from_rfrac(S, cfg, cfg.dev.exp['iv_analyze'], 
			bg_map_fp, rfrac=0.5, bias_groups=bgs_to_bias)
do.bias_detectors_from_sc(S, chosen_biases_fp)
print(S.get_tes_bias_bipolar_array())

