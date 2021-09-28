import numpy as np
import matplotlib
from pprint import pprint

matplotlib.use('agg')
from sodetlib.det_config import DetConfig
import time
from sodetlib.smurf_funcs.smurf_ops import take_g3_data
import sodetlib.smurf_funcs.det_ops as do

cfg = DetConfig()
cfg.load_config_files(slot=2)
S = cfg.get_smurf_control(dump_configs=True, make_logfile=True)

dur = 21600
bands = [0,1,2,3]
bgs = [0,1,2,3,4,5]

for band in bands:
    band_cfg = cfg.dev.bands[band]
    S.set_att_uc( band, band_cfg['uc_att'] )
    S.set_att_dc( band, band_cfg['dc_att'] )

    S.setup_notches(band, tone_power=band_cfg['drive'],
                    new_master_assignment=False)
    for _ in range(3):
        S.run_serial_gradient_descent(band)
        S.run_serial_eta_scan(band)

    S.tracking_setup(band, reset_rate_khz=cfg.dev.bands[band]['flux_ramp_rate_khz'],
                     fraction_full_scale=cfg.dev.bands[band]['frac_pp'], 
                     make_plot=False, save_plot=False, show_plot=False,
                     nsamp=2**18, lms_freq_hz=band_cfg['lms_freq_hz'], 
                     meas_lms_freq=False,
                     feedback_start_frac=cfg.dev.bands[band]['feedback_start_frac'],
                     feedback_end_frac=cfg.dev.bands[band]['feedback_end_frac'],
                     lms_gain=cfg.dev.bands[band]['lms_gain'])

chosen_biases_fp = '/data/smurf_data/20210614/1623690456/outputs/1623704051_bias_points.npy'
do.bias_detectors_from_sc(S, chosen_biases_fp)
take_g3_data(S, dur, tag='ufm_cv4,optical,bias_noise')

for bg in bgs:
    S.set_tes_bias_bipolar(bg,0)

