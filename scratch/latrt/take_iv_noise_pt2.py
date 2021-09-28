import numpy as np
import matplotlib
from pprint import pprint

matplotlib.use('agg')
from sodetlib.det_config import DetConfig
import time
from sodetlib.smurf_funcs.det_ops import take_iv


cfg = DetConfig()
cfg.load_config_files(slot=2)
S = cfg.get_smurf_control(dump_configs=True, make_logfile=True)

bands = [0,1,2,3]
bgs = [1,2,3,4,5]

for band in bands:
    band_cfg = cfg.dev.bands[band]
    S.set_att_uc( band, band_cfg['uc_att'] )
    S.set_att_dc( band, band_cfg['dc_att'] )

    S.find_freq(band, tone_power=band_cfg['drive'], make_plot=True,
                save_plot=True)
    S.setup_notches(band, tone_power=band_cfg['drive'],
                    new_master_assignment=True)
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

S.take_stream_data(20, return_data=True, register_file=True)

for bg in bgs:
    iv_info_fp = take_iv(S=S,bias_groups=[bg],bias_high=19.9,bias_low=0.0,
            bias_step=0.1,cool_wait=60., high_current_mode=False,do_analysis=False,
            cool_voltage=10.0,overbias_voltage=19.9)
    time.sleep(30)

bias_range = np.linspace(0,10,11)

for bias in bias_range:
    S.overbias_tes_all(bias_groups=bgs, overbias_voltage=19.9, tes_bias = bias)
    S.take_stream_data(20, return_data=True, register_file=True)

for bg in bgs:
    S.set_tes_bias_bipolar(bg,0)

