import numpy as np
import matplotlib
from pprint import pprint
import time

matplotlib.use('agg')
from sodetlib.det_config import DetConfig

cfg = DetConfig()
cfg.load_config_files(slot=2)
S = cfg.get_smurf_control(dump_configs=True, make_logfile=True)

#Band 3 tracking
band = 3
S.pA_per_phi0 = 9e6
S.R_sh=400e-6
S.bias_line_resistance=16400.0
cfg.dev.update_band(3, {'lms_freq_hz':15611.557801177205})
band_cfg = cfg.dev.bands[band]
S.set_att_dc(band, 0)
S.set_att_uc(band, 28)

S.load_tune()
S.setup_notches(band, new_master_assignment=False)
#S.relock(band)  #S.setup_notches runs reload at the end

for _ in range(3):
    S.run_serial_gradient_descent(band)
    S.run_serial_eta_scan(band)

f, df, sync = S.tracking_setup(band, reset_rate_khz=4,lms_freq_hz=band_cfg['lms_freq_hz'],
                            meas_lms_freq=False, fraction_full_scale=0.2,
                            make_plot=True,show_plot=True,
                            channel=S.which_on(band),nsamp=2**18,
                        feedback_start_frac=0.02,feedback_end_frac=0.94)

on_channels = S.which_on(band)
fspan = np.max(f,0)-np.min(f,0)
bad = np.any( [fspan>0.35, fspan<0.001],axis=0)
bad_chan = np.where(bad)[0]
#for ch in bad_chan:
  #  if ch == 98 or ch == 458:
  #      continue
   # S.channel_off(band, ch)

data, results = S.take_noise_psd(meas_time=10, low_freq=[1],
                                 high_freq=[10],nperseg=2**16, return_noise_params=True,
                                save_data=True,show_plot=False,make_channel_plot=True)

iv_file = S.run_iv(bias_groups=[1,4],bias_high=15.,bias_low=0.,bias_step=0.1,
                   make_plot=True, save_plot=True, channels=S.which_on(band=3),
                   high_current_mode=False,overbias_voltage=15.,cool_wait=30.)

S.overbias_tes(1,overbias_voltage=15.,tes_bias=6,high_current_mode=False)
S.overbias_tes(4,overbias_voltage=15.,tes_bias=6,high_current_mode=False)

time.sleep(60)

data, results = S.take_noise_psd(meas_time=10, low_freq=[1],
                                 high_freq=[10],nperseg=2**16, return_noise_params=True,
                                save_data=True,show_plot=False,make_channel_plot=True)

