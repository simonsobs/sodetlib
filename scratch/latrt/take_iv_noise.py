import numpy as np
import matplotlib
from pprint import pprint

matplotlib.use('agg')
from sodetlib.det_config import DetConfig
#from sodetlib.smurf_funcs.health_check import optimize_bias
#from sodetlib.smurf_funcs import health_check
#from sodetlib.smurf_funcs import optimize_uc_atten
#from sodetlib.smurf_funcs import get_median_noise

cfg = DetConfig()
cfg.load_config_files(slot=2)
S = cfg.get_smurf_control(dump_configs=True, make_logfile=True)

S.load_tune()
S.relock(1)
S.relock(3)

for _ in range(3):
    S.run_serial_gradient_descent(band=1)
    S.run_serial_eta_scan(band=1)
    S.run_serial_gradient_descent(band=3)
    S.run_serial_eta_scan(band=3)

# Band 1
# Tracking
band=1
band_cfg = cfg.dev.bands[band]
S.set_att_dc(band, 0)
S.set_att_uc(band, 26)
f, df, sync = S.tracking_setup(band, reset_rate_khz=4,lms_freq_hz=band_cfg['lms_freq_hz'],
                            meas_lms_freq=False, fraction_full_scale=0.2,
                            make_plot=True,show_plot=False,
                            channel=S.which_on(band),nsamp=2**18,
                            feedback_start_frac=0.02,feedback_end_frac=0.94)
# turn off bad channels
on_channels = S.which_on(band)
fspan = np.max(f,0)-np.min(f,0)
bad = np.any( [fspan>0.35, fspan<0.001],axis=0)
bad_chan = np.where(bad)[0]
for ch in bad_chan:
    S.channel_off(band, ch)

# Band 3 tracking
band=3
band_cfg = cfg.dev.bands[band]
S.set_att_dc(band, 0)
S.set_att_uc(band, 20)
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
for ch in bad_chan:
    S.channel_off(band, ch)

data, results = S.take_noise_psd(meas_time=10, low_freq=[1],
                                 high_freq=[10],nperseg=2**16, return_noise_params=True,
                                save_data=True,show_plot=True,make_channel_plot=False)


iv_file = S.run_iv(bias_groups=[1],bias_high=15.,bias_low=0.,bias_step=0.1,
                   make_plot=True, save_plot=True, channels=S.which_on(band=3),
                   high_current_mode=False,overbias_voltage=15.,cool_wait=30.)

bias_group_1_good_chan =  np.array([ 10, 18, 50, 58, 186, 282, 306, 354, 370, 418, 434 ])

S.noise_vs_bias(bias_group=1,
                band = 3*np.ones_like(bias_group_1_good_chan),
                channel=bias_group_1_good_chan,
                bias=[11,10,9,8,7,6,5,4,3,2,1,0.],high_current_mode=False,
                overbias_voltage=15.,meas_time=10.,analyze=True,cool_wait=10.,
                make_timestream_plot=True)

iv_file = S.run_iv(bias_groups=[4],bias_high=15.,bias_low=0.,bias_step=0.1,
                   make_plot=True, save_plot=True, channels=S.which_on(band=3),
                   high_current_mode=False,overbias_voltage=15.,cool_wait=30.)

bias_group_4_good_chan = np.array([2,12,20,28,56,60,76,116,124,140,268,276,
                          284,308,316,332,340,364,372,380,404,428,444])

S.noise_vs_bias(bias_group=4,
                band = 3*np.ones_like(bias_group_4_good_chan),
                channel=bias_group_4_good_chan,
                bias=[11,10,9,8,7,6,5,4,3,2,1,0.],high_current_mode=False,
                overbias_voltage=15.,meas_time=10.,analyze=True,cool_wait=10.,
                make_timestream_plot=True)
