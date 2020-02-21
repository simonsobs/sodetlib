import pysmurf
import numpy as np
import matplotlib.pyplot as plt

band = 2
frac_pp = 0.14139048298824738
for atten in np.arange(30,-2,-2):
    print('UC Atten:',atten)
    S.set_att_uc(band,atten)
    S.run_serial_gradient_descent(band)
    S.run_serial_eta_scan(band)
    S.tracking_setup(band,reset_rate_khz=4,lms_freq_hz=20000,fraction_full_scale=frac_pp,make_plot=True,show_plot=False,channel=S.which_on(band),nsamp=2**18)
    S.take_noise_psd(band=band,meas_time=60,low_freq=[1],high_freq=[5],nperseg=2**16,save_data=True,show_plot=False,make_channel_plot=True)
