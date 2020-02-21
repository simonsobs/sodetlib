import pysmurf
import numpy as np
import matplotlib.pyplot as plt

band = 2
frac_pp = 0.14139048298824738

for dr in np.arange(11,4,-1):
    dr_arr = S.get_amplitude_scale_array(band)
    cur_dr = dr_arr[np.where(dr_arr != 0)][0]
    new_dr = dr_arr*(dr/cur_dr)
    new_dr = new_dr.astype(int)
    S.set_amplitude_scale_array(band,new_dr)
    S.run_serial_gradient_descent(band)
    S.run_serial_eta_scan(band)
    S.tracking_setup(band,reset_rate_khz=4,lms_freq_hz=20000,fraction_full_scale=frac_pp,make_plot=True,show_plot=False,channel=S.which_on(band),nsamp=2**18)
    S.take_noise_psd(band=band,meas_time=60,low_freq=[1],high_freq=[5],nperseg=2**16,save_data=True,show_plot=False,make_channel_plot=True)
