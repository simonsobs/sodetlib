import pysmurf
import numpy as np
import matplotlib.pyplot as plt

band = 2
frac_pp_5 = 0.14139048298824738

frac_pp = frac_pp_5*np.asarray([2,3,4,5,6,7,8,9,10])/5

lms_freq_adj = 20000*np.asarray([2,3,4,5,6,7,8,9,10])/5

for i, fpp in enumerate(frac_pp):
    print('Feedback Start:',start)
    S.tracking_setup(band,reset_rate_khz=4,lms_freq_hz=lms_freq_adj[i],fraction_full_scale=fpp,make_plot=True,show_plot=False,channel=S.which_on(band),nsamp=2**18,feedback_start_frac = 0.02,feedback_end_frac = 0.98)
    S.take_noise_psd(band=band,meas_time=60,low_freq=[1],high_freq=[5],nperseg=2**16,save_data=True,show_plot=False,make_channel_plot=True)
