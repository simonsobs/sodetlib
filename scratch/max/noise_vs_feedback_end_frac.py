import pysmurf
import numpy as np
import matplotlib.pyplot as plt

band = 2
frac_pp = 0.14139048298824738

feedback_end = 1.0 - np.asarray([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4])

for end in feedback_end:
    print('Feedback End:',end)
    S.tracking_setup(band,reset_rate_khz=4,lms_freq_hz=20000,fraction_full_scale=frac_pp,make_plot=True,show_plot=False,channel=S.which_on(band),nsamp=2**18,feedback_start_frac = 0.05,feedback_end_frac = end)
    S.take_noise_psd(band=band,meas_time=60,low_freq=[1],high_freq=[5],nperseg=2**16,save_data=True,show_plot=False,make_channel_plot=True)
