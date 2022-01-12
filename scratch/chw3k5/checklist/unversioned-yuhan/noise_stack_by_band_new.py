'''
Code written in Oct 2021 by Yuhan Wang
taking short time stream and report noise property by band
'''

import matplotlib
matplotlib.use('Agg')

import pysmurf.client
import argparse
import numpy as np
import os
import time
import glob
from sodetlib.det_config  import DetConfig
import numpy as np
from scipy.interpolate import interp1d
import argparse
import time
import csv
from scipy import signal
import os
import time

import warnings
warnings.filterwarnings("ignore")


fs = S.get_sample_frequency()
# hard coded (for now) variables
stream_time = 60

# non blocking statement to start time stream and return the dat filename
dat_path = S.stream_data_on()
# collect stream data
time.sleep(stream_time)
# end the time stream
S.stream_data_off()

start_time = dat_path[-14:-4]  

timestamp, phase, mask, tes_bias = S.read_stream_data(dat_path, return_tes_bias=True)
print(f'loaded the .dat file at: {dat_path}')

# hard coded variables
bands, channels = np.where(mask != -1)
phase *= S.pA_per_phi0 / (2.0 * np.pi)  # pA
sample_nums = np.arange(len(phase[0]))
t_array = sample_nums / fs

# reorganize the data by band then channel
stream_by_band_by_channel = {}
for band, channel in zip(bands, channels):
    if band not in stream_by_band_by_channel.keys():
        stream_by_band_by_channel[band] = {}
    ch_idx = mask[band, channel]
    stream_by_band_by_channel[band][channel] = phase[ch_idx]



fmin=5
fmax=50
detrend='constant'
# plot the band channel data
fig, axs = plt.subplots(4, 2, figsize=(12, 24), gridspec_kw={'width_ratios': [2, 2]},dpi=50)
for band in sorted(stream_by_band_by_channel.keys()):
    wl_list_temp = []
    stream_single_band = stream_by_band_by_channel[band]
    ax_this_band = axs[band // 2, band % 2]
    for channel in sorted(stream_single_band.keys()):
        stream_single_channel = stream_single_band[channel]


        f, Pxx = signal.welch(stream_single_channel, fs=fs, detrend=detrend,nperseg=2**16)
        Pxx = np.sqrt(Pxx)
        fmask = (fmin < f) & (f < fmax)
        wl = np.median(Pxx[fmask])
        wl_list_temp.append(wl)
        stream_single_channel_norm = stream_single_channel - np.mean(stream_single_channel)
        ax_this_band.plot(t_array, stream_single_channel_norm, color='C0', alpha=0.002)
    wl_median = np.median(wl_list_temp)
    band_yield = len(stream_single_band)
    ax_this_band.set_xlabel('time [s]')
    ax_this_band.set_ylabel('Phase [pA]')
    ax_this_band.grid()
    ax_this_band.set_title(f'band {band} yield {band_yield} median noise {wl_median:.2f}')
    ax_this_band.set_ylim([-10000, 10000])

save_name = f'{start_time}_band_noise_stack.png'
print(f'Saving plot to {os.path.join(S.plot_dir, save_name)}')
plt.savefig(os.path.join(S.plot_dir, save_name))

fig, axs = plt.subplots(4, 4, figsize=(24, 24), gridspec_kw={'width_ratios': [2, 2,2,2]},dpi=50)
for band in sorted(stream_by_band_by_channel.keys()):
    wl_list_temp = []
    stream_single_band = stream_by_band_by_channel[band]
    ax_this_band = axs[band // 2 , band % 2 * 2]
    for channel in sorted(stream_single_band.keys()):
        stream_single_channel = stream_single_band[channel]
        f, Pxx = signal.welch(stream_single_channel,
                fs=fs, detrend=detrend,nperseg=2**16)
        Pxx = np.sqrt(Pxx)
        fmask = (fmin < f) & (f < fmax)
        wl = np.median(Pxx[fmask])
        wl_list_temp.append(wl)
        ax_this_band.loglog(f, Pxx, color='C0', alpha=0.2)

    wl_median = np.median(wl_list_temp)


    band_yield = len(stream_single_band)
    ax_this_band.set_xlabel('Frequency [Hz]')
    ax_this_band.set_ylabel('Amp [pA/rtHz]')
    ax_this_band.grid()
    ax_this_band.axvline(1.4,linestyle='--', alpha=0.6,label = '1.4 Hz',color = 'C1')
    ax_this_band.axvline(60,linestyle='--', alpha=0.6,label = '60 Hz',color = 'C2')
    # ax_this_band.axvline(3.,linestyle='--', alpha=0.6,label = '60 Hz',color = 'C3')
    # ax_this_band.axvline(4.,linestyle='--', alpha=0.6,label = '60 Hz',color = 'C3')
    # ax_this_band.axvline(5.,linestyle='--', alpha=0.6,label = '60 Hz',color = 'C3')
    # ax_this_band.axvline(6.,linestyle='--', alpha=0.6,label = '60 Hz',color = 'C3')
    # ax_this_band.axvline(7.,linestyle='--', alpha=0.6,label = '60 Hz',color = 'C3')
    ax_this_band.set_title(f'band {band} yield {band_yield}')
    ax_this_band.set_ylim([1,1e4])



    ax_this_band_2 = axs[band // 2  , band % 2 * 2+ 1]
    ax_this_band_2.set_xlabel('Amp [pA/rtHz]')
    ax_this_band_2.set_ylabel('count')
    ax_this_band_2.hist(wl_list_temp, range=(0,300),bins=60)
    ax_this_band_2.axvline(wl_median, linestyle='--', color='gray')
    ax_this_band_2.grid()
    ax_this_band_2.set_title(f'band {band} yield {band_yield} median noise {wl_median:.2f}')
    ax_this_band_2.set_xlim([0,300])

save_name = f'{start_time}_band_psd_stack.png'
print(f'Saving plot to {os.path.join(S.plot_dir, save_name)}')
plt.savefig(os.path.join(S.plot_dir, save_name))






S.save_tune()    


print('plotting directory is:')
print(S.plot_dir)
