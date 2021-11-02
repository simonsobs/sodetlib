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
import scipy.signal as signal


start_time=S.get_timestamp()
fs = S.get_sample_frequency()
# hard coded (for now) variables
stream_time = 20

# non blocking statement to start time stream and return the dat filename
dat_path = S.stream_data_on()
# collect stream data
time.sleep(stream_time)
# end the time stream
S.stream_data_off()

timestamp, phase, mask, tes_bias = S.read_stream_data(dat_path, return_tes_bias=True)
print(f'loaded the .dat file at: {dat_path}')

# hard coded variables
bands, channels = np.where(mask != -1)
phase *= S.pA_per_phi0 / (2.0 * np.pi)  # uA
sample_nums = np.arange(len(phase[0]))
t_array = sample_nums / fs

# reorganize the data by band then channel
stream_by_band_by_channel = {}
for band, channel in zip(bands, channels):
    if band not in stream_by_band_by_channel.keys():
        stream_by_band_by_channel[band] = {}
    ch_idx = mask[band, channel]
    stream_by_band_by_channel[band][channel] = phase[ch_idx]

# plot the band channel data
fig, axs = plt.subplots(4, 2, figsize=(12, 24), gridspec_kw={'width_ratios': [2, 2]})
for band in sorted(stream_by_band_by_channel.keys()):
    stream_single_band = stream_by_band_by_channel[band]
    ax_this_band = axs[band // 2, band % 2]
    for channel in sorted(stream_single_band.keys()):
        stream_single_channel = stream_single_band[channel]
        stream_single_channel_norm = stream_single_channel - np.mean(stream_single_channel)
        ax_this_band.plot(t_array, stream_single_channel_norm, color='C0', alpha=0.002)
    band_yield = len(stream_single_band)
    ax_this_band.set_xlabel('time [s]')
    ax_this_band.set_ylabel('Phase [pA]')
    ax_this_band.grid()
    ax_this_band.set_title(f'band {band} yield {band_yield}')
    ax_this_band.set_ylim([-10000, 10000])

save_name = f'{start_time}_band_noise_stack.png'
print(f'Saving plot to {os.path.join(S.plot_dir, save_name)}')
plt.savefig(os.path.join(S.plot_dir, save_name))
# plt.show()

nperseg=2**12
detrend='constant'
# plot the PSD channel data
fig, axs = plt.subplots(4, 2, figsize=(12, 24), gridspec_kw={'width_ratios': [2, 2]})
for band in sorted(stream_by_band_by_channel.keys()):
    stream_single_band = stream_by_band_by_channel[band]
    ax_this_band = axs[band // 2, band % 2]
    for channel in sorted(stream_single_band.keys()):
        stream_single_channel = stream_single_band[channel]
        f, Pxx = signal.welch(stream_single_channel, nperseg=nperseg,
                fs=fs, detrend=detrend)
        Pxx = np.sqrt(Pxx)
        
        ax_this_band.loglog(f, Pxx, color='C0', alpha=0.002)
    band_yield = len(stream_single_band)
    ax_this_band.set_xlabel('Frequency [Hz]')
    ax_this_band.set_ylabel('Amp [pA/rtHz]')
    ax_this_band.grid()
    ax_this_band.set_title(f'band {band} yield {band_yield}')
    ax_this_band.set_ylim([1,1e4])

save_name = f'{start_time}_band_psd_stack.png'
print(f'Saving plot to {os.path.join(S.plot_dir, save_name)}')
plt.savefig(os.path.join(S.plot_dir, save_name))


# stupid way of doing per band now


for band in sorted(stream_by_band_by_channel.keys()):
    fig, ax = plt.subplots(figsize=(8, 6))
    stream_single_band = stream_by_band_by_channel[band]
    for channel in sorted(stream_single_band.keys()):
        stream_single_channel = stream_single_band[channel]
        f, Pxx = signal.welch(stream_single_channel, nperseg=nperseg,
                fs=fs, detrend=detrend)
        Pxx = np.sqrt(Pxx)
        
        ax .loglog(f, Pxx, color='C0', alpha=0.05)
    band_yield = len(stream_single_band)
    ax.axvline(1.4,linestyle='--', alpha=0.5,label = '1.4 Hz',color = 'C1')
    ax.axvline(60,linestyle='--', alpha=0.5,label = '60 Hz',color = 'C2')
    ax.axvline(3.56,linestyle='--', alpha=0.5,label = '3.56 Hz',color = 'C3')
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('Amp [pA/rtHz]')
    ax.grid()
    ax.set_title(f'band {band} yield {band_yield}')
    ax.set_ylim([1,1e4])
    ax.legend(loc=1)

    save_name = f'{start_time}_band_psd_stack_band{band}.png'
    print(f'Saving plot to {os.path.join(S.plot_dir, save_name)}')
    plt.savefig(os.path.join(S.plot_dir, save_name))



