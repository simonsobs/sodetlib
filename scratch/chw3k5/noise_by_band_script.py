import os

from time import sleep
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt


# hard coded (for now) variables
stream_time = 20

# non blocking statement to start time stream and return the dat filename
dat_path = S.stream_data_on()
# collect stream data
sleep(stream_time)
# end the time stream
S.stream_data_off()

timestamp, phase, mask, tes_bias = S.read_stream_data(dat_path, return_tes_bias=True)
print(f'loaded the .dat file at: {dat_path}')

# hard coded variables
bands, channels = np.where(mask != -1)
S._bias_line_resistance = 15600
ch_idx_self = 0
S.pA_per_phi0 = 9e6
phase *= S.pA_per_phi0 / (2.0 * np.pi)  # uA
S.high_low_current_ratio = 6.08
fs = 200
S._R_sh = 0.0004
sample_nums = np.arange(len(phase[ch_idx_self]))
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
    ax_this_band.set_xlabel('time [s]')
    ax_this_band.set_ylabel('Phase [pA]')
    ax_this_band.grid()
    ax_this_band.set_title(f'band {band}')
    ax_this_band.set_ylim([-10000, 10000])

current_dir = os.path.basename(os.path.abspath(__file__))
plt.savefig(os.path.join(current_dir, "plt_stream_by_band.png"))
plt.show()
