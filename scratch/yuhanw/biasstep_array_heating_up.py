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


#detectors should already be at around 50 percent Rn before doing this.
out_fn = '/data/smurf_data/UFM_testing/Mv6/array_heating/trial.csv'
bath_temp = 100



start_time=S.get_timestamp()
fs = S.get_sample_frequency()
# hard coded (for now) variables
fieldnames = ['bath_temp','bias_voltage', 'bias_line', 'band', 'data_path','note']

perp_bl = 11
victim_bl = 8


bias_voltage = 4
step_size = 0.01
stream_time = 4

signal = np.ones(2048)
S.get_tes_bias_bipolar(victim_bl)
signal = np.ones(2048)
signal *= bias_voltage / (2*S._rtm_slow_dac_bit_to_volt) #defining the bias step level (lower step)
signal[1024:] += step_size / (2*S._rtm_slow_dac_bit_to_volt) #defining the bias step level (upper step)
freq = 1
period = 1/freq
ts = int(period*4/(6.4e-9 * 2048))
S.set_rtm_arb_waveform_timer_size(ts, wait_done = True)
S.set_rtm_arb_waveform_enable(1)
S.play_tes_bipolar_waveform(victim_bl,signal)

# non blocking statement to start time stream and return the dat filename
dat_path = S.stream_data_on()
# collect stream data
time.sleep(stream_time)
# end the time stream
S.stream_data_off()

row = {}
row['data_path'] = dat_path
row['bias_voltage'] = bias_voltage
row['note'] = 'bl8 step around 50 percent Rn, bl11 goes from 0V to 12V'
row['bias_line'] = 'all'
row['band'] = 'all'
row['bath_temp'] = bath_temp
with open(out_fn, 'a', newline = '') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writerow(row)

S.set_rtm_arb_waveform_enable(0)

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

