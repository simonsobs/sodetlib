import os
import sys
from time import sleep
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
sys.path.append(f'/home/yuhanw/repos/pysmurf/python')
import pysmurf.client
S = pysmurf.client.SmurfControl(offline=True)
print(f'import complete at {datetime.now()}')


# hard coded (for now) variables
nfs_dir = os.path.join('/', 'data', 'legacy', 'smurfsrv')


# convert to the path used on the analysis machine function
def path_convert(path_smurf_server):
    print(path_smurf_server)
    _smurf_data_dir, local_path = path_smurf_server.split(f'/smurf_data/', 1)
    return os.path.join(nfs_dir, local_path)


def load_tune_file(tune_path_smurf_server, on_smurf_server=True):
    # get the tune file
    if on_smurf_server:
        tune_path = tune_path_smurf_server
    else:
        # assumed to be on the analysis machine
        tune_path = path_convert(path_smurf_server=tune_path_smurf_server)
    tune_data = np.load(tune_path, allow_pickle=True).item()
    print(f'loaded the tune file at: {tune_path}  on {datetime.now()}')
    return tune_data


def load_dat_file(dat_path_smurf_server, on_smurf_server=True):
    # get the noise PSD from the .dat file
    if on_smurf_server:
        dat_path = dat_path_smurf_server
    else:
        # assumed to be on the analysis machine
        dat_path = path_convert(path_smurf_server=dat_path_smurf_server)

    timestamp, phase, mask, tes_bias = S.read_stream_data(dat_path, return_tes_bias=True)
    print(f'loaded the .dat file at: {dat_path}')
    return timestamp, phase, mask, tes_bias


def plt_stream_by_band(dat_path_smurf_server, on_smurf_server=True):
    # get the noise PSD from the .dat file
    timestamp, phase, mask, tes_bias = load_dat_file(dat_path_smurf_server=dat_path_smurf_server,
                                                     on_smurf_server=on_smurf_server)

    # hard coded variables
    bands, channels = np.where(mask != -1)
    S._bias_line_resistance = 15600
    ch_idx_self = 0
    S.pA_per_phi0 = 9e6
    phase *= S.pA_per_phi0/(2.0 * np.pi)  # uA
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
    plt.savefig("plt_stream_by_band.png")
    plt.show()


def take_noise_by_band_make_stack_plots(stream_time=120, on_smurf_server=False):
    # non blocking statement to start time stream and return the dat filename
    datafile_self = S.stream_data_on()
    # collect stream data
    sleep(stream_time)
    # end the time stream
    S.stream_data_off()
    # plot the .dat time stream data file
    plt_stream_by_band(dat_path_smurf_server=datafile_self, on_smurf_server=on_smurf_server)


if __name__ == "__main__":
    # options and file targets
    # tune_path_on_smurf_server = "/data/smurf_data/tune/1631847058_tune.npy"
    # dat_path_on_smurf = '/data/smurf_data/20210917/crate1slot3/1631844343/outputs/1631846413.dat'
    take_noise_by_band_make_stack_plots(stream_time=20)


