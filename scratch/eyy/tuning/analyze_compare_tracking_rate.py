import pysmurf.client
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.signal as signal

####
# Assumes you've already setup the system.
####

### System Configuration ###
datafile_file = os.path.join('/data/smurf_data/20200603/1591207496/outputs/',
    '1591207689_compare_tracking_noise.npy')
reset_rate_khzs = np.load(datafile_file.replace('_compare_tracking_noise.npy',
    '_reset_rate_khz.npy'))
n_phi0s = np.load(datafile_file.replace('_compare_tracking_noise.npy',
    '_n_phi0.npy'))
plot_dir = os.path.split(datafile_file)[0].replace('outputs', 'plots')

timestamp = os.path.split(datafile_file)[1].split('_')[0]

nperseg = 2**12
bins_min = np.array([.2, .5, 1, 3, 10])
bins_max = np.array([.5, 1, 3, 10, 30])
n_bins = len(bins_min)

epics_prefix = 'smurf_server_s5'
config_file = os.path.join('/data/pysmurf_cfg/experiment_fp30_cc02-03_lbOnlyBay0.cfg')


# Instatiate pysmurf object
S = pysmurf.client.SmurfControl(epics_root=epics_prefix, cfg_file=config_file,
    setup=False, make_logfile=False, shelf_manager='shm-smrf-sp01')

datafiles = np.load(datafile_file, allow_pickle=True).item()

pxx_all = {}
bin_vals_all = {}
bands_all = {}
channels_all = {}

for kk in datafiles:
    datafile = datafiles[kk]
    fs = reset_rate_khzs[kk] * 1.0E3  # downsample fitler set to 1

    # Load data
    print(datafile)
    t, d, m = S.read_stream_data(datafile)
    d *= S.pA_per_phi0/2/np.pi

    # Extract channels
    bands, channels = np.where(m != -1)
    n_chan = len(bands)

    # Calculate psds
    pxx = np.zeros((n_chan, nperseg//2+1))
    bin_vals = np.zeros((n_chan, n_bins))
    for i, (b, ch) in enumerate(zip(bands, channels)):
        key = m[b, ch]
        f, pxx[i] = signal.welch(d[key], fs=fs, nperseg=nperseg)

        for j, (bmin, bmax) in enumerate(zip(bins_min, bins_max)):
            idx = np.where(np.logical_and(f > bmin, f < bmax))
            bin_vals[i, j] = np.median(pxx[key][idx])

    bands_all[kk] = bands
    channels_all[kk] = channels

    bin_vals_all[kk] = bin_vals
    pxx_all[kk] = np.sqrt(pxx)

# Find unique channels
all_channels = np.array([])
for k in bands_all.keys():
    bands = bands_all[k]
    channels = channels_all[k]
    all_channels = np.append(all_channels, bands*512+channels)
all_channels = np.unique(all_channels)

cm = plt.get_cmap('gist_rainbow')
for cchh in all_channels:
    b = cchh // 512
    ch = cchh % 512

    plt.figure()
    nm = np.max(list(datafiles.keys()))
    for kk in datafiles:
        color = cm(kk/nm)
        if b in bands_all[kk] and ch in channels_all[kk]:
            idx = np.ravel(np.where(np.logical_and(bands_all[kk] == b,
                channels_all[kk] == ch)))[0]
            label = f'FR {reset_rate_khzs[kk]} kHz ' + r'n$\phi_0$ ' + \
                f'{n_phi0s[kk]}'
            plt.semilogx(f, pxx_all[kk][idx], label=label, color=color)

    plt.legend(loc='upper right')
    plt.xlabel('Freq [Hz]')
    plt.ylabel('Amp [pA/rtHz]')
    plt.title(f'b{int(b)}ch{int(ch):03}')
    plt.tight_layout()

    plt.savefig(os.path.join(plot_dir,
        f'{timestamp}_noise_b{int(b)}ch{int(ch):03}.png'),
        bbox_inches='tight')
    plt.close()