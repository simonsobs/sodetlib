import pysmurf
import matplotlib.pyplot as plt
import scipy.signal as signal
import numpy as np
import os

plt.ioff()

config_file = '/usr/local/src/pysmurf/cfg_files/experiment_fp28_docker.cfg'
S = pysmurf.SmurfControl(make_logfile=False,epics_root='test_epics2',
                         cfg_file=config_file, offline=True)

datadir = '/data/smurf_data/20190418/1555618703/outputs/'
plotdir = datadir.replace('outputs', 'plots')

#filename = '1555622545.dat'
filename = '1555628193.dat'
fn = filename.split('.')[0]

datafile = os.path.join(datadir, filename)

t,d,m = S.read_stream_data(datafile, n_samp=10000)
_, n_res = np.shape(np.where(m!=-1))

d *= S.pA_per_phi0/(2*np.pi)  # phase to pA

# Take SVD
u, s, vh = S.noise_svd(d, m)

# ============================================================
# Totally done with SVD-ing. Everything below just diagnostics
# ============================================================


# Make summary plots
S.plot_svd_modes(vh, show_plot=False, save_plot=True,
                 save_name='{}_svd_modes.png'.format(fn))
S.plot_svd_summary(u, s, show_plot=False, save_plot=True,
                   save_name='{}_svd_summary.png'.format(fn))
plt.title('{}'.format(filename))

modes = 1
d_clean = S.remove_svd(d, m, u, s, vh, modes=modes)

nperseg = 2**12
dirty = np.zeros((n_res,3))
clean = np.zeros((n_res,3))

import yaml
with open(datafile.replace('.dat','.yml'), 'r') as stream:
    try:
        data_loaded = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

bands, chans = np.where(m!=-1)
freqs = np.zeros(len(bands))
for i, (b, c) in enumerate(zip(bands, chans)):
    subband = S.get_subband_from_channel(b, c)
    _, sbc = S.get_subband_centers(b, as_offset=False)
    offset = float(data_loaded['AMCc']['FpgaTopLevel']['AppTop']['AppCore']['SysgenCryo']['Base[{}]'.format(b)]['CryoChannels']['CryoChannel[{}]'.format(c)]['centerFrequencyMHz'])
    freqs[i] = sbc[subband] + offset

freqs += 2000 # High band hack
    
plt.ioff()
    
for i in np.arange(n_res):
#    fig, ax = plt.subplots(1,2, figsize=(10,4.5))
    
    f, pxx = signal.welch(d[i], nperseg=nperseg, fs=200)
    pxx = np.sqrt(pxx)
    popt, pcov, f_fit, pxx_fit = S.analyze_psd(f, pxx)
    dirty[i] = popt

#    ax[0].plot(d[i])
#    ax[1].plot(f, pxx, label='dirty')

    f, pxx = signal.welch(d_clean[i], nperseg=nperseg, fs=200)
    pxx = np.sqrt(pxx)
    popt, pcov, f_fit, pxx_fit = S.analyze_psd(f, pxx)
    clean[i] = popt

#    ax[0].plot(d_clean[i])
#    ax[1].plot(f, pxx, label='clean')

#    ax[1].set_yscale('log')
#    ax[0].set_ylabel('Amp [pA]')
#    ax[1].set_xlabel('Freq [Hz]')
#    ax[1].set_ylabel('pA/rtHz')

#    fig.suptitle('{} {:4.2f}'.format(filename, freqs[i]))
#    plt.savefig(os.path.join(plotdir,
#                             '{}_{:6}.png'.format(fn, int(freqs[i]*100))),
#                bbox_inches='tight')
#    plt.close()
    
#plt.figure(figsize=(8,4.5))
#plt.plot(freqs, dirty[:,0], '.', label='original')
#plt.plot(freqs, clean[:,0], 'x', label='- {} modes'.format(modes))
#plt.legend()
#plt.xlabel('Freq [MHz]')
#plt.ylabel('White Noise [pA/rtHz]')
#plt.yscale('log')
#plt.ylim((10,1000))
#plt.title(filename)
