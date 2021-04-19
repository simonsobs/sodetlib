import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import os
import seaborn as sns
import glob
import pysmurf

datadir = '/data/smurf_data/20190216/1550347814/outputs'
datafile = os.path.join(datadir, '1550349384.dat')

S = pysmurf.SmurfControl(make_logfile=False,
                         epics_root='test_epics',
                         cfg_file='/usr/local/controls/Applications/'+\
                             'smurf/pysmurf/pysmurf/cfg_files/'+\
                             'experiment_fp28_smurfsrv04.cfg',
                         no_dir=True, offline=True)

t, d, m = S.read_stream_data(datafile)

# Extract useful values                                                                     
_, filename = os.path.split(datafile)
timestamp = filename.split('.')[0]

#d = d[np.where(m!=-1)]
_, n_samp = np.shape(d)
bands, channels = np.where(m!=-1)
dat = np.zeros((len(channels), n_samp))
for i, (b, ch) in enumerate(zip(bands, channels)):
    dat[i] = d[m[b,ch]]

print(np.shape(dat))

# mean subtract
dat = (dat.T - np.mean(dat, axis=1)).T

n_det, n_samp = np.shape(dat)

nperseg = 2048
fs = 200
psd = np.zeros((n_det, n_det, nperseg//2+1), dtype='complex')
for i in np.arange(n_det):
    for j in np.arange(n_det):
        if i > j:
            psd[i,j] = psd[j,i]
        else:
            f, psd[i,j] = signal.csd(dat[i], dat[j], nperseg=nperseg, fs=fs)

# Project into frequency order
cor_mat = np.zeros((2*512, 2*512, nperseg//2+1), dtype='complex') * np.nan

bands = np.array([2,3])
subbands = np.arange(128)
total_counter = 0
counter = 0
projection = -1 * np.ones(2*512, dtype=int)
bh = -1* np.ones_like(projection)
chh = -1 * np.ones_like(projection)

for b in bands:
    for sb in subbands:
        print(sb)
        chs = S.get_channels_in_subband(b, sb)
        for i, ch in enumerate(chs):
            if m[b, ch] != -1:
                if ch == 195:
                    print('HERE!! {} {}'.format(b, ch))
                print(b, ch, total_counter, counter)
                bh[total_counter] = b
                chh[total_counter] = ch
                projection[total_counter] = counter
                counter += 1
            total_counter += 1

print(np.shape(projection))

for i, c in enumerate(projection):
    for j, d in enumerate(projection):
        #print(i, c, j, d)
        if c != -1 and d != -1:
            cor_mat[i,j] = psd[c,d]

idx_lf = np.where(np.logical_and(f>.5, f<2))[0]
psd_lf = np.mean(cor_mat[:,:,idx_lf], axis=2)
normalization_lf = np.sqrt(np.real(np.outer(np.diag(psd_lf), np.diag(psd_lf))))
psd_lf = np.real(psd_lf)/normalization_lf

idx_hf = np.where(np.logical_and(f>5, f<10))[0]
psd_hf = np.mean(cor_mat[:,:,idx_hf], axis=2)
normalization_hf = np.sqrt(np.real(np.outer(np.diag(psd_hf), np.diag(psd_hf))))
psd_hf = np.real(psd_hf)/normalization_hf

idx_60 = np.where(np.logical_and(f>59, f<61))[0]
psd_60 = np.mean(cor_mat[:,:,idx_60], axis=2)
normalization_60 = np.sqrt(np.real(np.outer(np.diag(psd_60), np.diag(psd_60))))
psd_60 = np.real(psd_60)/normalization_60

plt.figure()
sns.heatmap(psd_lf, cmap='RdBu', vmin=-.5, vmax=.5)
plt.title('0.5 - 2 Hz bin')
plt.savefig(os.path.join(datadir.replace('outputs', 'plots'),
                        '{}_corr_noise_.5-2.png'.format(timestamp)),
            bbox_inches='tight')
plt.show()

plt.figure()
sns.heatmap(psd_hf, cmap='RdBu', vmin=-.5, vmax=.5)
plt.title('5 - 10 Hz bin')
plt.savefig(os.path.join(datadir.replace('outputs', 'plots'),
                        '{}_corr_noise_5-10.png'.format(timestamp)),
            bbox_inches='tight')
plt.show()

plt.figure()
sns.heatmap(psd_60, cmap='RdBu', vmin=-.5, vmax=.5)
plt.title('60 Hz bin')
plt.savefig(os.path.join(datadir.replace('outputs', 'plots'),
                        '{}_corr_noise_60.png'.format(timestamp, i, j)),
            bbox_inches='tight')
plt.show()
