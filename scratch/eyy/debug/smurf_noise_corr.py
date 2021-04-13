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
                         no_dir=True)

t, d, m = S.read_stream_data(datafile)

# Extract useful values                                                                     
_, filename = os.path.split(datafile)
timestamp = filename.split('.')[0]

# Channels with IV curves                                                                   
ivch = np.array([16,32,64,165,171,179,197,203,213,222,256,389,395,398,415,421,427,447])
d = d[m[2][ivch]]

# mean subtract
d = (d.T - np.mean(d, axis=1)).T

n_det, n_samp = np.shape(d)

nperseg = 2048
fs = 200
psd = np.zeros((n_det, n_det, nperseg//2+1), dtype='complex')
for i in np.arange(n_det):
    for j in np.arange(n_det):
        if i > j:
            psd[i,j] = psd[j,i]
        else:
            f, psd[i,j] = signal.csd(d[i], d[j], nperseg=nperseg, fs=fs)
        
# Make plots
#for i in np.arange(n_det):
#    for j in np.arange(1, n_det):
#        fig, ax = plt.subplots(2, sharex=True)
#        ax[0].loglog(f, np.abs(psd[i,i]), '.', label='Det {:02}'.format(i))
#        ax[0].loglog(f, np.abs(psd[j,j]), '.', label='Det {:02}'.format(j))
#        ax[0].loglog(f, np.abs(psd[i,j]), '.k', label='{:02} x {:02}'.format(i,j))
#        ax[0].legend(loc='upper right')

#        ax[0].set_ylabel('Amp')

#        idx = np.ravel(np.where(f<59))  # Lazy way to ignore 60 Hz line
#        ax[1].plot(f[idx], np.real(psd[i,j,idx]), '.', label='real')
#        ax[1].plot(f[idx], np.imag(psd[i,j,idx]), '.', label='imag')
#        ax[1].legend(loc='upper right')

#        ax[1].set_ylabel('Amp')
#        ax[1].set_xlabel('Freq [Hz]')
        
#        plt.savefig(os.path.join(datadir.replace('outputs', 'plots'), 
#                                 '{}_{:02}x{:02}.png'.format(timestamp, i, j)),
#                    bbox_inches='tight')
#        plt.close()


idx_lf = np.where(np.logical_and(f>.5, f<2))[0]
psd_lf = np.mean(psd[:,:,idx_lf], axis=2)
normalization_lf = np.sqrt(np.real(np.outer(np.diag(psd_lf), np.diag(psd_lf))))
psd_lf = np.real(psd_lf)/normalization_lf

idx_hf = np.where(np.logical_and(f>5, f<10))[0]
psd_hf = np.mean(psd[:,:,idx_hf], axis=2)
normalization_hf = np.sqrt(np.real(np.outer(np.diag(psd_hf), np.diag(psd_hf))))
psd_hf = np.real(psd_hf)/normalization_hf

idx_60 = np.where(np.logical_and(f>59, f<61))[0]
psd_60 = np.mean(psd[:,:,idx_60], axis=2)
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


# Now do all the channels in frequency order
cor_mat = np.zeros((2*512, 2*15, nperseg//2+1))

bands = np.array([2,3])
subbands = np.arange(128)
counter = 0
projection = np.zeros()
for b in bands:
    for sb in subbands:
        chs = S.get_channels_in_subband(b, sb)
        for i, ch in enumerate(chs):
            if m[b, ch] != -1:
                cor_mat[counter] =
