import pysmurf
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os
import seaborn as sns
import glob

S = pysmurf.SmurfControl(make_logfile=False,
                         epics_root='test_epics',
                         cfg_file='/usr/local/controls/Applications/'+\
                             'smurf/pysmurf/pysmurf/cfg_files/'+\
                             'experiment_fp28_smurfsrv04.cfg',
                         no_dir=True)

datadir = '/data/smurf_data/20190216/1550347814/outputs'
datafiles = glob.glob(os.path.join(datadir, '*.dat'))

#datafile = os.path.join(datadir, '1550348586.dat')
for datafile in datafiles:
    t, d, m = S.read_stream_data(datafile)
    
    # Extract useful values
    dirname, filename = os.path.split(datafile)
    timestamp = filename.split('.')[0]

    # Channels with IV curves
    ivch = np.array([16,32,64,165,171,179,197,203,213,222,256,389,395,398,415,421,427,447])
    d = d[m[2][ivch]]

    # Take PCA
    pca = PCA(svd_solver='full')
    pca.fit(d.T)
    d2 = pca.transform(d.T).T
    coeff = pca.components_.T
    
    fig, ax = plt.subplots(6,3, figsize=(12,12), sharex=True)
    for i in np.arange(18):
        y = i // 3
        x = i % 3
        
        ax[y,x].plot(d2[i])
        ax[y,x].set_title('Mode {}'.format(i))
        if y == 5:
            ax[y,x].set_xlabel('Sample')
        if x == 0:
            ax[y,x].set_ylabel('Amp')
    fig.suptitle(timestamp)

    plt.savefig(os.path.join(dirname, '{}_modes.png'.format(timestamp)),
            bbox_inches='tight')
    plt.close()

    fig, ax = plt.subplots(2,1, figsize=(5,7));
    sns.heatmap(coeff, cmap='RdBu', vmin=-1, vmax=1, ax=ax[0])
    ax[0].set_xlabel('Mode')
    ax[0].set_ylabel('Channel')
    ax[1].plot(pca.explained_variance_ratio_, '.')
    ax[1].set_xlabel('Channel')
    ax[1].set_ylabel('Variance ratio')
    fig.suptitle(timestamp)
    plt.savefig(os.path.join(dirname, '{}_amplitudes.png'.format(timestamp)),
                bbox_inches='tight')

    plt.close()
#cm = plt.get_cmap('viridis')
#n_mode = 5
#for i, ch in enumerate(ivch):
#    plt.figure()
#    plt.plot(d[i]-np.mean(d[i]), 'k')
#    plt.title(ch)
#    for j in np.arange(n_mode):
#        plt.plot(coeff[j,i]*d2[i], color=cm(j/n_mode), alpha=.5,
#                 label='Mode {}'.format(j))
#    plt.legend()

#    plt.savefig(os.path.join(dirname, '{}_ch{:03}.png'.format(timestamp, ch)),
#                bbox_inches='tight')
#    plt.close()

