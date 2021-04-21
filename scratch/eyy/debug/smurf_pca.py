import pysmurf
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os
import seaborn as sns

S = pysmurf.SmurfControl(make_logfile=False,
                         epics_root='test_epics',
                         cfg_file='/usr/local/controls/Applications/'+\
                             'smurf/pysmurf/pysmurf/cfg_files/'+\
                             'experiment_fp28_smurfsrv04.cfg')

datafile = S.take_stream_data(5)
t, d, m = S.read_stream_data(datafile)

ivch = np.array([16,32,64,165,171,179,197,203,213,222,256,389,395,398,415,421,427,447])
d = d[m[2][ivch]]
# d = d[m[np.where(m!=-1)]]
#print(np.shape(d))

pca = PCA(svd_solver='full')
pca.fit(d.T)
d2 = pca.transform(d.T).T
print(np.shape(d2))

fig, ax = plt.subplots(3,4, figsize=(12,7), sharex=True)

for i in np.arange(12):
    y = i // 4
    x = i % 4

    ax[y,x].plot(d2[i])
    ax[y,x].set_title('Mode {}'.format(i))

dirname, filename = os.path.split(datafile)
timestamp = filename.split('.')[0]
plt.savefig(os.path.join(dirname, '{}_modes.png'.format(timestamp)),
            bbox_inches='tight')

plt.show()


cm = plt.get_cmap('viridis')

for i, ch in enumerate(ivch):
    plt.figure()
    plt.plot(d[m[i], 'k')
    plt.title(ch)
    plt.savefig(os.path.join(dirname, '{}_ch{:03}.png'.format(timestamp, ch)),
                bbox_inches='tight')
    plt.close()

