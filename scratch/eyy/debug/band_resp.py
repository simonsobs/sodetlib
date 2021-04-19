import numpy as np
import glob
import os
import matplotlib.pyplot as plt

#fileroot = '/home/common/data/cpu-b000-hp01/cryo_data/data2/20180920/' + \
#    '1537461840/outputs'
fileroot = '/data/smurf_data/20181107/1541621474/outputs'
files = glob.glob(os.path.join(fileroot, '*freq_full_band_resp.txt'))
#files = np.array(['/home/common/data/cpu-b000-hp01/cryo_data/data2/20180920/1537461840/outputs/1537461850_freq_full_band_resp.txt'])
n_files = len(files)
freq = np.loadtxt(files[0])
idx = np.argsort(freq)
freq = freq[idx]
n_pts = len(freq)

resp = np.zeros((n_files, n_pts), dtype=complex)

for i, f in enumerate(files):
    print('loading data from {}'.format(f))
    resp[i] = (np.loadtxt(f.replace('freq', 'real')) + \
        1.j*np.loadtxt(f.replace('freq', 'imag')))[idx]

fig, ax = plt.subplots(3,3, sharex=True, sharey=True, figsize=(12,9))
resp_mean = np.mean(resp, axis=0)
cm = plt.get_cmap('viridis')
for i in np.arange(n_files):
    y = i//3
    x = i%3
    ax[y,x].semilogy(freq, np.abs(resp[i]))
    ax[y,x].set_title('Att {}'.format(i*3))
    ax[y,x].plot(freq, np.abs(resp_mean))

plt.tight_layout()

#import pysmurf
#S = pysmurf.SmurfControl()

#grad_loc = S.find_peak(freq, resp)
#fig, ax = plt.subplots(1)
#ax.plot(freq, np.abs(resp), '-bD', markevery=grad_loc)
#ax.set_yscale('log')

