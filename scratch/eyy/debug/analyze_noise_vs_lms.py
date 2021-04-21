import numpy as np
import matplotlib.pyplot as plt
import pysmurf
import scipy.signal as signal
import os

S = pysmurf.SmurfControl(setup=False,cfg_file='/usr/local/controls/Applications/smurf/pysmurf/pysmurf/cfg_files/experiment_kx_mapodaq.cfg',make_logfile=False)

savedir = '/home/cryo/ey/noise_vs_lms'

lms_freq = np.array([12605, 12505, 12405, 12705, 12805])

datafile = np.array([
        '/data/smurf_data/20181218/1545116665/outputs/1545118168.dat',
        '/data/smurf_data/20181218/1545116665/outputs/1545118358.dat',
        '/data/smurf_data/20181218/1545116665/outputs/1545118743.dat',
        '/data/smurf_data/20181218/1545116665/outputs/1545118907.dat',
        '/data/smurf_data/20181218/1545116665/outputs/1545119044.dat'
])

nperseg=2**12
fs = 180.

#for lms, dfile in zip(lms_freq, datafile):
#    t, d, m = S.read_stream_data(dfile)
#    d *= S.pA_per_phi0/(2*np.pi)
#    band, channel = np.where(m != -1)
#    for b, ch in zip(band, channel):
#        f, pxx = signal.welch(d[m[b,ch]], nperseg=nperseg,
#                              fs=fs)
#        np.save(os.path.join(savedir, 'lms{}_b{}_ch{:03}'.format(lms, b, ch)), pxx)
#    np.save(os.path.join(savedir, 'lms{}_f'.format(lms)),f)


plt.ioff()

cm = plt.get_cmap('viridis')
# Just horribly abusing scope...
f = np.load(os.path.join(savedir, 'lms{}_f.npy'.format(lms_freq[0])))
mask = S.make_mask_dict(datafile[0].replace('.dat', '_mask.txt'))
band, channel = np.where(mask != -1)
wl_hold = np.zeros((len(lms_freq), len(band)))
counter = 0
for b, ch in zip(band, channel):
    #plt.figure()
    for i, lms in enumerate(lms_freq):
        print(i, lms)
        color = cm(i/5)
        pxx = np.load(os.path.join(savedir, 
                                   'lms{}_b{}_ch{:03}.npy'.format(lms, b, ch)))
        pxx = np.sqrt(pxx)
        popt, pcov, f_fit, Pxx_fit = S.analyze_psd(f, pxx)
        wl, n, f_knee = popt
        wl_hold[i, counter] = wl

    counter += 1
        #plt.semilogy(f, pxx, color=color, label='{} {:4.1f}'.format(lms, wl))
        #plt.axhline(wl, color=color, linestyle=':')
    #plt.title('Band {} Ch {:03}'.format(b, ch))
    #plt.xlabel('Freq [Hz]')
    #plt.ylabel(r'pA/\sqrt{Hz}')
    #plt.legend()
    #plt.savefig(os.path.join(savedir, 'b{}_ch{:03}.png'.format(b, ch)), 
    #            bbox_inches='tight')
    #plt.close()
