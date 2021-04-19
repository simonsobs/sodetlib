import pysmurf.client
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.signal as signal

####
# Assumes you've already setup the system.
####

### System Configuration ###
epics_prefix = 'smurf_server_s5'
config_file = os.path.join('/data/pysmurf_cfg/experiment_fp30_cc02-03_lbOnlyBay0.cfg')
tune_file = '/data/smurf_data/tune/1590781150_tune.npy'

### Function variables ###
band = 2
channel = 443
nperseg = 2**17
reset_rate_khzs = np.array([4, 10, 15, 20, 21, 22, 23, 24, 25, 28, 29, 30, 31,
    32, 33, 34, 35, 37, 39, 41, 45, 50])
# n_phi0s = np.array([4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4])
n_phi0s = (np.ones(len(reset_rate_khzs)) * 4).astype(int)
lms_enable2 = False
lms_enable3 = False
lms_gain = 3
filter_order = 4
data_fs = 4000
plt.ion()

n_steps = len(reset_rate_khzs)

# Instatiate pysmurf object
S = pysmurf.client.SmurfControl(epics_root=epics_prefix, cfg_file=config_file,
    setup=False, make_logfile=False, shelf_manager='shm-smrf-sp01')

S.load_tune(filename=tune_file)
S.relock(band)
S.band_off(band)
S.set_amplitude_scale_channel(band, channel, 12)
S.run_serial_gradient_descent(band)
S.run_serial_eta_scan(band)

print(S.which_on(band))

I, Q, sync = S.take_debug_data(band=band, channel=channel, rf_iq=True,
    IQstream=False)
d = I * 1.j*Q
ff_nofr, pxx_nofr = signal.welch(d, fs=S.get_channel_frequency_mhz()*1.0E6,
    nperseg=nperseg)
idx = np.argsort(ff_nofr)

f = {}
df = {}
ff = {}
pxx = {}
noise_datafile = {}

_, _, _ = S.tracking_setup(band, reset_rate_khz=reset_rate_khzs[0],
    fraction_full_scale=.5, make_plot=True, show_plot=False, nsamp=2**18,
    lms_gain=lms_gain, lms_freq_hz=None, meas_lms_freq=False,
    feedback_start_frac=.25, feedback_end_frac=.98, meas_flux_ramp_amp=True,
    n_phi0=n_phi0s[0], lms_enable2=lms_enable2, lms_enable3=lms_enable3)
fraction_full_scale = S.get_fraction_full_scale()

for i in np.arange(n_steps):
    print(i, reset_rate_khzs[i])
    f[i], df[i], sync = S.tracking_setup(band, reset_rate_khz=reset_rate_khzs[i],
        fraction_full_scale=fraction_full_scale, make_plot=True,
        show_plot=False, nsamp=2**18,
        lms_gain=lms_gain, lms_freq_hz=n_phi0s[i]*reset_rate_khzs[i]*1.0E3,
        meas_lms_freq=False,
        feedback_start_frac=.25, feedback_end_frac=.98, meas_flux_ramp_amp=False,
        n_phi0=n_phi0s[i], lms_enable2=lms_enable2, lms_enable3=lms_enable3)
    print(i, reset_rate_khzs[i], S.get_flux_ramp_freq())

    I, Q, sync = S.take_debug_data(band=band, channel=channel, rf_iq=True)
    d = I * 1.j*Q

    S.set_downsample_factor(reset_rate_khzs[i]*1.0E3//data_fs)
    S.set_downsample_filter(4, data_fs/4)

    ff[i], pxx[i] = signal.welch(d, fs=S.get_channel_frequency_mhz()*1.0E6,
        nperseg=nperseg)
    noise_datafile[i] = S.take_stream_data(8)

noise = {}
bin_low = np.array([.5, 1, 2, 5, 10 ])
bin_high = np.array([1, 2, 5, 10, 20])
for i in np.arange(n_steps):
    t, d, m = S.read_stream_data(noise_datafile[i])
    chidx = m[band, channel]
    d *= S.pA_per_phi0 / 2 / np.pi
    ftmp, pxxtmp = signal.welch(d[chidx], fs=data_fs, nperseg=2**11)

    noise_tmp = np.zeros(len(bin_low))
    for j, (bl, bh) in enumerate(zip(bin_low, bin_high)):
        tmp_idx = np.where(np.logical_and(ftmp > bl, ftmp < bh))
        noise_tmp[j] = np.median(pxxtmp[tmp_idx])

    noise[i] = noise_tmp

cm = plt.get_cmap('viridis')
fig, ax = plt.subplots(2, figsize=(8,6.5), sharex=True)
for i in np.arange(n_steps):
    color = cm(i/n_steps)
    ax[0].semilogy(ff[i][idx]*1.0E-3, pxx[i][idx], color=color,
        label=f'{reset_rate_khzs[i]*n_phi0s[i]} kHz')
    ax[1].plot(reset_rate_khzs[i]*n_phi0s[i], np.sqrt(noise[i][-1]),'.',
        color=color)
ax[0].plot(ff_nofr[idx]*1.0E-3, pxx_nofr[idx], color='k', label='None')
ax[1].set_xlabel('Freq [kHz]')
ax[0].set_ylabel('Resp')
ax[1].set_ylabel('Noise [pA/rtHz]')
ax[0].legend(loc='upper right')
ax[1].set_xlim((-400, 400))
plt.show()
