import numpy as np
import matplotlib.pyplot as plt
import pysmurf

# This is an analysis script run on smurf-srv03. It compares
# The tuning from gradient_descent and find_freq + setup_notches.

slot=4
epics_prefix = 'smurf_server_s4'
config_file='/data/pysmurf_cfg/experiment_fp29_srv03_dspv3_cc02-02_lbOnlyBay0.cfg'

S = pysmurf.SmurfControl(epics_root=epics_prefix,
                         cfg_file=config_file,make_logfile=False)

# Load find_freq tuning file  - band 2 file
band = 2
S.load_tune('/data/smurf_data/tune/1558465619_tune.npy')
S.relock(band)

ch_ff = S.which_on(band)
freq_ff = np.zeros(len(ch_ff), dtype=float)
for i, ch in enumerate(ch_ff):
    freq_ff[i] = S.channel_to_freq(band, ch)

# Load serial data
S.band_off(band)
#S.load_tune('/data/smurf_data/tune/1558468360_tune.npy')
S.load_tune('/data/smurf_data/tune/1558473416_tune.npy')
S.relock(band)

ch_ser = S.which_on(band)
freq_ser = np.zeros(len(ch_ser))
for i, ch in enumerate(ch_ser):
    freq_ser[i] = S.channel_to_freq(band, ch)

ch_intersect = np.intersect1d(ch_ff, ch_ser)

idx_ff = np.zeros_like(ch_intersect)
idx_ser = np.zeros_like(ch_intersect)

for i, ch in enumerate(ch_intersect):
    idx_ff[i] = np.where(ch == ch_ff)[0]
    idx_ser[i] = np.where(ch == ch_ser)[0]

plt.figure(figsize=(6,3.5))
plt.axhline(0, linestyle=':', color='k')
plt.plot(freq_ser[idx_ser],freq_ff[idx_ff]-freq_ser[idx_ser],'.')
plt.ylim(-.05,.05)
plt.xlabel('Freq [MHz]')
plt.ylabel('find_freq - serial [MHz]')
plt.tight_layout()

df = freq_ff[idx_ff]-freq_ser[idx_ser]


