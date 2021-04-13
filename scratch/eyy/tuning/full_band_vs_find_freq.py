import pysmurf
import matplotlib.pyplot as plt
import numpy as np

slot=4
epics_prefix = 'smurf_server_s4'
config_file='/data/pysmurf_cfg/experiment_fp29_srv03_dspv3_cc02-02_lbOnlyBay0.cfg'

S = pysmurf.SmurfControl(epics_root=epics_prefix,
                         cfg_file=config_file,make_logfile=False)

band = 2

S.set_att_dc(band, 0)
uc = np.arange(0,31,6)

resp = np.zeros((len(uc), 262144), dtype=complex)

for i, u in enumerate(uc):
    S.set_att_uc(band, u, wait_after=1.0)
    f, resp[i] = S.full_band_resp(band, make_plot=False)

f *= 1.0E-6  # convert ot MHz

subbands = np.arange(45,70)
S.set_att_uc(band, 24)
S.find_freq(band, subbands, drive_power=13)


# Make plots
cm = plt.get_cmap('plasma')
fig, ax = plt.subplots(2, figsize=(8,5.5), sharex=True)
for i, u in enumerate(uc):
    color = cm(float(i/len(uc)))
    ax[0].plot(f, np.abs(resp[i]), color=color, label='UC {:02}'.format(u))
ax[0].set_ylim((0, 5))

for sb in subbands:
    ff = S.freq_resp[band]['find_freq']['f'][sb]
    rr = S.freq_resp[band]['find_freq']['resp'][sb]
    ax[1].plot(ff, np.abs(rr), '.k')
    
ax[1].set_xlabel('Freq [MHz]')
ax[0].set_ylabel('full_band_resp')
ax[1].set_ylabel('find_freq')
ax[0].legend()
