import pysmurf
import numpy as np
import matplotlib.pyplot as plt

S = pysmurf.SmurfControl(setup=False,
                         cfg_file='/usr/local/controls/Applications/smurf/pysmurf/pysmurf/cfg_files/experiment_kx_mapodaq.cfg',make_logfile=False)

tones = np.arange(5,16)
freq = 5193.799997711181
resp = np.zeros((len(tones), 201), dtype=complex)

for i, t in enumerate(tones):
    print('Tone {}'.format(t))
    f, resp[i], eta = S.eta_estimator(2, freq, t, f_sweep_half=.3,
                              df_sweep=.003)

fig, ax = plt.subplots(3, sharex=True, figsize=(5,8))
cm = plt.get_cmap('viridis')
for i, t in enumerate(tones):
    color = cm(i/len(tones))
    ax[0].plot(f, np.abs(resp[i]), color=color, label='{:02}'.format(t))
    ax[1].plot(f, np.real(resp[i]), color=color)
    ax[2].plot(f, np.imag(resp[i]), color=color)

ax[0].legend(loc='upper left', ncol=2)
ax[0].set_ylabel('Abs')
ax[1].set_ylabel('Real')
ax[2].set_ylabel('Imag')
ax[2].set_xlabel('Freq [MHz]')
plt.tight_layout()
