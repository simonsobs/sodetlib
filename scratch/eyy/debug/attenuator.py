import numpy as np
import pysmurf
import time
import matplotlib.pyplot as plt

S = pysmurf.SmurfControl(setup=False, make_logfile=False)

# band = 3

uc_att_vals = np.arange(0, 32, 7)
dc_att_vals = np.arange(0, 32, 7)
n_samples = 2**19
r = {}

for d, dc_att in enumerate(dc_att_vals):
    S.set_att_dc(band, dc_att, input_band=True, write_log=True)
    for u, uc_att in enumerate(uc_att_vals):
        # Set only upconverted values for now
        S.set_att_uc(band, uc_att, input_band=True, write_log=True)
        time.sleep(1)
        # f, p[i] = S.full_band_resp(band, n_samples=n_samples, make_plot=False)
        S.tune_band(band, n_samples=n_samples)
        r['uc{}_dc{}'.format(uc_att, dc_att)] = \
            [S.freq_resp[k]['r2'] for k in S.freq_resp.keys()]


# --------------------------------- Uncomment Above -----------------


# fig, ax = plt.subplots(1, figsize=(8,5))
# for k in r.keys():
#     ax.hist(r[k])

# cm = plt.get_cmap('viridis')

# fig, ax = plt.subplots(1, figsize=(8,5))
# for i, att in enumerate(att_vals):
#     color = cm(float(i)/len(att_vals))
#     ax.plot(f, np.abs(p[i]), label='att {}'.format(att), color=color)

# ax.set_yscale('log')
# r = np.load('/home/cryo/eyy/uc_dc_r.npy').item()
bins = np.arange(0, 1.01, .05)

fig, ax = plt.subplots(5, 5, figsize=(10,7), sharex=True, sharey=True)
for d, dc_att in enumerate(dc_att_vals):
    for u, uc_att in enumerate(uc_att_vals):
        y = d
        x = u
        ax[y,x].hist(r['uc{}_dc{}'.format(uc_att, dc_att)], bins=bins)
        if x == 0:
            ax[y,x].set_ylabel('DC {}'.format(dc_att))
        if y == 4:
            ax[y,x].set_xlabel('UC {}'.format(uc_att))

plt.tight_layout()

# ax.legend()

plt.show()