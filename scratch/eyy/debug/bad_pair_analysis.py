import numpy as np
import matplotlib.pyplot as plt
import os

f_cutoff = .25
df_cutoff = .05

data_dir = '/data/smurf_data/20181214/1544843999/outputs'
f2, df2 = np.load(os.path.join(data_dir, 'band3_badres.npy'))
f2p, df2p = np.load(os.path.join(data_dir, 'band3_badpair.npy'))

m = np.ravel(np.where(np.logical_or(f2 > f_cutoff, df2 > df_cutoff)))

f2[m] = np.nan
df2[m] = np.nan
f2p[m,0] = np.nan
f2p[m-1,1] = np.nan
df2p[m,0] = np.nan
df2p[m-1,1] = np.nan

n, _ = np.shape(df2p)
xp = np.arange(1,n)

fig, ax = plt.subplots(2, 2, sharex=True, figsize=(8,7))

ax[0,0].plot(f2, color='k')
ax[0,0].plot(f2p[:-1,0])
ax[0,0].plot(xp, f2p[:-1, 1])
ax[0,0].set_title('f')


ax[0,1].plot(df2, color='k', label='Solo')
ax[0,1].plot(df2p[:-1,0], label='R on')
ax[0,1].plot(xp, df2p[:-1,1], label='L on')
ax[0,1].set_title('df')
ax[0,1].legend()

delta_ron_f2 = f2[:-1] - f2p[:-1,0]  # right on
delta_lon_f2 = f2[1:] - f2p[:-1,1]  # left one
ax[1,0].plot(delta_ron_f2)
ax[1,0].plot(xp, delta_lon_f2)

delta_ron_df2 = df2[:-1] - df2p[:-1,0]  # right on
delta_lon_df2 = df2[1:] - df2p[:-1,1]  # left one
ax[1,1].plot(delta_ron_df2)
ax[1,1].plot(xp, delta_lon_df2)
ax[1,0].set_xlabel('Res #')
ax[1,1].set_xlabel('Res #')

fig, ax = plt.subplots(1,2, figsize=(8, 4))

bins = np.arange(-.1, 0.06, .01)

hist_mask_r = np.where(~np.isnan(delta_ron_df2))
hist_mask_l = np.where(~np.isnan(delta_lon_df2))
ax[1].hist(delta_ron_df2[hist_mask_r], bins=bins,
    histtype='step', label='R on')
ax[1].hist(delta_lon_df2[hist_mask_l], bins=bins,
    histtype='step', label='L on')

ax[1].axvline(0, color='k', linestyle=':')
ax[1].legend()
# ax[2,1].hist(delta_lon_df2[])