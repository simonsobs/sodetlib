import numpy as np
import matplotlib.pyplot as plt

ivs = np.load('/home/common/data/cpu-b000-hp01/cryo_data/data2/20180924/' + \
    '1537827343/outputs/1537827354_iv.npy').item()

bias = ivs['bias']
channels = np.array([0, 16, 32, 64, 139, 147, 179, 203, 245, 288, 395, 415, 427,
    447, 459, 494])

print(bias)

fig, ax = plt.subplots(1)
# cm = plt.cm('viridis')
for i, ch in enumerate(channels):
    # color = float(i)/len(channels)
    ax.plot(bias[::-1], ivs[ch]['R']/ivs[ch]['Rn'],
        label='Ch {:03}'.format(ch))

ax.legend()
ax.set_ylim(0,1.1)
ax.set_xlim(0,10)
ax.set_xlabel('Commanded bias [V]')
ax.set_ylabel(r'R/R_N')