import numpy as np
import pysmurf
import matplotlib.pyplot as plt


# datafile = '/home/common/data/cpu-b000-hp01/cryo_data/data2/20180921/1537561706/outputs/1537562122.dat'
# datafile = '/home/common/data/cpu-b000-hp01/cryo_data/data2/20180921/1537569911/outputs/1537569951.dat'
datafile = '/home/common/data/cpu-b000-hp01/cryo_data/data2/20180924/1537818448/outputs/1537818492.dat'
data_dir = '/home/common/data/cpu-b000-hp01/cryo_data/data2/20180924/1537818448/outputs/'
S = pysmurf.SmurfControl(make_logfile=False, setup=False, 
    data_dir=data_dir)

print('Loading data')
timestamp, I, Q = S.read_stream_data(datafile)

# ch = 0
# chs = np.array([0, 16, 32, 64, 139, 147, 179, 203, 245, 288, 395, 398, 415, 
#     427, 447, 459])
chs = np.array([16])
# chs = np.arange(16)

bias = np.arange(19.9, 0, -.1)

ivs = {}
ivs['bias'] = bias
for ch in chs:
    phase = S.iq_to_phase(I[ch], Q[ch]) * 1.443

    print('Running IV analysis')
    r, rn, idx = S.analyze_iv(bias, phase, make_plot=True, 
        show_plot=True, save_plot=True, band=3, channel=ch, 
        basename='1537818492')
    ivs[ch] = {
        'R' : r,
        'Rn' : rn,
        'idx': idx
    }

    plt.show()
