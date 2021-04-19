import numpy as np
import pysmurf
import glob
import os

filedir = '/home/common/data/cpu-b000-hp01/cryo_data/data2/20180926/'+\
    '1538008791/outputs'
# filenames = ['1537924144.dat', '1537924208.dat']
bias = np.arange(6,3-.5, -.5)

filenames = np.sort(glob.glob(os.path.join(filedir, "*.dat")))
filenames = filenames[-len(bias):]

S = pysmurf.SmurfControl(setup=False, make_logfile=False)

# bias = np.array([8, 7.8])
datafile = np.array([os.path.join(filedir, x) for x in filenames])

print(bias)
print(datafile)

S.analyze_noise_vs_bias(bias, datafile, channel=np.array([0, 16, 32, 64, 139,
    147, 179, 203, 395, 415, 427, 459]), band=3, 
    fs=4.0E3)