#!/usr/bin/env python3

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt 

# define inputs
smurf_band = 3
tbath = 70.0
offset = -131.5e6

# define static variables
path_to_vna = '/home/zatkins/so/data/vna_data/CL_=_16.6K_UFMv3prime-cold-9_22'
path_to_data = '/home/zatkins/repos/stslab/testbed_optics/deteff/data/misc'
fns = {
    1: ['1600798129.79-F_4.40_TO_4.60-BW_100.0-ATTEN_45.3-VOLTS_0.000.CSV',
    '1600798396.97-F_4.60_TO_4.80-BW_100.0-ATTEN_44.8-VOLTS_0.000.CSV',
    '1600798664.18-F_4.80_TO_5.00-BW_100.0-ATTEN_44.3-VOLTS_0.000.CSV'],
    3: ['1600799465.71-F_5.40_TO_5.60-BW_100.0-ATTEN_42.9-VOLTS_0.000.CSV',
    '1600799732.80-F_5.60_TO_5.80-BW_100.0-ATTEN_42.5-VOLTS_0.000.CSV',
    '1600799999.93-F_5.80_TO_6.00-BW_100.0-ATTEN_42.0-VOLTS_0.000.CSV']
}
mux_band_dict = {
    1: [4],
    3: [11, 12]
}
tune_dict = {
    1: '1599786596_channel_assignment_b1.txt',
    3: '1599787547_channel_assignment_b3.txt'
}
minmax_dict = {
    1: (4.55e9, 4.75e9),
    3: (5.7825e9, 5.86e9)
}

# below from Kaiwen Zheng
def read_vna_data(filename):
	# Reads vna data in s2p or csv format
	# outputs frequency, real and imaginary parts
	# You should use the function below instead.
    if filename.endswith('S2P'):
        s2pdata = rf.Network(filename)
        freq=np.array(s2pdata.frequency.f)
        real=np.squeeze(s2pdata.s21.s_re)
        imag=np.squeeze(s2pdata.s21.s_im)
    elif filename.endswith('CSV'):
        csvdata=pd.read_csv(filename,header=2)
        freq=np.array(csvdata['Frequency'])
        real=np.array(csvdata[' Formatted Data'])
        imag=np.array(csvdata[' Formatted Data.1'])
    else:
        freq=0;real=0;imag=0
        print('invalid file type')
    return freq,real,imag

# below from Kaiwen Zheng
def read_vna_data_array(filenames):
	# Input an array of vna filenames or just one file
	# Outputs all data in the file, organized by frequency
    if np.array([filenames]).size==1:
        freq,real,imag=read_vna_data(filenames)
    elif np.array([filenames]).size>1:
            freq=np.array([])
            real=np.array([])
            imag=np.array([])
            for onefile in list(filenames):
                ft,rt,it=read_vna_data(onefile)
                freq=np.append(freq,ft)
                real=np.append(real,rt)
                imag=np.append(imag,it)
    L=sorted(zip(freq,real,imag))
    f,r,i=zip(*L)
    return np.array(f),np.array(r),np.array(i)
    #return freq,real,imag

# get S21
fn_list = [f'{path_to_vna}/{fn}' for fn in fns[smurf_band]]
f,r,i = read_vna_data_array(fn_list)
s21 = 10*np.log10(r**2 + i**2)
s21_func = interp1d(f,s21)

# get mux data
mux_band = mux_band_dict[smurf_band]
mux = np.genfromtxt(f'{path_to_data}/v3_2_mux.txt', delimiter = ',', names = True)

# get smurf channels
tune_file = np.loadtxt(f'{path_to_data}/{tune_dict[smurf_band]}', delimiter=',')
actual_freqs, _, smurf_chans, _ = tune_file.T
actual_freqs *= 1e6
smurf_chans = smurf_chans.astype(int)

good_chans = np.load(f'{path_to_data}/min_goodchans.npy', allow_pickle=True).item()[smurf_band][tbath]

# plot S21, design freqs
plt.figure(figsize=(32,8))
plt.plot(f + offset, s21, label = f'S21, offset = {offset / 1e6} MHz')
ax = plt.gca()
label = 'start'
for b in np.atleast_1d(mux_band):
    mask = mux['Band'].astype(int) != b
    design_freqs = mux['FrequencyMHz'][~mask]
    for freq in design_freqs:
        if label == 'start': label = 'Design frequency'
        else: label = ''
        ax.axvline(x = freq, color = 'grey', label = label)

# plot actual freqs and good chans
_, _, idxs = np.intersect1d(good_chans, smurf_chans, return_indices=True)
plt.scatter(actual_freqs + offset, s21_func(actual_freqs), c = 'black', marker = 'x', label='all channels')
plt.scatter(actual_freqs[idxs] + offset, s21_func(actual_freqs[idxs]), color = 'red', marker = 'o', s = 50, label = 'good channels')

# plot details
plt.title(f'{tune_dict[smurf_band].split(".")[0]}')
plt.legend()   
plt.xlim(minmax_dict[smurf_band])