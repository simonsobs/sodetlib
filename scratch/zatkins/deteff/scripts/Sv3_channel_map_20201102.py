#!/usr/bin/env python3

import numpy as np
import pandas as pd
import scipy
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt 

# define inputs
smurf_band = 1
title_add = 'mux_band_4_subband_c_d'
minmax_dict = {
    1: (4.65e9, 4.72e9),
    3: (5.782e9, 5.855e9)
}

# define static variables
offset_dict = {
    1: (-108.5e6, -105.75e6),
    3: (-132.25e6, -129.5e6)
}

tbath = 70.

path_to_data = '/home/zatkins/repos/stslab/testbed_optics/deteff/data/misc'

path_to_dark = '/home/zatkins/so/data/vna_data/20200801_UFMv3prime/100mK'
dark_fns = {
    1: ['1596399079.75-F_4.40_TO_4.60-BW_100.0-ATTEN_46.7-VOLTS_0.000.CSV',
    '1596399346.93-F_4.60_TO_4.80-BW_100.0-ATTEN_46.1-VOLTS_0.000.CSV',
    '1596399614.11-F_4.80_TO_5.00-BW_100.0-ATTEN_45.5-VOLTS_0.000.CSV'],
    3: ['1596400415.75-F_5.40_TO_5.60-BW_100.0-ATTEN_43.8-VOLTS_0.000.CSV',
    '1596400682.94-F_5.60_TO_5.80-BW_100.0-ATTEN_43.2-VOLTS_0.000.CSV',
    '1596400950.19-F_5.80_TO_6.00-BW_100.0-ATTEN_42.6-VOLTS_0.000.CSV']
}

path_to_opt = '/home/zatkins/so/data/vna_data/CL_=_16.6K_UFMv3prime-cold-9_22'
opt_fns = {
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

path_to_save = '/home/zatkins/so/cold_load/analysis/plots/'

# below from Kaiwen Zheng
#########################
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

def s21_find_baseline(fs, s21, avg_over=800):
    #freqsarr and s21arr are your frequency and transmission
    #average the data every avg_over points to find the baseline
    #of s21.
    #written by Heather, modified so that number of datapoints
    #doesn't have to be multiples of avg_over.
    num_points_all = s21.shape[0]
    num_2 = num_points_all%avg_over
    num_1= num_points_all-num_2
    s21_reshaped = s21[:num_1].reshape(num_1//avg_over, avg_over)
    fs_reshaped = fs[:num_1].reshape(num_1//avg_over, avg_over)
    #s21_avg = s21_reshaped.mean(1)
    #fs_avg = fs_reshaped.mean(1)
    x = np.squeeze(np.median(fs_reshaped, axis=1))
    y = np.squeeze(np.amax(s21_reshaped, axis=1))
    if (num_2 !=0):
        x2=np.median(fs[num_1:num_points_all])
        y2=np.amax(s21[num_1:num_points_all])
        x=np.append(x,x2)
        y=np.append(y,y2)
    tck = scipy.interpolate.splrep(x, y, s=0)
    ynew = scipy.interpolate.splev(fs, tck, der=0)
    return ynew

def correct_trend(freq,real,imag,avg_over=800):
	#Input the real and imaginary part of a s21
	#Out put the s21 in db, without the trend. 
    s21=real+1j*imag
    s21_db=20*np.log10(np.abs(s21))
    baseline=s21_find_baseline(freq, s21, avg_over)
    bl_db=20*np.log10(baseline)
    s21_corrected=s21_db-bl_db
    return s21_corrected
#####################################

# get S21
dark_fn_list = [f'{path_to_dark}/{fn}' for fn in dark_fns[smurf_band]]
dark_f,r,i = read_vna_data_array(dark_fn_list)
dark_s21 = correct_trend(dark_f, r, i, avg_over = 1000)
dark_s21_func = interp1d(dark_f, dark_s21)

# get S21
opt_fn_list = [f'{path_to_opt}/{fn}' for fn in opt_fns[smurf_band]]
opt_f,r,i = read_vna_data_array(opt_fn_list)
opt_s21 = correct_trend(opt_f, r, i, avg_over = 1000)
opt_s21_func = interp1d(opt_f, opt_s21)

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
low, high = minmax_dict[smurf_band]
dark_offset, opt_offset = offset_dict[smurf_band]

plt.figure(figsize=(32,8))
plt.plot(dark_f + dark_offset, dark_s21, label = f'Dark S21, offset = {dark_offset / 1e6} MHz')
plt.plot(opt_f + opt_offset, opt_s21, label = f'Opt S21, offset = {opt_offset / 1e6} MHz')

ax = plt.gca()
label = 'start'
for b in np.atleast_1d(mux_band):
    mask = np.logical_or(mux['FrequencyMHz'] < low, mux['FrequencyMHz'] > high)
    mask = np.logical_or(mask, mux['Band'].astype(int) != b)
    pads = mux['Pad'].astype(int)[~mask]
    design_freqs = mux['FrequencyMHz'][~mask]
    for i, freq in enumerate(design_freqs):
        if label == 'start':
            label = 'Design frequency'
        else:
            label = ''
        ax.axvline(x = freq, color = 'grey', label = label)
        plt.annotate(f'{b},{pads[i]}', (freq, -50))

# plot actual freqs and good chans
_, _, idxs = np.intersect1d(good_chans, smurf_chans, return_indices=True)
plt.scatter(actual_freqs + opt_offset, opt_s21_func(actual_freqs), c = 'black', marker = 'x', label='all smurf channels')
plt.scatter(actual_freqs[idxs] + opt_offset, opt_s21_func(actual_freqs[idxs]), color = 'red', marker = 'o', s = 50, label = 'good smurf channels')
for i, freq in enumerate(actual_freqs[idxs]):
    if np.logical_and(low < freq + opt_offset, freq + opt_offset < high):
        plt.annotate(f'{smurf_chans[idxs[i]]}', (freq + opt_offset, opt_s21_func(freq) - 2), color = 'red')

# plot details
plt.title(f'{tune_dict[smurf_band].split(".")[0]} {title_add}')
plt.legend()   
plt.xlim(low, high)

plt.savefig(f'{path_to_save}/{tune_dict[smurf_band].split(".")[0]} {title_add}.png', bbox_inches = 'tight')