"""
vna_func.py

Original Source: Kaiwen Zheng
Additional Author(s): Caleb Wheeler
"""

import os
import glob
import time
from functools import wraps

import scipy
import skrf as rf
import numpy as np
import pandas as pd

from timer_wrap import timing
from peak_finder_v2 import get_dip_depth, get_peaks_v2
from noise_analysis import fit_noise_model
from resonator_model import get_qi, get_br, full_fit
from read_stream_data_gcp_save import read_stream_data_gcp_save



@timing
def s21_find_baseline(fs, s21, avg_over=800):
    # freqsarr and s21arr are your frequency and transmission
    # average the data every avg_over points to find the baseline
    # of s21.
    # written by Heather, modified so that number of datapoints
    # doesn't have to be multiples of avg_over.

    num_points_all = s21.shape[0]
    num_2 = num_points_all % avg_over
    num_1 = num_points_all - num_2

    s21_reshaped = s21[:num_1].reshape(num_1 // avg_over, avg_over)
    fs_reshaped = fs[:num_1].reshape(num_1 // avg_over, avg_over)

    # s21_avg = s21_reshaped.mean(1)
    # fs_avg = fs_reshaped.mean(1)

    x = np.squeeze(np.median(fs_reshaped, axis=1))
    y = np.squeeze(np.amax(s21_reshaped, axis=1))

    if (num_2 != 0):
        x2 = np.median(fs[num_1:num_points_all])
        y2 = np.amax(s21[num_1:num_points_all])
        x = np.append(x, x2)
        y = np.append(y, y2)

    tck = scipy.interpolate.splrep(x, y, s=0)
    ynew = scipy.interpolate.splev(fs, tck, der=0)

    return ynew


@timing
def correct_trend(freq, real, imag, avg_over=800):
    # Input the real and imaginary part of a s21
    # Out put the s21 in db, without the trend.
    s21 = real + 1j * imag
    s21_db = 20 * np.log10(np.abs(s21))
    baseline = s21_find_baseline(freq, s21, avg_over)
    bl_db = bl_db = 20 * np.log10(baseline)
    s21_corrected = s21_db - bl_db
    return s21_corrected


@timing
def get_all_tune_timestamps(tune_files):
    # pass a list of tunefiles and
    # get a dict of the timestamps associated with the tune files
    timestamps = []
    for tune_file in tune_files:
        tune_file_only = os.path.basename(tune_file)
        tune_ts = int(tune_file_only.split('_')[0])
        timestamps.append(tune_ts)
    tune_file_dict = dict(zip(timestamps, tune_files))
    return (tune_file_dict)


@timing
def get_all_noise_timestamps(noise_files):
    # pass a list of noise files and
    # get a dict of the timestamps associated with them
    timestamps = []
    for noise_file in noise_files:
        noise_file_only = os.path.basename(noise_file)
        noise_ts = int(noise_file_only.split('.')[0])
        timestamps.append(noise_ts)
    noise_file_dict = dict(zip(timestamps, noise_files))
    return (noise_file_dict)


@timing
def find_nearest_tune_file(noise_file_dict, tune_file_dict, d_file):
    # Given two dictionaries of noise and tun files with key of ctime
    # Associate each noise file to tunning file and append to d_file
    noise_ts_arr = np.sort(list(noise_file_dict.keys()))
    tune_ts_arr = np.sort(list(tune_file_dict.keys()))

    for noise_ts in noise_ts_arr:
        try:
            closest_tune_ts = tune_ts_arr[tune_ts_arr < noise_ts].max()
            tune_file = tune_file_dict[closest_tune_ts]
            noise_file = noise_file_dict[noise_ts]
            d_file = d_file.append({'noise_ctime': str(noise_ts), 'tune_file': tune_file, 'noise_file': noise_file},
                                   ignore_index=True)
        except:
            pass
    return d_file


@timing
def match_file_names(tune_dir, noise_dir):
    # Pass the directory of tunning file and noise file
    # Associates all the files in the directory
    tune_files = glob.glob(tune_dir + '*tune.npy')
    noise_files = glob.glob(noise_dir + '*part*')
    tune_file_dict = get_all_tune_timestamps(tune_files)
    noise_file_dict = get_all_noise_timestamps(noise_files)
    d_file = {'noise_ctime': [], 'tune_file': [], 'noise_file': []}
    d_file = pd.DataFrame(d_file)
    d_file = find_nearest_tune_file(noise_file_dict, tune_file_dict, d_file)
    return d_file


@timing
def associate_noise_to_frame(d_file, fs=200, pA_per_phi0=9e6 / (2. * np.pi)):
    # Given a dataframe of associated noise and tunning files. Fit each
    # tunning file and append the fitting parameter to the frame.
    d_noise = {'noise_file': [], 'tune_file': [], 'file_index': [],
               'mask_index': [], 'Savg': [], 'Sw': [], 'fk': [], 'alpha': []}
    d_noise = pd.DataFrame(d_noise)

    for index, row in d_file.iterrows():
        timestream = row['noise_file']
        tune = row['tune_file']
        time, data, mask = read_stream_data_gcp_save(timestream)

        file_idx = index
        band, chan = np.where(mask != -1)
        for i in np.arange(len(chan)):
            try:
                ts_pA = (data[i] - np.mean(data[i])) * pA_per_phi0
                wn_average, fr, wl, f_knee, n = fit_noise_model(ts_pA, fs)
                d_noise = d_noise.append({'noise_file': timestream, 'tune_file': tune,
                                          'file_index': file_idx, 'mask_index': (band[i] * 512 + chan[i]),
                                          'Savg': wn_average, 'Sw': wl, 'fk': f_knee,
                                          'alpha': n}, ignore_index=True)
            except:
                pass

    return d_noise


@timing
def read_vna_data(filename):
    # Reads vna data in s2p or csv format
    # outputs frequency, real and imaginary parts
    # You should use the function below instead.

    if filename.endswith('S2P'):
        s2pdata = rf.Network(filename)
        freq = np.array(s2pdata.frequency.f)
        real = np.squeeze(s2pdata.s21.s_re)
        imag = np.squeeze(s2pdata.s21.s_im)
    elif filename.endswith('CSV'):
        csvdata = pd.read_csv(filename, header=2)
        freq = np.array(csvdata['Frequency'])
        real = np.array(csvdata[' Formatted Data'])
        imag = np.array(csvdata[' Formatted Data.1'])
    else:
        freq = 0
        real = 0
        imag = 0
        print('invalid file type')
    return freq, real, imag


@timing
def read_vna_data_array(filenames):
    # Input an array of vna filenames or just one file
    # Outputs all data in the file, organized by frequency
    if np.array([filenames]).size == 1:
        freq, real, imag = read_vna_data(filenames)
    elif np.array([filenames]).size > 1:
        freq = np.array([])
        real = np.array([])
        imag = np.array([])
        for onefile in list(filenames):
            ft, rt, it = read_vna_data(onefile)
            freq = np.append(freq, ft)
            real = np.append(real, rt)
            imag = np.append(imag, it)
    else:
        raise IndexError("When reading in VNA data an empty list of filenames was sent, this is not allowed.")
    L = sorted(zip(freq, real, imag))
    f, r, i = zip(*L)
    return np.array(f), np.array(r), np.array(i)
    # return freq,real,imag


@timing
def vna_data_into_frame(freq, real, imag, f0s, resonance_s21, low_indice=[], high_indice=[], delta=2e5):
    # This function takes in s21 data and position of the peak, 
    # fits the peaks into models and outputs a dataframe of the parameters
    # Input the frequency, real and imaginary data of s21
    # f0s and resonance_s21 are the frequency and magnitude of the peak
    # You can choose to input f_delta to specify the width of the peak.
    # Alternatively, you can choose to input two index arrays to specify the
    # left and right bounds of the peak (peak_finder_v2 will pass you these arrays.) 

    s21 = real + 1j * imag

    dres = {'resonator_index': [], 'f0': [], 'Qi': [],
            'Qc': [], 'Q': [], 'br': [], 'depth': []}
    dfres = pd.DataFrame(dres)
    k = 0
    for k in np.arange(len(np.array(f0s))):
        fs = f0s[k]
        if (len(low_indice) == 0):
            mask = (freq > (fs - delta)) & (freq < (fs + delta))
        else:
            mask = np.arange(low_indice[k], high_indice[k])
        f_res = freq[mask]
        s21_res = s21[mask]
        real_res = s21.real[mask]
        imag_res = s21.imag[mask]
        try:
            result = full_fit(f_res, real_res, imag_res)
            s21_fit = np.abs(result.best_fit)

            f0 = result.best_values['f_0']
            Q = result.best_values['Q']
            Qc = result.best_values['Q_e_real']
            Qi = get_qi(result.best_values['Q'], result.best_values['Q_e_real'])
            br = get_br(result.best_values['Q'], result.best_values['f_0']) / 1.e6
            res_index = k
            depth = get_dip_depth(result.best_fit.real, result.best_fit.imag)
            dfres = dfres.append(
                {'resonator_index': int(res_index), 'f0': f0, 'Qi': Qi, 'Qc': Qc, 'Q': Q, 'br': br, 'depth': depth},
                ignore_index=True)
            k = k + 1
        except:
            pass
    return dfres


@timing
def read_smurf_tuning_data(filename):
    # Reads the smurf file and extract frequency,
    # complex s21 and resonator index.
    # You should use the function below instead.

    dres = {'frequency': [], 'response': [], 'index': []}
    dfres = pd.DataFrame(dres)
    data = np.load(filename, allow_pickle=True).item()
    for band in list(data.keys()):
        if 'resonances' in list(data[band].keys()):
            for idx in list(data[band]['resonances'].keys()):
                scan = data[band]['resonances'][idx]
                f = np.array(scan['freq_eta_scan'])
                s21 = np.array(scan['resp_eta_scan'])
                res_index = scan['channel'] + band * 512
                dfres = dfres.append({'frequency': f, 'response': s21, 'index': res_index}, ignore_index=True)
    return dfres


@timing
def read_smurf_tuning_data_array(filenames):
    # Reads one or an array of smurf files.

    if np.array([filenames]).size == 1:
        frame = read_smurf_tuning_data(filenames)
    elif np.array([filenames]).size > 1:
        frame = pd.DataFrame({'frequency': [], 'response': [], 'index': []})
        for onefile in list(filenames):
            f = read_smurf_tuning_data(onefile)
            frame = frame.append(f, ignore_index=True)
    return frame


@timing
def smurf_into_frame(filename):
    # Takes a smurf file. Fit each resonator into a model and outputs a
    # dataframe of the parameters. You should use the function below instead.
    dres = {'time': [], 'resonator_index': [], 'f0': [], 'Qi': [],
            'Qc': [], 'Q': [], 'br': [], 'depth': []}
    dfres = pd.DataFrame(dres)
    data = np.load(filename, allow_pickle=True).item()
    for band in list(data.keys()):
        if 'resonances' in list(data[band].keys()):
            for idx in list(data[band]['resonances'].keys()):
                scan = data[band]['resonances'][idx]
                f = scan['freq_eta_scan']
                s21 = scan['resp_eta_scan']
                result = full_fit(f, s21.real, s21.imag)

                f0 = result.best_values['f_0']
                Q = result.best_values['Q']
                Qc = result.best_values['Q_e_real']
                Qi = get_qi(result.best_values['Q'], result.best_values['Q_e_real'])
                br = get_br(result.best_values['Q'], result.best_values['f_0'])
                res_index = scan['channel'] + band * 512
                time = data[band]['find_freq']['timestamp'][0]
                depth = get_dip_depth(result.best_fit.real, result.best_fit.imag)
                dfres = dfres.append({'time': time, 'resonator_index': int(res_index), 'f0': f0, 'Qi': Qi,
                                      'Qc': Qc, 'Q': Q, 'br': br, 'depth': depth}, ignore_index=True)
    return dfres


@timing
def smurf_data_into_frame(filenames):
    # Takes an array of smurf files and output the
    # fitting parameter.
    if np.array([filenames]).size == 1:
        frame = smurf_into_frame(filenames)
    elif np.array([filenames]).size > 1:
        frame = pd.DataFrame({'time': [], 'resonator_index': [], 'f0': [], 'Qi': [],
                              'Qc': [], 'Q': [], 'br': [], 'depth': []})
        for onefile in list(filenames):
            f = smurf_into_frame(onefile)
            frame = frame.append(f)
    return frame


@timing
def associate_smurf_and_vna(df_smurf, df_vna, tolerance=0.5):
    # Given a dataframe of smurf fitting parameter and VNA fitting parameter,
    # the function associates their peaks and combines the dataframe.
    # If smurf and VNA peaks are closer than a bandwidth or the tolerance (in MHz) then they
    # are considered to be the same resonator.

    df_smurf = df_smurf[['Q', 'Qc', 'Qi', 'br', 'depth', 'f0', 'resonator_index', 'time']]
    df_vna = df_vna[['Q', 'Qc', 'Qi', 'br', 'depth', 'f0', 'resonator_index']]
    smurf = np.array(df_smurf)
    vna = np.array(df_vna)
    vna_f0 = np.array(df_vna['f0'])
    smurf_f0 = np.array(df_smurf['f0'])
    df_both = np.zeros(0)
    for i in np.arange(len(df_smurf)):
        j = np.argmin(abs(vna_f0 - smurf_f0[i]))
        if abs(smurf_f0[i] - vna_f0[j]) < max(smurf[i, 3], tolerance):
            tmp = np.hstack((smurf[i], vna[j]))
            if len(df_both) == 0:
                df_both = tmp
            else:
                df_both = np.vstack((df_both, tmp))

    df_smurf_vna = pd.DataFrame(
        {'s_f0': df_both[:, 5], 'v_f0': df_both[:, 13], 's_Q': df_both[:, 0], 'v_Q': df_both[:, 8],
         's_Qc': df_both[:, 1], 'v_Qc': df_both[:, 9], 's_Qi': df_both[:, 2], 'v_Qi': df_both[:, 10],
         's_br': df_both[:, 3], 'v_br': df_both[:, 11], 's_depth': df_both[:, 4], 'v_depth': df_both[:, 12],
         'chan': df_both[:, 6]})  # 'v_index':comp[:,14]})
    return df_smurf_vna


@timing
def get_peaks_from_vna(vna_files):
    f, r, i = read_vna_data_array(vna_files)
    s21_corrected = correct_trend(f, r, i, avg_over=1000)
    f0s, resonance_s21, low, high = get_peaks_v2(f, s21_corrected, f_delta=1e5, det_num=1000, baseline=0.1)
    return f0s


@timing
def assign_freq_index(band, freqlist, dict_thru, highband):
    if ((highband == "S") & (band > 3)) | ((highband == "N") & (band <= 3)):
        missing_chip = dict_thru["S"]
    else:
        missing_chip = dict_thru["N"]
    counts = len(freqlist)
    mux_band = np.zeros(counts)
    mux_index = np.zeros(counts)
    init_band = np.floor((band % 4) * 3.5)
    miss = (3.5 * 66) - counts
    offset = 0
    if band % 2 == 1:
        offset = 33
    for i in np.arange(4):
        if (init_band + i) in missing_chip:
            if missing_chip in [3, 10]:
                miss -= 33
                offset = 33
            else:
                miss -= 66
                offset = 66
            continue
        start = max(0, 66 * i - offset)
        end = min(counts, (i + 1) * 66 - offset)
        mux_band[start:end] += i + init_band
        mux_index[start:end] = np.arange(end - start)
    print("Band %i misses %i resonators (%.2f percent)\n"
          % (band, miss, 100 * miss / (miss + len(freqlist))))
    return mux_band, mux_index, miss