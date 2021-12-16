# -*- coding: utf-8 -*-
"""
Created on Oct 28th, 2020?

@author: Kaiwen Zheng

peak finder

given an S21 sweep across multiple resonators, this
finds the peaks, which offers a starting point
for finding f0s; you should then cut on Q etc. using the fitting
function after.
"""

import numpy as np

from timer_wrap import timing


@timing
def find_nearest(array, value):
    # stolen from stack exchange
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


@timing
def get_dip_depth(real, imag):
    # Input an array of real and imaginary s21, find the
    # depth between maximal and minimal point.
    s21_mag = np.abs(real + 1j * imag)
    depth = max(20 * np.log10(s21_mag)) - min(20 * np.log10(s21_mag))
    return depth


@timing
def get_peaks_v2(f, s21_corrected, f_delta=1e5, det_num=300, baseline=-0.5, min_depth=0.5):
    # Input the frequency and s21. S21 must be in db and the trend should be corrected already.
    # f_delta is the minimal width of each peak. det_num is the maximal number of detectors.
    # Make sure det_num is larger than actual detector numbers. The function rejects peaks with
    # a depth smaller than min_depth or peaks whose minimum is above baseline. These peaks are 
    # considered to be noise.
    # The funtion returns the resonator frequency, resonator peak s21, and the indice for the 
    # left and right bound of each peak. The data is ordered by frequency. 

    s21_loop = s21_corrected
    resonance_freq = []
    resonance_s21 = []
    low_indice = []
    high_indice = []

    mask = (f > min(f)) & (f < max(f))
    grad = np.gradient(f, s21_loop)

    i = 0
    while (i < det_num):
        i = i + 1
        if (len(np.unique(mask)) > 1):
            min_index = s21_loop[mask].argmin()
            s21_min = s21_loop[mask][min_index]
            f_min = f[mask][min_index]

            # print(resonance_freq)
            low_range = max(0, min_index - 100)
            high_range = max(min_index + 100, len(s21_loop))
            f_low_index = [ind for ind in np.arange(low_range, high_range) if
                           (f[ind] < (f_min - f_delta)) & (grad[ind] > 0)]
            f_high_index = [idx for idx in np.arange(low_range, high_range) if
                            (f[idx] > (f_min + f_delta)) & (grad[idx] < 0)]

            f_low_index.append(0)
            low = np.array(f_low_index).max()
            f_high_index.append(len(s21_loop) - 1)
            high = np.array(f_high_index).min()
            depth = max(s21_loop[low] - s21_min, s21_loop[high] - s21_min)
            depth2 = min(s21_loop[low] - s21_min, s21_loop[high] - s21_min)
            depth_bool = (depth > min_depth) & (depth2 > 0)

            if (depth_bool) & (s21_min <= baseline):
                resonance_freq.append(f_min)
                resonance_s21.append(s21_min)
                low_indice.append(low)
                high_indice.append(high)

            mask[low:high] = 0
            if (s21_min > baseline):
                i = det_num + 500
                print("Reaches Baseline.")

    resonance_freq = np.array(resonance_freq)
    resonance_s21 = np.array(resonance_s21)
    L = sorted(zip(resonance_freq, resonance_s21, low_indice, high_indice))
    resonance_freq, resonance_s21, low_indice, high_indice = zip(*L)

    return np.array(resonance_freq), np.array(resonance_s21), np.array(low_indice), np.array(high_indice)
