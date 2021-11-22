"""
vna_func.py

Original Source: Heather McCarrick
Additional Author(s): Kaiwen Zheng, Caleb Wheeler
"""

import numpy as np
import matplotlib as plt
from scipy import optimize, signal


def noise_model_pysmurf(f, wl, fk, alpha):
    # f, frequency
    # wl, white noise level
    # fk, fknee
    # alpha, index

    # make sure this matches your config file
    b, a = signal.butter(4, 63, analog=True, btype='low')

    w, h = signal.freqs(b, a, worN=f)
    tf = np.absolute(h) ** 2

    return (wl) * (1 + (fk / f) ** alpha) * tf


def find_nearest(a, b):
    idx = (np.abs(a - b)).argmin()
    return idx


def fit_noise_model(ts_pA, fs):
    bounds_low = [0., 0., 0.]
    bounds_high = [np.inf, np.inf, np.inf]
    bounds = (bounds_low, bounds_high)

    pxx, fr = plt.mlab.psd(ts_pA, NFFT=2 ** 12, Fs=fs)
    # fr,pxx=signal.welch(ts_pA, nperseg=2**13, fs=180)
    pxx = np.sqrt(pxx)

    fr_1 = find_nearest(fr, 0.5)
    fr_10 = find_nearest(fr, 20)
    wn_average = np.average(pxx[fr_1:fr_10])

    p0 = [wn_average, 0.02, 0.02]
    popt, pcov = optimize.curve_fit(noise_model_pysmurf, fr[1:], pxx[1:], p0=p0, bounds=bounds)
    wl, f_knee, n = popt
    return wn_average, fr, wl, f_knee, n
