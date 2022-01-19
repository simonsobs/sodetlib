"""
This file is soon to be deprecated.

The goal is to read SMuRF level 2 data from Simons1. This is the result of not being able to finish
a project because I had to push and merge an example for someone. Then I felt I needed to document a
file I was hoping to trash. It is seems to be some sort of infinite recursion in which I can never
do any real work. So do not look too hard into the code or documentation here. January 13, 2022 -Caleb

The file originally (seems to) come from Zach Atkins and can be found in sodetlib at
scratch/zatkins/deteff/modules/metadata.py

Caleb Wheeler extracted this file an uncommitted build of sodetlib from Kaiwen Zheng at Princeton.
"""
import os
from typing import NamedTuple
from operator import attrgetter
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from sodetlib.detmap.detmap_config import abs_path_detmap


class PsatBath(NamedTuple):
    psat: float
    temp_k: float


def get_psat(data_dict, band, chan, unit=1e-12, level=0.9, greedy=False):
    """Returns the conventional P_sat from a SMuRF IV curve dictionary.

    Parameters
    ----------
    data_dict : dict
        The dictionary containing the IV curve data
    band : int
        The smurf band to extract from the dict
    chan : int
        The channel to extract from the band
    unit: float
        The conversion to SI units, by default 1e-12
    level : float
        The definition of P_sat, i.e. power when R = level*R_n
    greedy : bool, optional
        If True, will return -1000 if the R/Rn curve crosses the level more than once, by default False.
        If False, returns the power at the first instance when R/Rn crosses the level.

    Returns
    -------
    float
        The conventional P_sat
    """
    chan_data = data_dict[band][chan]

    p = chan_data['p_tes']
    rn = chan_data['R'] / chan_data['R_n']

    cross_idx = np.where(np.logical_and(rn - level >= 0, np.roll(rn - level, 1) < 0))[0]
    try:
        assert len(cross_idx) == 1
    except AssertionError:
        if greedy:
            return -1000
        else:
            cross_idx = cross_idx[:1]

    try:
        cross_idx = cross_idx[0]
    except IndexError:
        # print(f'band {band}, chan {chan} has IndexError, returning 0')
        # print(cross_idx)
        return 0

    try:
        rn2p = interp1d(rn[cross_idx - 1:cross_idx + 1], p[cross_idx - 1:cross_idx + 1])
    except ValueError:
        # print(f'band {band}, chan {chan} has ValueError, returning 0')
        return 0

    return unit * rn2p(level)


def read_psat(coldload_ivs, make_plot=False):
    psat_by_temp = {}
    psat_by_band_chan = {}
    for coldload_iv in coldload_ivs:
        ivfile = abs_path_detmap
        for dir_or_file in coldload_iv['data_path'].split('/'):
            ivfile = os.path.join(ivfile, dir_or_file)
        iv = np.load(ivfile, allow_pickle=True).item()
        if coldload_iv['band'] != 'all':
            print('Reading band ', int(coldload_iv['band']), ' in ', ivfile)
            band_list = [int(coldload_iv['band'])]
        else:
            band_list = [b for b in iv.keys() if type(b) == np.int64]
        temp_k = coldload_iv['bath_temp']
        if temp_k not in psat_by_temp.keys():
            psat_by_temp[temp_k] = {}
        for band in band_list:
            band = int(band)
            if band not in psat_by_temp[temp_k].keys():
                psat_by_temp[temp_k][band] = {}
            if band not in psat_by_band_chan.keys():
                psat_by_band_chan[band] = {}
            for chan in iv[band].keys():
                chan = int(chan)
                if chan not in psat_by_band_chan[band].keys():
                    psat_by_band_chan[band][chan] = set()
                ch_psat = np.float(get_psat(iv, band, chan, level=0.9, greedy=False))
                psat_by_temp[temp_k][band][chan] = ch_psat
                psat_bath = PsatBath(psat=ch_psat, temp_k=temp_k)
                psat_by_band_chan[band][chan].add(psat_bath)

    if make_plot:
        for band in sorted(psat_by_band_chan.keys()):
            for chan in sorted(psat_by_band_chan[band].keys()):
                temps_k = []
                psats = []
                for psat, temp_k in sorted(psat_by_band_chan[band][chan], key=attrgetter('temp_k')):
                    temps_k.append(temp_k)
                    psats.append(psat)
                plt.plot(temps_k, psats, marker='o', alpha=0.5)
        plt.xlabel('Temp(K)')
        plt.ylabel('Psat(W)')
        plt.show()

    return psat_by_band_chan, psat_by_temp