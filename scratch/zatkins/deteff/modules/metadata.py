#!/usr/bin/env python3

# author: zatkins
# desc: functions for generating the optical efficiency of given detectors from data

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

import pathlib
import os
import csv
import re

# repo data folder for loading
# TODO: this is basically a proxy for a full installation
fpath = pathlib.Path(__file__).parent.parent / 'data'
raw_data_path = '/home/zatkins/so'

def get_psat(data_dict, band, chan, unit = 1e-12, level = 0.9, greedy = False):
    '''Returns the conventional P_sat from a SMuRF IV curve dictionary.

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
    '''
    chan_data = data_dict[band][chan]

    p = chan_data['p_tes']
    rn = chan_data['R']/chan_data['R_n']

    cross_idx = np.where(np.logical_and(rn - level >= 0, np.roll(rn - level, 1) < 0))[0]
    try:
        assert len(cross_idx) == 1
    except AssertionError:
        if greedy: return -1000
        else: cross_idx = cross_idx[:1]

    try:
        cross_idx = cross_idx[0]
    except IndexError:
        print(f'band {band}, chan {chan} has IndexError, returning 0')
        return 0

    try:
        rn2p = interp1d(rn[cross_idx-1:cross_idx+1], p[cross_idx-1:cross_idx+1])
    except ValueError:
        print(f'band {band}, chan {chan} has ValueError, returning 0')
        return 0

    return unit*rn2p(level)

def get_cl_ramp_metadata(assem, run_grp):
    fns = os.listdir(fpath / 'assemblies' / str(assem) / str(run_grp))
    fn_re = 'cl_ramp_(.+)'
    found = False
    for fn in fns:
        searcher = re.search(fn_re, fn)
        if searcher is not None:
            assert not found
            found = True
            md_fn = 'cl_ramp_' + searcher.groups()[0]
        else:
            continue
    assert found

    md = {}
    md_re = '(.+)_iv_raw_data.npy'
    with open(f'../data/assemblies/{assem}/{run_grp}/{md_fn}', 'r', newline='') as md_f:
        reader = csv.DictReader(md_f)
        for row in reader:
            T = float(row['cold_load_temp'])
            bl = int(row['bias_line'])
            band = row['band']
            fn = row['data_path']

            searcher = re.search(md_re, fn)
            md_fbase = searcher.groups()[0]
            md[T, bl, band] = md_fbase + '_iv.npy'

    return md

def get_sliced_channels(smurf2pixdet, smurf2bl, rhombuses=None, optband=None, pols=None, bias_lines = None, \
    gc=None, bc=None):
    pix_ids = []
    pixdet_ids = []
    smurf_ids = []
    for smurf, pixdet in smurf2pixdet.items():
        if pixdet == 'unrouted': continue

        pix = pixdet[:3]
        det = pixdet[3:]
        o, p = det
        
        if rhombuses is not None:
            if pix[0] not in rhombuses: continue
        if optband is not None:
            if o != optband: continue
        if pols is not None:
            if p not in pols: continue
        if bias_lines is not None:
            if smurf2bl[smurf] not in bias_lines: continue
        if gc is not None:
            if list(smurf) not in gc.tolist(): continue
        if bc is not None:
            if list(smurf) in bc.tolist(): continue

        pix_ids.append(pix)
        pixdet_ids.append(pixdet)
        smurf_ids.append(smurf)

    return pix_ids, pixdet_ids, smurf_ids

def get_smurf_chan_assignment_fname_dict(assem, tune_grp):
    out = {}
    fns = os.listdir(fpath / 'assemblies' / str(assem) / str(tune_grp))
    fn_re = '(.+)_channel_assignment_b(.+).txt'
    for fn in fns: 
        searcher = re.search(fn_re, fn)
        if searcher is None:
            continue
        _, band_str = searcher.groups() 
        band = int(band_str) 
        out[band] = fn 
    return out

def get_smurf_band_chan2mux_band_pad_from_UMM(assem, tune_grp, UMM_fname, \
    out_fname = 'smurf_band_chan2mux_band_pad.csv', method = 'closest', save = True):
    
    method_dict = {
        'closest': get_closest_resonator
    }
    method = method_dict[method]

    smurf_chan_assignment_fname_dict = get_smurf_chan_assignment_fname_dict(assem, tune_grp)
    UMM_data = np.genfromtxt(fpath / 'assemblies' / str(assem) / str(UMM_fname), delimiter = ',', names = True)

    out = []
    seen = []
    for smurf_band in smurf_chan_assignment_fname_dict:
        tune_fname = smurf_chan_assignment_fname_dict[smurf_band]
        tune_data = np.loadtxt(fpath / 'assemblies' / str(assem) / str(tune_grp) / str(tune_fname), delimiter=',')
        found_freqs, _, smurf_chans, _ = tune_data.T
        found_freqs *= 1e6
        smurf_chans = smurf_chans.astype(int)

        for smurf_chan, found_freq in zip(smurf_chans, found_freqs):
            
            assert (smurf_band, smurf_chan) not in seen

            mux_band, mux_pad, actual_freq = method(found_freq, UMM_data)
            print(smurf_band, smurf_chan, (found_freq-actual_freq)/1e6)
            out.append({
                'smurf_band': smurf_band,
                'smurf_chan': smurf_chan,
                'mux_band': mux_band,
                'mux_pad': mux_pad
                })
            seen.append((smurf_band, smurf_chan))

    if save:
        with open(fpath / 'assemblies' / str(assem) / str(tune_grp) / str(out_fname), 'w', newline = '') as csvfile:
            fieldnames = ['smurf_band', 'smurf_chan', 'mux_band', 'mux_pad']
            writer = csv.DictWriter(csvfile, fieldnames)
            writer.writeheader()
            writer.writerows(out)

    return out

def get_closest_resonator(found_freq, UMM_data):
    bands = UMM_data['Band'].astype(int)
    pads = UMM_data['Pad'].astype(int)
    actual_freqs = UMM_data['UMM_Frequency'] + UMM_data['UFM_Frequency_Shift']

    min_idx = np.abs(actual_freqs - found_freq).argmin()
    return bands[min_idx], pads[min_idx], actual_freqs[min_idx]

def get_smurf2smurf(assem, tune_grp1, tune_grp2):
    out = {}

    smurf_chan_assignment_fname_dict1 = get_smurf_chan_assignment_fname_dict(assem, tune_grp1)
    smurf_chan_assignment_fname_dict2 = get_smurf_chan_assignment_fname_dict(assem, tune_grp2)
    
    for smurf_band1 in smurf_chan_assignment_fname_dict1:
        tune_fname1 = smurf_chan_assignment_fname_dict1[smurf_band1]
        tune_data1 = np.loadtxt(fpath / 'assemblies' / str(assem) / str(tune_grp1) / str(tune_fname1), delimiter=',')
        found_freqs1, _, smurf_chans1, _ = tune_data1.T # freqs in MHz
        smurf_chans1 = smurf_chans1.astype(int)

        for smurf_chan1, found_freq1 in zip(smurf_chans1, found_freqs1):
            
            assert (smurf_band1, smurf_chan1) not in out

            out_smurf_band2 = -1
            smurf_chan2 = -1
            smallest_delta = np.inf
            second_smallest_delta = np.inf
            for smurf_band2 in smurf_chan_assignment_fname_dict2:
                tune_fname2 = smurf_chan_assignment_fname_dict2[smurf_band2]
                tune_data2 = np.loadtxt(fpath / 'assemblies' / str(assem) / str(tune_grp2) / str(tune_fname2), delimiter=',')
                found_freqs2, _, smurf_chans2, _ = tune_data2.T # freqs in MHz
                smurf_chans2 = smurf_chans2.astype(int)

                delta = np.abs(found_freq1 - found_freqs2)
                min_idxs = np.argsort(delta)
                
                if delta[min_idxs[0]] < smallest_delta:
                    out_smurf_band2 = smurf_band2
                    smurf_chan2 = smurf_chans2[min_idxs[0]]
                    smallest_delta = delta[min_idxs[0]]

                if delta[min_idxs[1]] < second_smallest_delta:
                    second_smallest_delta = delta[min_idxs[1]]

            out[(smurf_band1, smurf_chan1)] = (out_smurf_band2, smurf_chan2, smallest_delta, second_smallest_delta)

    return out

def get_mux_band2mux_posn(assem, fname):
    out = {}
    band2posn_file = np.genfromtxt(fpath / 'assemblies' / str(assem) / str(fname), \
        delimiter=',', names=True, dtype=int, encoding='utf-8-sig')
    for row in band2posn_file:
        band = row['mux_band']
        posn = row['mux_posn']
        out[band] = posn

    return out

def get_smurf2pad(assem, tune_grp, mux_band2mux_posn, fname = 'smurf_band_chan2mux_band_pad.csv'):
    out = {}
    smurf2pad_file = np.genfromtxt(fpath / 'assemblies' / str(assem) / str(tune_grp) / str(fname), \
        delimiter=',', names=True, dtype=int, encoding='utf-8-sig')
    for row in smurf2pad_file:
        smurf_band = row['smurf_band']
        smurf_chan = row['smurf_chan']
        mux_band = row['mux_band']
        mux_posn = mux_band2mux_posn[mux_band]
        mux_pad = row['mux_pad']
        
        assert (smurf_band, smurf_chan) not in out
        
        out[(smurf_band, smurf_chan)] = (mux_posn, mux_pad)

    return out

def get_pix2loc(fname):
    out = {}
    wafer_file = np.genfromtxt(fpath / 'wafers' / fname, \
        delimiter=',', names=True, dtype=None, encoding='utf-8-sig')

    for row in wafer_file:
        rhomb = row['DTPixelsection']

        if row['DTSignaldescription'] != 'NC':
            r = row['DTPixelrow'] 
            c = row['DTPixelcolumn']
            x = row['DTPixelxcenter'] / 1e3
            y = row['DTPixelycenter'] / 1e3
        else:
            r = row['DCXpos'] / 1e3
            c = row['DCypos'] / 1e3
            x = np.nan
            y = np.nan
        
        if (rhomb, r, c) not in out:
            out[(rhomb, r, c)] = (x, y)
    
    return out

def get_pad2pix(fname):
    out = {}
    wafer_file = np.genfromtxt(fpath / 'wafers' / fname, \
        delimiter = ',', names=True, dtype = None, encoding = 'utf-8-sig')

    pad_re = 'SQ_(.+)_Ch_(.+)_\+'
    for row in wafer_file:
        pad_str = row['SQUID_PIN']
        searcher = re.search(pad_re, pad_str)
        if searcher is None:
            continue
        _, pad_str = searcher.groups()
        pad = int(pad_str)
        posn = row['Mux_chip_position']

        assert (posn, pad) not in out

        rhomb = row['DTPixelsection']
        if row['DTSignaldescription'] != 'NC':
            r = row['DTPixelrow'] 
            c = row['DTPixelcolumn']
        else:
            r = row['DCXpos'] / 1e3
            c = row['DCypos'] / 1e3

        out[(posn, pad)] = (rhomb, r, c)

    return out

def get_pad2det(fname):
    out = {}
    wafer_file = np.genfromtxt(fpath / 'wafers' / fname, \
        delimiter = ',', names=True, dtype = None, encoding = 'utf-8-sig')

    pad_re = 'SQ_(.+)_Ch_(.+)_\+'
    freq_re = '(.+)ghz'
    for row in wafer_file:
        pad_str = row['SQUID_PIN']
        searcher = re.search(pad_re, pad_str)
        if searcher is None:
            continue
        _, pad_str = searcher.groups()
        pad = int(pad_str)
        posn = row['Mux_chip_position']

        assert (posn, pad) not in out

        freq_str = row['DTSignaldescription']
        searcher = re.search(freq_re, freq_str)
        if searcher is None:
            freq = 0
        else:
            freq_str = searcher.groups()
            freq = int(*freq_str)

        pol_str = row['DTPadlabel']
        if pol_str[0] in ['T', 'R']: 
            type_ = 'A'
        elif pol_str[0] in ['B', 'L']:
            type_ = 'B'
        elif pol_str[0] in ['X']:
            type_ = 'D'
        else:
            assert False

        out[(posn, pad)] = (freq, type_)
    
    return out

def get_pad2bl(fname):
    out = {}
    wafer_file = np.genfromtxt(fpath / 'wafers' / fname, \
        delimiter = ',', names=True, dtype = None, encoding = 'utf-8-sig')

    pad_re = 'SQ_(.+)_Ch_(.+)_\+'
    for row in wafer_file:
        pad_str = row['SQUID_PIN']
        searcher = re.search(pad_re, pad_str)
        if searcher is None:
            continue
        _, pad_str = searcher.groups()
        pad = int(pad_str)
        posn = row['Mux_chip_position']

        assert (posn, pad) not in out

        bl = int(row['Bias_line'])

        out[(posn, pad)] = bl

    return out

def get_pad2pixdet(fname):
    out = {}

    pad2pix = get_pad2pix(fname)
    pad2det = get_pad2det(fname)

    for pad in pad2pix:
        assert pad in pad2det # one-to-one

    for pad in pad2det:
        assert pad in pad2pix # one-to-one
        assert pad not in out # no repeats

        out[pad] = pad2pix[pad] + pad2det[pad]

    return out

def get_trace(x2k, k2y):
    out = {}
    for x in x2k:
        k = x2k[x]
        if k not in k2y:
            out[x] = 'unrouted'
        else:
            out[x] = k2y[k]
    return out


class DataManager:

    def __init__(self, assem, tune_grp_dict, run_grp_dict, mux_band2mux_posn_file_dict, wafer_file, \
        goodchans = True, badchans = True, dark_sides = ['N', 'S'], dark_bias_lines = range(12), dark_pols = ['A','B','D'], \
            sides = ['N','S'], rhombuses = ['A','B','C'], bias_lines = range(12), freqs = [20,30,90,150,220,270], \
                pols = ['A','B','D'], opts = [0, 1], exclude_temps_geq_dict = None, get_power_kwargs = None):
        
        self.assem = str(assem)
        if dark_bias_lines is None:
            dark_bias_lines = range
        self.dark_sides = dark_sides
        self.dark_bias_lines = dark_bias_lines
        self.dark_pols = dark_pols

        self.sides = sides
        self.rhombuses = rhombuses
        self.bias_lines = bias_lines
        self.freqs = freqs
        self.pols = pols
        self.opts = opts

        if exclude_temps_geq_dict is None:
            exclude_temps_geq_dict = {}
        self.exclude_temps_geq_dict = exclude_temps_geq_dict

        if get_power_kwargs is None:
            get_power_kwargs = {
                90: {'band': '1', 'min': 70e9, 'max': 120e9},
                150: {'band': '2', 'min': 120e9, 'max': 180e9},
            }
        self.get_power_kwargs = get_power_kwargs

        self._mux_band2mux_posn_dict = {}
        for side, fname in mux_band2mux_posn_file_dict.items():
            self._get_mux_band2mux_posn(side, fname)
        
        self.smurf2pad = {}
        for side, tune_grp in tune_grp_dict.items():
            self._get_smurf2pad(side, tune_grp)
        
        self.pad2wafer_info = {}
        self._get_pad2wafer_info(wafer_file)
        self.smurf2wafer_info = self.get_trace(self.smurf2pad, self.pad2wafer_info)

        self.channels = {}
        self.unrouted = {}
        for smurf, wafer_info in self.smurf2wafer_info.items():
            if wafer_info != 'unrouted':
                self.channels[smurf] = Channel(smurf, wafer_info)
            else:
                self.unrouted[smurf] = 'unrouted'
                
        self.md = {}
        self.badchannels = {}
        for side, run_grp in run_grp_dict.items():
            self._get_cl_ramp_metadata(side, run_grp)
            self._match_channel2psat(side, run_grp, goodchans, badchans)

    @staticmethod 
    def get_pix2loc(fname):
        out = {}
        wafer_file = np.genfromtxt(fpath / 'wafers' / fname, \
            delimiter=',', names=True, dtype=None, encoding='utf-8-sig')

        pad_re = 'SQ_(.+)_Ch_(.+)_\+'
        for row in wafer_file:
            pad_str = row['SQUID_PIN']
            searcher = re.search(pad_re, pad_str)
            if searcher is None:
                continue
            if row['DTSignaldescription'] != 'NC':
                rhomb = row['DTPixelsection']
            else:
                rhomb = row['DTPixelsection'] + 'D'
 
            r = int(row['DTPixelrow'])
            c = int(row['DTPixelcolumn'])
            x = float(row['x']) / 1e3
            y = float(row['y']) / 1e3

            if (rhomb, r, c) not in out:
                out[rhomb, r, c] = (x, y)

        return out

    def _get_mux_band2mux_posn(self, side, fname):
        band2posn_file = np.genfromtxt(fpath / 'assemblies' / self.assem / str(fname), \
            delimiter=',', names=True, dtype=int, encoding='utf-8-sig')
        for row in band2posn_file:
            band = row['mux_band']
            posn = row['mux_posn']

            assert (side, band) not in self._mux_band2mux_posn_dict

            self._mux_band2mux_posn_dict[side, band] = posn

    def _get_smurf2pad(self, side, tune_grp, fname = 'smurf_band_chan2mux_band_pad.csv'):
        smurf2pad_file = np.genfromtxt(fpath / 'assemblies' / self.assem / str(tune_grp) / str(fname), \
            delimiter=',', names=True, dtype=int, encoding='utf-8-sig')
        for row in smurf2pad_file:
            smurf_band = row['smurf_band']
            smurf_chan = row['smurf_chan']
            mux_band = row['mux_band']
            posn = self._mux_band2mux_posn_dict[side, mux_band]
            pad = row['mux_pad']
            
            assert (side, smurf_band, smurf_chan) not in self.smurf2pad
            
            self.smurf2pad[side, smurf_band, smurf_chan] = (posn, pad)

    def _get_pad2wafer_info(self, fname):
        wafer_file = np.genfromtxt(fpath / 'wafers' / fname, \
            delimiter=',', names=True, dtype=None, encoding='utf-8-sig')

        pad_re = 'SQ_(.+)_Ch_(.+)_\+'
        for row in wafer_file:
            pad_str = row['SQUID_PIN']
            searcher = re.search(pad_re, pad_str)
            if searcher is None:
                continue
            pad, wafer_info = self._get_pad_wafer_info(row)
            assert pad not in self.pad2wafer_info
            self.pad2wafer_info[pad] = wafer_info
        
    def _get_pad_wafer_info(self, row):
        posn = row['Mux_chip_position']
        pad_re = 'SQ_(.+)_Ch_(.+)_\+'
        pad_str = row['SQUID_PIN']
        searcher = re.search(pad_re, pad_str)
        _, pad_str = searcher.groups() 
        pad = int(pad_str)
        
        if row['DTSignaldescription'] != 'NC':
            rhomb = row['DTPixelsection']
        else:
            rhomb = row['DTPixelsection'] + 'D'
 
        r = int(row['DTPixelrow'])
        c = int(row['DTPixelcolumn'])

        bias_line = int(row['Bias_line'])

        freq_re = '(.+)ghz'
        freq_str = row['DTSignaldescription']
        searcher = re.search(freq_re, freq_str)
        if searcher is None:
            freq = 150 # Unique to Si UFMs so far!
        else:
            freq_str = searcher.groups()
            freq = int(*freq_str)

        pol_str = row['DTPadlabel']
        if pol_str[0] in ['T', 'R']: 
            pol = 'A'
        elif pol_str[0] in ['B', 'L']:
            pol = 'B'
        elif pol_str[0] in ['X']:
            pol = 'D'
        else:
            assert False

        if (bias_line in self.dark_bias_lines) or (pol == 'D'):
            opt = 0
        else:
            opt = 1

        x = float(row['x']) / 1e3
        y = float(row['y']) / 1e3

        return (posn, pad), (rhomb, r, c, bias_line, freq, pol, opt, x, y)

    def get_trace(self, x2k, k2y):
        out = {}
        for x in x2k:
            k = x2k[x]
            if k not in k2y:
                out[x] = 'unrouted'
            else:
                out[x] = k2y[k]
        return out

    def _get_cl_ramp_metadata(self, side, run_grp):
        if side not in self.md:
            self.md[side] = {}

        fns = os.listdir(fpath / 'assemblies' / self.assem / str(run_grp))
        fn_re = 'cl_ramp_(.+)'
        found = False
        for fn in fns:
            searcher = re.search(fn_re, fn)
            if searcher is not None:
                assert not found, f'already found a ramp metadata file in {fpath / "assemblies" / self.assem / str(run_grp)}'
                found = True
                md_fn = 'cl_ramp_' + searcher.groups()[0]
            else:
                continue
        assert found

        md_re = '(.+)_iv_raw_data.npy'
        with open(f'../data/assemblies/{self.assem}/{run_grp}/{md_fn}', 'r', newline='') as md_f:
            reader = csv.DictReader(md_f)
            for row in reader:
                T = float(row['cold_load_temp'])
                bl = int(row['bias_line'])
                band = row['band']

                if T not in self.md[side]:
                    self.md[side][T] = {}
                if bl not in self.md[side][T]:
                    self.md[side][T][bl] = {}
                if band not in self.md[side][T][bl]:
                    fn = row['data_path']
                    searcher = re.search(md_re, fn)
                    md_fbase = searcher.groups()[0]
                    self.md[side][T][bl][band] = md_fbase + '_iv.npy'
                else:
                    assert False, f'Temp {T}, Bias Line {bl}, Band {band} already touched'

    def _match_channel2psat(self, side, run_grp, goodchans, badchans):
        if goodchans:
            gc = np.load(f'../data/assemblies/{self.assem}/{run_grp}/goodchans.npy', allow_pickle=True).tolist()
        else:
            gc = [[chan.band, chan.chan] for chan in self.channels.values()]
        
        if badchans:
            try:
                bc = np.load(f'../data/assemblies/{self.assem}/{run_grp}/badchans.npy', allow_pickle=True).tolist()
            except FileNotFoundError:
                bc = []
                print(f'No badchans found in {self.assem}/{run_grp}')
        else:
            bc = []

        for T in self.md[side]:
            for bl in self.md[side][T]:
                for band in self.md[side][T][bl]:
                                            
                    iv = np.load(raw_data_path + self.md[side][T][bl][band], allow_pickle=True).item()

                    for smurf, channel in self.channels.items():

                        s, b, c = smurf
                        if s != side:
                            continue

                        if T >= self.exclude_temps_geq_dict.get(channel.freq, np.inf):
                            continue
                        
                        if channel.opt == 1:
                            if channel.side not in self.sides:
                                self.badchannels[smurf] = channel
                                continue
                            if channel.rhombus not in self.rhombuses:
                                self.badchannels[smurf] = channel
                                continue
                            if channel.bias_line not in self.bias_lines:
                                self.badchannels[smurf] = channel
                                continue
                            if channel.freq not in self.freqs:
                                self.badchannels[smurf] = channel
                                continue
                            if channel.pol not in self.pols:
                                self.badchannels[smurf] = channel
                                continue
                            if channel.opt not in self.opts:
                                self.badchannels[smurf] = channel
                                continue

                        # filter darks
                        if channel.opt == 0:
                            if channel.side not in self.dark_sides:
                                self.badchannels[smurf] = channel
                                continue
                            if channel.bias_line not in self.dark_bias_lines and channel.pol != 'D':
                                self.badchannels[smurf] = channel
                                continue
                            if channel.pol not in self.dark_pols:
                                self.badchannels[smurf] = channel
                                continue
                            if channel.freq not in self.freqs:
                                self.badchannels[smurf] = channel
                                continue
 
                        if [b, c] not in gc:
                            self.badchannels[smurf] = channel
                            continue
                        if [b, c] in bc:
                            self.badchannels[smurf] = channel
                            continue

                        found = False
                        if b not in iv:
                            continue
                        if c not in iv[b]:
                            continue
                        if found:
                            print(f'found {smurf} already')
                            self.badchannels[smurf] = channel
                            continue
                        found = True

                        if channel.bias_line != bl:
                            # print(f'{smurf} with line {channel.bias_line} found in wrong iv: {T},{bl},{band}')
                            self.badchannels[smurf] = channel
                            continue

                        channel.Ts = np.append(channel.Ts, T)
                        channel.p_sats = np.append(channel.p_sats, get_psat(iv, b, c))

                    for smurf in self.badchannels:
                        if smurf in self.channels:
                            self.channels.pop(smurf)

    def get_pixel_eff_dict(self, T_err = 0, p_sat_err = 0, delta_r = 1000):
        out = {}
        pixel_ids = []

        for channel in self.channels.values():
            if channel.opt == 0:
               continue

            pixdet = channel.get_pixdet_id()
            if pixdet[:3] not in pixel_ids:
                pixel_ids.append(pixdet[:3])
            
            out[pixdet] = {}
            out[pixdet]['Ts'] = np.array(channel.Ts)
            out[pixdet]['p_sats'] = np.array(channel.p_sats)
            out[pixdet]['T_errs'] = np.full(len(channel.Ts), T_err)
            out[pixdet]['p_sat_errs'] = np.full(len(channel.Ts), p_sat_err)
            p_darks, p_dark_errs = self.get_p_darks(channel, delta_r)
            out[pixdet]['p_darks'] = p_darks
            out[pixdet]['p_dark_errs'] = p_dark_errs
            out[pixdet]['get_power_kwargs'] = self.get_power_kwargs[channel.freq]

        return out, pixel_ids

    def get_p_darks(self, channel, delta_r):
        out = []
        out_channels = []
        all = []
        all_channels = []

        for c in self.channels.values():
            if c.opt != 0:
                continue
            if c.freq != channel.freq:
                continue

            try:
                dark_f = interp1d(c.Ts, c.p_sats, bounds_error=True, kind='cubic')
            except ValueError:
                print(c.get_pixdet_id(), c.bias_line, c.Ts, c.p_sats)
            
            all.append(dark_f(channel.Ts))
            all_channels.append(c)

            if (c.r < channel.r - delta_r) or (c.r > channel.r + delta_r):
                continue

            dark_f = interp1d(c.Ts, c.p_sats, bounds_error=True, kind='cubic')
            out.append(dark_f(channel.Ts))
            out_channels.append(c)

        if len(out) == 0:
            out = all
            out_channels = all_channels

        if len(out) == 0:
            out = np.atleast_2d(np.zeros(len(channel.Ts)))

        out2 = np.zeros(np.array(out).shape)
        for i in range(len(out)):
            out2[i] = out[i] - np.mean(out[i] - np.mean(out, axis=0))

        p_darks = np.mean(out2, axis=0)-np.max(np.mean(out2, axis=0))
        p_dark_errs = np.std(out2, axis=0) 
        channel.dark_channels = out_channels
        return p_darks, p_dark_errs

    def plot_noodle(self):
        plt.figure(figsize=(6,4), dpi=300)
        color_dict = {
            (90, 1): 'orange', (90, 0): 'red',
            (150, 1): 'purple', (150, 0): 'blue'
        }
        seen_dict = {}
        opt_dict = {0: 'Dark', 1: 'Optical'}
        for channel in self.channels.values():
            if (channel.freq, channel.opt) not in seen_dict:
                label = f'{channel.side} {channel.freq}GHz, {opt_dict[channel.opt]}'
                plt.plot(channel.Ts, 1e12*channel.p_sats, label = label, color = color_dict[channel.freq, channel.opt])
                seen_dict[channel.freq, channel.opt] = True
            else:
                plt.plot(channel.Ts, 1e12*channel.p_sats, color = color_dict[channel.freq, channel.opt])
        plt.xlabel('Cold Load Temp [K]')
        plt.ylabel('$P_{sat}$ [pW]')
        plt.legend(title = f'N = {len(self.channels)}')
        plt.grid(which = 'both')
        plt.title(f'$P_{{sat}}$ Curves from Wafer Sides: {self.sides}')
        #plt.savefig('/home/zatkins/so/test.png',bbox_inches='tight')
        plt.show()


class Channel:

    def __init__(self, smurf, wafer_info):
        side, band, chan = smurf
        rhombus, row, col, bias_line, freq, pol, opt, x, y = wafer_info

        self.side = side
        self.band = band
        self.chan = chan
        self.rhombus = rhombus
        self.row = row
        self.col = col
        self.bias_line = bias_line
        self.freq = freq
        self.pol = pol
        self.opt = opt
        self.x = x
        self.y = y
        self.r = np.sqrt(x**2 + y**2)
        
        self.Ts = np.array([])
        self.p_sats = np.array([])

    def get_pixdet_id(self):
        return (self.rhombus, self.row, self.col, self.freq, self.pol, self.side, self.band, self.chan)
