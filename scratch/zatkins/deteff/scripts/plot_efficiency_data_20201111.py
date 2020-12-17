#!/usr/bin/env python3
# author: zatkins

import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

import sys
sys.path.append('/home/zatkins/repos/stslab/testbed_optics/deteff/modules')
import detector as det
import efficiency as eff
import metadata as meda

import os
import csv
import re

# get args
setupname = 'M_D_011'
assem = 'Sv5'
wafer_file = 'UFM_Si.csv'

side = 'N'
tune_grp = 'tune3'
run_grp = 'run2'
dark_grp = 'dark1'
dark_subtract = True

rhombuses = ['A']
optband = 150
pols = ['A', 'B']
bias_lines = None

T_err = 0.3
p_sat_err = 0.25e-12
exclude_temps = []
beam_N = int(1e4)
det_N = int(1e3)

# get defaults
exclude_temps_geq_dict = {
    90: 18,
    150: 100,
    0: 100,
    None: 100
}
exclude_temps_geq = exclude_temps_geq_dict[optband]

side_mux_posn_dict = {
    'N': 'Nmux_band2mux_posn.csv',
    'S': 'Smux_band2mux_posn.csv'
}
side_mux_posn = side_mux_posn_dict[side]

get_power_kwargs = {
    90: {'band': '1', 'min': 70e9, 'max': 120e9},
    150: {'band': '2', 'min': 120e9, 'max': 180e9},
    0: {},
    None: {}
}

# get darks
dark_dict = np.load(f'../data/assemblies/{assem}/{dark_grp}/dark_curves.npy', allow_pickle=True).item()
dark_T, dark_P, dark_P_err = dark_dict[optband]
dark_f = interp1d(dark_T, dark_P, kind='cubic', bounds_error=True)
dark_f_err = interp1d(dark_T, dark_P_err, kind='cubic', bounds_error=True)

if dark_subtract:
    dark_flag = 1
    method = '2d'
else:
    dark_flag = 0
    method = '1d'

# get good channels and bad channels
gc = np.load(f'../data/assemblies/{assem}/{run_grp}/goodchans.npy', allow_pickle=True)
try:
    bc = np.load(f'../data/assemblies/{assem}/{run_grp}/badchans.npy', allow_pickle=True)
except FileNotFoundError:
    bc = None
    print(f'No badchans found in {assem}/{run_grp}')

# get pixels, detectors based on detector type inputs
# and selected goodchannels (and any bad channels) from our run group

# get channel maps
band2posn = meda.get_mux_band2mux_posn(assem, side_mux_posn)
smurf2pad = meda.get_smurf2pad(assem, tune_grp, band2posn)
pad2pixdet = meda.get_pad2pixdet(wafer_file)
pad2bl = meda.get_pad2bl(wafer_file)

smurf2pixdet = meda.get_trace(smurf2pad, pad2pixdet)
smurf2bl = meda.get_trace(smurf2pad, pad2bl)

# get channels
pix_ids, pixdet_ids, smurf_ids = meda.get_sliced_channels(smurf2pixdet, smurf2bl, \
    rhombuses, optband, pols, bias_lines, gc, bc)

# get smurf metadata
md = meda.get_cl_ramp_metadata(assem, run_grp)

# get data from smurf for all channels
Ts = {}
p_sats = {}
T_errs = {}
p_sats_errs = {}
p_darks = {}
p_dark_errs = {}

for smurf in smurf_ids:
    b, c = smurf
    Ts[b, c] = []
    p_sats[b, c] = []
    T_errs[b, c] = []
    p_sats_errs[b, c] = []
    p_darks[b, c] = []
    p_dark_errs[b, c] = []

    found = True
    current_T = -1
    for md_id, fn in md.items():

        T, bl, band = md_id
        if T in exclude_temps or T >= exclude_temps_geq:
            continue

        if T != current_T:
            if found is False: print('not found')
            found = False
            current_T = T

        iv_poss = np.load('/home/zatkins/so' + md[T, bl, band], allow_pickle=True).item()
        if b not in iv_poss: continue
        if c not in iv_poss[b]: continue
        if found: print(f'already found {smurf} at temp {T}')
        found = True

        Ts[b, c].append(T)
        p_sats[b, c].append(meda.get_psat(iv_poss, b, c))
        T_errs[b, c].append(T_err)
        p_sats_errs[b, c].append(p_sat_err)
        p_darks[b, c].append(dark_flag*dark_f(T))
        p_dark_errs[b, c].append(dark_flag*dark_f_err(T))

# match pixels, detectors with data from smurf
d = {}
for i in range(len(pixdet_ids)):
    if dark_subtract:
        pixdet = pixdet_ids[i]
    else:
        pixdet = smurf_ids[i]

    b, c = smurf_ids[i]

    d[pixdet] = {}
    d[pixdet]['Ts'] = np.array(Ts[b, c])
    d[pixdet]['p_sats'] = np.array(p_sats[b, c])
    d[pixdet]['T_errs'] = np.array(T_errs[b, c])
    d[pixdet]['p_sat_errs'] = np.array(p_sats_errs[b, c])
    d[pixdet]['p_darks'] = np.array(p_darks[b, c])
    d[pixdet]['p_dark_errs'] = np.array(p_dark_errs[b, c])
    d[pixdet]['get_power_kwargs'] = get_power_kwargs[optband]
    d[pixdet]['method'] = method

if dark_subtract:

    # get array
    a = det.Assembly(setupname, assem, pixel_ids=pix_ids, \
        beam_kwargs=dict(N = beam_N), pixel_kwargs=dict(N = det_N))

    # get efficiency with passband and scatter
    print('get efficiency')
    a.get_efficiency(d)

    print(f'{setupname}, {assem}, Rhombus {rhombuses}, Optical Band {optband}, Detector Polarizations {pols}, \
        Excluded Temps {exclude_temps} and Temps >= {exclude_temps_geq}')
    a.plot_efficiency('wafer_scatter')
    a.plot_efficiency('histogram')
    a.plot_efficiency('radial_scatter', fmt = 'o', capsize = 5)
