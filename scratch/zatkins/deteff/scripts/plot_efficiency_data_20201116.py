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
efficiency = True

setupname = 'M_D_011'
assem = 'Sv5'
tune_grp_dict = {'N':'tune3','S':'tune_south'}
run_grp_dict = {'N':'run2','S':'run3'}
band2posn_dict = {'N':'Nmux_band2mux_posn.csv','S':'Smux_band2mux_posn.csv'}
wafer_file = 'UFM_Si.csv'
exclude_temps_geq_dict = {90: 18}

dark_bias_lines = [0,3,6,7,8,9,10,11]
dark_sides = ['N','S']
dark_pols = ['A','B']

rhombuses = ['A','B','C']
bias_lines = [0,1,2,3,4,5,6,7,8,9,10,11]
sides = ['N','S']
freqs = [90]
pols = ['A','B','D']

get_power_kwargs = None
# get_power_kwargs = {
#     90: {'min': 77e9, 'max': 105e9},
#     150: {'min': 128e9, 'max': 168e9}
# }

T_err = 0.3
p_sat_err = 1e-14
delta_r = 10

beam_N = int(1e4)
beam_phi = '90'
det_N = int(2e3)

# load data
dm = meda.DataManager(assem, tune_grp_dict, run_grp_dict, band2posn_dict, wafer_file, \
    dark_sides=dark_sides, dark_bias_lines=dark_bias_lines, dark_pols=dark_pols, \
        sides=sides, rhombuses=rhombuses, bias_lines=bias_lines, freqs=freqs, pols=pols, \
        exclude_temps_geq_dict=exclude_temps_geq_dict, get_power_kwargs=get_power_kwargs)

d, pixel_ids = dm.get_pixel_eff_dict(T_err=T_err, p_sat_err=p_sat_err, delta_r=delta_r)

print('Darks:', dark_sides, dark_bias_lines, dark_pols)
print('Opts:', sides, freqs, pols)
print('Args:', T_err, p_sat_err, delta_r)

if efficiency:

    # get array
    a = det.Assembly(setupname, assem, pixel_ids=pixel_ids, \
        beam_kwargs=dict(N = beam_N, phi = beam_phi), pixel_kwargs=dict(N = det_N))

    # get efficiency with passband and scatter
    print('get efficiency')
    a.get_efficiency(d)

    a.plot_efficiency('wafer_scatter')
    a.plot_efficiency('histogram')
    a.plot_efficiency('radial_scatter')
