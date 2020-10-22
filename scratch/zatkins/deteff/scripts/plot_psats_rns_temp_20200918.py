#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

# get args
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--band', type = int)
parser.add_argument('--tbath', type = float)
parser.add_argument('--minset',type = bool, default = False)
parser.add_argument('--show', type = bool, default = False)

args = parser.parse_args()
band = args.band
tbath = args.tbath
minset = args.minset
show = args.show

# get metadata
d = np.load('metadata.npy', allow_pickle = True).item()
min_goodchans = np.load('min_goodchans.npy', allow_pickle = True).item()

# file names
fbase = '/home/zatkins/so/data/smurf_data/{}/outputs/'
fname = '{}_band{}_cl{}_tbath{}_p_sat_p_trans_rn_medians{}.npy'

# get data for this run
metadata = d[band][tbath]

Ts = list(metadata.keys())
if minset:
    Ts_new = []
    for T in Ts:
        if T not in min_goodchans['exclude_temps']:
            Ts_new.append(T)
    Ts = Ts_new

# initialize data structures
p_sat = []
p_sat_errs = []

p_trans = []
p_trans_errs = []

R_n = []
R_n_errs = []

if not minset:
    mname = ''
    title = f'Band {band}, Tbath = {tbath}mK'
else:
    mname = '_min_goodchans'
    title = f'Minimal Channel Set, Band {band}, Tbath = {tbath}mK'

for T in Ts:
    dirs, ctime = metadata[T]
    point = np.load(fbase.format(dirs) + fname.format(ctime, band, T, tbath, mname), allow_pickle = True).item()

    p_sat.append(point['p_sat'][0])
    p_sat_errs.append(point['p_sat'][1])

    p_trans.append(point['p_trans'][0])
    p_trans_errs.append(point['p_trans'][1])

    R_n.append(point['R_n'][0])
    R_n_errs.append(point['R_n'][1])

# plot
plt.errorbar(Ts, p_sat, yerr = p_sat_errs, fmt = 'o', label = '$P_{sat}$ [pW]')
plt.errorbar(Ts, p_trans, yerr = p_trans_errs, fmt = 'o', label = '$P_{trans}$ [pW]')
plt.errorbar(Ts, R_n, yerr = R_n_errs, fmt = 'o', label = '$R_{n}$ [mOhm]')
plt.grid(which = 'both')
plt.ylim(0,8)

plt.xlabel('$T_{cl}$ [K]')
plt.title(title)

plt.legend()
plt.tight_layout()
plt.savefig(f'/home/zatkins/so/cold_load/analysis/plots/band{band}_tbath{tbath}{mname}.png')
if show: plt.show()
