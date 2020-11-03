#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append('/home/zatkins/repos/stslab/testbed_optics/deteff/modules')
import efficiency as eff

# get args
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--band', type = int)
parser.add_argument('--tbath', type = float)
parser.add_argument('--method', type = str, default = 'p_sat')
parser.add_argument('--level', type = float, default = 0.9)
parser.add_argument('--show', type = bool, default = False)

args = parser.parse_args()

band = args.band
tbath = args.tbath
method = args.method
level = args.level
show = args.show

# get metadata
d = np.load('metadata.npy', allow_pickle = True).item()
min_goodchans = np.load('min_goodchans.npy', allow_pickle = True).item()

# file names
fbase = '/home/zatkins/so/data/smurf_data/{}/outputs/'
data_fn = '{}_iv.npy'

# get data for this run
metadata = d[band][tbath]

# make figure
fig, ax = plt.subplots(nrows = 2, ncols = 2, figsize = (11,8.5))

# load data
fbase = fbase.format(dirs)
d = np.load(fbase + data_fn.format(ctime), allow_pickle = True).item()

good_chans = min_goodchans[band][tbath]
N = len(good_chans)
N_per_plot = np.ceil()

for i in range(2):
    for j in range(2):
        pl = ax[i, j]

        # make title
        sub = method.split('_')[1]
        title = f'Band {band}, CL = {cl}K, Tbath = {tbath}mK $P_{{{sub}}}$'
        if method == 'p_sat': title += f', ${level}R_{{n}}$'

        # get data to plot
        if method == 'p_trans':
            psats = np.array([d[band][chan]['p_trans'] for chan in good_chans])
        elif method == 'p_sat':
            psats = np.array([eff.get_psat(d,band,chan,level) for chan in good_chans])
        data = psats[psats >= 0]

        label = f'Median = {np.round(median, 2)} pW'
        xlabel = f'$P_{{{sub}}}$ [pW]'
        
        pl.hist(data, bins = 30)
        pl.set_xlabel(xlabel)
        pl.set_ylabel('Count')
        pl.axvline(median, color = 'r', label = label)
        pl.legend(title = f'N = {len(psats)}')
        pl.set_title(title)

fig.tight_layout()
plt.savefig(fbase.replace('outputs', 'plots') + f'{ctime}_band{band}_cl{cl}_tbath{tbath}_p_sat_p_trans_rn.png')
if show: plt.show()

if not minset: mname = ''
else: mname = '_min_goodchans'

np.save(fbase + f'{ctime}_band{band}_cl{cl}_tbath{tbath}_p_sat_p_trans_rn_medians{mname}.npy', np.array(out))

