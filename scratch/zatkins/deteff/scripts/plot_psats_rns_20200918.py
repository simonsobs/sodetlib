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
parser.add_argument('--cl', type = float)
parser.add_argument('--level', type = float, default = 0.9)
parser.add_argument('--show', type = bool, default = False)

args = parser.parse_args()

band = args.band
tbath = args.tbath
cl = args.cl 
level = args.level
show = args.show

# get metadata
d = np.load('metadata.npy', allow_pickle = True).item()
min_goodchans_md = np.load('min_goodchans.npy', allow_pickle = True).item()

# file names
fbase = '/home/zatkins/so/data/smurf_data/{}/outputs/'
goodchans_fn = '{}_goodchans.txt'
data_fn = '{}_iv.npy'

# get data for this run
metadata = d[band][tbath][cl]
dirs, ctime = metadata

# make figure
fig, ax = plt.subplots(nrows = 2, ncols = 2, figsize = (11,8.5))

out = {}
out_minset = {}

# load data
fbase = fbase.format(dirs)
d = np.load(fbase + data_fn.format(ctime), allow_pickle = True).item()

for i in range(2):
    for j in range(2):
        pl = ax[i, j]

        if i == 0 and j == 0:
            method = 'p_sat'

            # make title
            sub = method.split('_')[1]
            title = f'Band {band}, CL = {cl}K, Tbath = {tbath}mK $P_{{{sub}}}$'
            if method == 'p_sat': title += f', ${level}R_{{n}}$'

            # get goodchans
            with open(fbase + goodchans_fn.format(ctime)) as f:
                goodchans = np.array(f.readline().split(',')).astype(int)

            # get data to plot
            if method == 'p_trans':
                psats = np.array([d[band][chan]['p_trans'] for chan in goodchans])
            elif method == 'p_sat':
                psats = np.array([eff.get_psat(d,band,chan,level) for chan in goodchans])
            data = psats[psats >= 0]
            median = np.median(data)
            std = np.std(data)

            label = f'Median = {np.round(median, 2)} pW'
            xlabel = f'$P_{{{sub}}}$ [pW]'
            
            out[method] = (median, std)

        if i == 0 and j == 1:
            # check if min_goodchans ok with cl temp
            if cl not in min_goodchans_md['exclude_temps']:
                method = 'p_sat'

                # make title
                sub = method.split('_')[1]
                title = f'Minimal Channel Set, Band {band}, CL = {cl}K, Tbath = {tbath}mK $P_{{{sub}}}$'
                if method == 'p_sat': title += f', ${level}R_{{n}}$'

                # get goodchans
                min_goodchans = min_goodchans_md[band][tbath]

                # get data to plot
                if method == 'p_trans':
                    psats_min = np.array([d[band][chan]['p_trans'] for chan in min_goodchans])
                elif method == 'p_sat':
                    psats_min = np.array([eff.get_psat(d,band,chan,level) for chan in min_goodchans])
                data_min = psats_min[psats_min >= 0]
                median_min = np.median(data_min)
                std_min = np.std(data_min)

                label = f'Median = {np.round(median_min, 2)} pW'
                xlabel = f'$P_{{{sub}}}$ [pW]'
                
                out_minset[method] = (median_min, std_min)
            else:
                pl.axis('off')
                continue

        if i == 1 and j == 0:
            method = 'p_trans'

            # make title
            sub = method.split('_')[1]
            title = f'Band {band}, CL = {cl}K, Tbath = {tbath}mK $P_{{{sub}}}$'
            if method == 'p_sat': title += f', ${level}R_{{n}}$'

            # get goodchans
            with open(fbase + goodchans_fn.format(ctime)) as f:
                goodchans = np.array(f.readline().split(',')).astype(int)
            min_goodchans = min_goodchans_md[band][tbath]

            # get data to plot
            if method == 'p_trans':
                psats = np.array([d[band][chan]['p_trans'] for chan in goodchans])
            elif method == 'p_sat':
                psats = np.array([eff.get_psat(d,band,chan,level) for chan in goodchans])
            data = psats[psats >= 0]
            median = np.median(data)
            std = np.std(data)

            label = f'Median = {np.round(median, 2)} pW'
            xlabel = f'$P_{{{sub}}}$ [pW]'
            
            out[method] = (median, std)
            
            if cl not in min_goodchans_md['exclude_temps']:
                # get data to save
                if method == 'p_trans':
                    psats_min = np.array([d[band][chan]['p_trans'] for chan in min_goodchans])
                elif method == 'p_sat':
                    psats_min = np.array([eff.get_psat(d,band,chan,level) for chan in min_goodchans])
                data_min = psats_min[psats_min >= 0]
                median_min = np.median(data_min)
                std_min = np.std(data_min)
                
                out_minset[method] = (median_min, std_min)

        if i == 1 and j == 1:
            # make title
            title = f'Band {band}, CL = {cl}K, Tbath = {tbath}mK $R_{{n}}$'

            # get goodchans
            with open(fbase + goodchans_fn.format(ctime)) as f:
                goodchans = np.array(f.readline().split(',')).astype(int)
            min_goodchans = min_goodchans_md[band][tbath]

            # get data to plot
            data = 1e3*np.array([d[band][chan]['R_n'] for chan in goodchans])
            median = np.median(data)
            std = np.std(data)

            label = f'Median = {np.round(median, 2)} mOhm'
            xlabel = '$R_{n}$ [mOhm]'

            out['R_n'] = (median, std)
            
            if cl not in min_goodchans_md['exclude_temps']:
                # get data to save
                data_min = 1e3*np.array([d[band][chan]['R_n'] for chan in min_goodchans])
                median_min = np.median(data_min)
                std_min = np.std(data_min)
                
                out_minset['R_n'] = (median_min, std_min)

        if i == 0 and j == 1:
            data = data_min

        pl.hist(data, bins = 30)
        pl.set_xlabel(xlabel)
        pl.set_ylabel('Count')
        pl.axvline(median, color = 'r', label = label)
        pl.legend(title = f'N = {len(data)}')
        pl.set_title(title)

fig.tight_layout()

plt.savefig(fbase.replace('outputs', 'plots') + f'{ctime}_band{band}_cl{cl}_tbath{tbath}_p_sat_p_trans_rn_min_goodchans.png')
if show: plt.show()
np.save(fbase + f'{ctime}_band{band}_cl{cl}_tbath{tbath}_p_sat_p_trans_rn_medians.npy', np.array(out))
np.save(fbase + f'{ctime}_band{band}_cl{cl}_tbath{tbath}_p_sat_p_trans_rn_medians_min_goodchans.npy', np.array(out_minset))


