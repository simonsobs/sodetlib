#!/usr/bin/env python3
# author: zatkins

import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append('/home/zatkins/repos/stslab/testbed_optics/deteff/modules')
import detector as det
import efficiency as eff

# get args
import argparse

optband = 90
cutoff = 5848 # in kHz
tbath = 70.0
exclude_temps = (0.,)
Terr = 0.3
setupname = '20200912'
arrayname = 'array'
beam_name = 'MF_F'

# get defaults
band_int_lims = {
    90: (70e9, 120e9),
    150: (120e9, 180e9)
}

band_suffix = {
    90: '1',
    150: '2'
}

band_pix = {
    90: '../data/misc/pix90.txt',
    150: '../data/misc/pix150.txt'
}

bad_chans = {
    1: [],
    3: [140]
}

chan_maps = {
    1: '../data/misc/1599786596_channel_assignment_b1.txt',
    3: '../data/misc/1599787547_channel_assignment_b3.txt'
}

# get metadata
md = np.load('../scripts/metadata.npy',allow_pickle=True).item()
mgc = np.load('../scripts/min_goodchans.npy',allow_pickle=True).item()
fbase = '/home/zatkins/so/data/smurf_data/{}/outputs/'
data_fn = '{}_iv.npy'
Ts = {}
Terrs = {}
iv = {}
gc = {}

def get_chan2freq(b):
    f = np.loadtxt(chan_maps[b], delimiter = ',')
    d = {}
    for i in range(len(f)):
        d[int(f[i, 2])] = f[i, 0]
    return d

chan2freq = {}

for b in (1, 3):
    Ts[b] = []
    iv[b] = {}
    for T in md[b][tbath]:
        if T in exclude_temps: continue
        Ts[b].append(T)
        dirs, ctime = md[b][tbath][T]
        iv[b][T] = np.load(fbase.format(dirs) + data_fn.format(ctime), allow_pickle=True).item()
    gc[b] = mgc[b][tbath]
    Terrs[b] = np.full(len(Ts[b]), Terr)
    chan2freq[b] = get_chan2freq(b)

bmin, bmax = band_int_lims[optband]
suffix = band_suffix[optband]
with open(band_pix[optband]) as fn:
    pix = np.array(fn.readline().split(',')).astype(int)

# get arrays
a = det.Array(setupname, arrayname, pixel_ids=pix)

# get avg posns
x = np.array([a.pixels[i].x for i in a.pixels]).mean()
y = np.array([a.pixels[i].y for i in a.pixels]).mean()
z = np.array([a.pixels[i].z for i in a.pixels]).mean() # silly

# extract data and put into dict for SPB.get_efficiency
def extract(scatter):
    d={}
    p_opts = {}
    p_opts_std = {}
    i = 1 # dummy pix index
    
    for b in (1, 3):
        p_opts[b] = np.array([a.pixels[i].get_power(Ts[b], min = bmin, max = bmax, band = suffix)[0] for i in a.pixels])
        p_opts_std[b] = np.std(p_opts[b], axis = 0, ddof = 1)
        
        for c in gc[b]:
            if c in bad_chans[b]: continue
            if optband == 90 and chan2freq[b][c] > cutoff: continue
            if optband == 150 and chan2freq[b][c] < cutoff: continue
        
            d[i] = {}
            d[i]['Ts'] = Ts[b]
            d[i]['p_sats'] = []
            for T in Ts[b]:
                d[i]['p_sats'].append(1e-12*eff.get_psat(iv[b][T], b, c))
            d[i]['p_sats'] = np.array(d[i]['p_sats'])
            d[i]['T_errs'] = Terrs[b]
            if scatter: d[i]['other_errs'] = p_opts_std[b]
            d[i]['get_power_kwargs'] = dict(band = suffix, min = bmin, max = bmax)
            
            i += 1
            
    return d

# get data and SPB
dnoscat = extract(False)
dscat = extract(True)
p = list(dscat.keys())

s = det.SPB(setupname, (x,y,z), beam_name, pixel_ids=p)

# get efficiency with passband and scatter
print('get efficiency with passband and scatter')
s.get_efficiency(dscat)
s.plot_efficiency('histogram', pixel_ids = p)
s.plot_efficiency('unordered_scatter', pixel_ids = p, fmt = 'o', capsize = 5)

# get efficiency with passband and no scatter
print('get efficiency with passband and no scatter')
s.get_efficiency(dnoscat)
s.plot_efficiency('histogram', pixel_ids = p)
s.plot_efficiency('unordered_scatter', pixel_ids = p, fmt = 'o', capsize = 5)
