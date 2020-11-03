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

databand = 3
optband = 3
tbath = 70.0
exclude_temps = (0.,)
T_err = 0.3
setupname = '20200912'
arrayname = 'array'
beam_name = 'MF_F'

# get defaults
bandlims = {
    1: {
        'b': (70e9, 120e9),
        't': (77e9, 105e9)
    },
    3: {
        'b': (120e9, 180e9),
        't': (128e9, 168e9)
    }
}

bandband = {
    1: '1',
    3: '2'
}

bandpix = {
    1: '../data/misc/band1pix90.txt',
    3: '../data/misc/band3pix150.txt'
}

badpix = {
    1: [],
    3: [140]
}

# get data
md = np.load('../scripts/metadata.npy',allow_pickle=True).item()
mgc = np.load('../scripts/min_goodchans.npy',allow_pickle=True).item()
fbase = '/home/zatkins/so/data/smurf_data/{}/outputs/'
data_fn = '{}_iv.npy'
Ts = []
iv = {}
for T in md[databand][tbath]:
    if T in exclude_temps: continue
    Ts.append(T)
    dirs, ctime = md[databand][tbath][T]
    iv[T] = np.load(fbase.format(dirs) + data_fn.format(ctime), allow_pickle=True).item()
gc = mgc[databand][tbath]

bmin, bmax = bandlims[optband]['b']
tmin, tmax = bandlims[optband]['t']
b = bandband[optband]
with open(bandpix[optband]) as fn:
    pix = np.array(fn.readline().split(',')).astype(int)
T_errs = np.full(len(Ts), T_err)

# get arrays
a = det.Array(setupname, arrayname, pixel_ids=pix)

# get avg posns
x = np.array([a.pixels[i].x for i in a.pixels]).mean()
y = np.array([a.pixels[i].y for i in a.pixels]).mean()
z = np.array([a.pixels[i].z for i in a.pixels]).mean() # silly

# get p_opt scatter from not knowing true locations
p_optsb_std = np.std([a.pixels[i].get_power(Ts, min = bmin, max = bmax, band = b)[0] \
    for i in a.pixels], axis = 0, ddof = 1)
p_optst_std = np.std([a.pixels[i].get_power(Ts, min = tmin, max = tmax)[0] for i in a.pixels], axis = 0, ddof = 1)

# function to extract data and put into dict for array.get_efficiency
def f(s, band, T_errs = None, oe = None, gpk = None):
    d={}
    for pix in s.pixels:
        d[pix] = {}
        d[pix]['p_sats'] = []
        for T in Ts:
            d[pix]['p_sats'].append(1e-12*eff.get_psat(iv[T], band, pix))
        d[pix]['p_sats'] = np.array(d[pix]['p_sats'])
        d[pix]['Ts'] = Ts
        if T_errs is not None:
            d[pix]['T_errs'] = T_errs
        if oe is not None:
            d[pix]['other_errs'] = oe
        if gpk is not None:
            d[pix]['get_power_kwargs'] = gpk
    return d

# get SPB and data
p = []
for c in gc:
    if c not in badpix[databand]:
        p.append(c)

s = det.SPB(setupname, (x,y,z), beam_name, pixel_ids=p)

dbnoscat = f(s, databand, T_errs = T_errs, gpk = dict(band = b, min = bmin, max = bmax))
dbscat = f(s, databand, T_errs = T_errs, oe = p_optsb_std, gpk = dict(band = b, min = bmin, max = bmax))
dtscat = f(s, databand, T_errs = T_errs, oe= p_optst_std, gpk = dict(min = tmin, max = tmax))

# get efficiency with passband and scatter
print('get efficiency with passband and scatter')
s.get_efficiency(dbscat)
s.plot_efficiency('histogram', pixel_ids = p)
s.plot_efficiency('unordered_scatter', pixel_ids = p, fmt = 'o', capsize = 5)

# get efficiency with passband and no scatter
print('get efficiency with passband and no scatter')
s.get_efficiency(dbnoscat)
s.plot_efficiency('histogram', pixel_ids = p)
s.plot_efficiency('unordered_scatter', pixel_ids = p, fmt = 'o', capsize = 5)

# get efficiency without passband but with scatter
print('get efficiency without passband but with scatter')
s.get_efficiency(dtscat)
s.plot_efficiency('histogram', pixel_ids = p)
s.plot_efficiency('unordered_scatter', pixel_ids = p, fmt = 'o', capsize = 5)
