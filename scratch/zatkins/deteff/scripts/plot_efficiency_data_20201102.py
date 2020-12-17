#!/usr/bin/env python3
# author: zatkins

import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append('/home/zatkins/repos/stslab/testbed_optics/deteff/modules')
import detector as det
import efficiency as eff
import metadata as meda

# get args
assem = 'Sv3'
optband = 150
dettype = 'B'
tbath = 70.0
exclude_temps = (0.,)
Terr = 0.3
setupname = '20200912'
arrayname = 'Sv3'

# get defaults
band_int_lims = {
    90: (70e9, 120e9),
    150: (120e9, 180e9)
}

band_suffix = {
    90: '1',
    150: '2'
}

bad_chans = [
    (3, 140),
]

# get channel maps
band2posn = meda.get_band2posn(assem, 'mux_band2mux_posn.csv')
smurf2pad = meda.get_smurf2pad(assem, 'smurf_band_chan2mux_band_pad.csv', band2posn)

wafer_file = 'UFM_Si.csv'
pad2pix = meda.get_pad2pix(wafer_file)
pad2det = meda.get_pad2det(wafer_file)

smurf2pix = meda.get_pad_trace(smurf2pad, pad2pix)
smurf2det = meda.get_pad_trace(smurf2pad, pad2det)

# get smurf metadata
md = np.load('../scripts/metadata.npy',allow_pickle=True).item()
mgc = np.load('../scripts/min_goodchans.npy',allow_pickle=True).item()
fbase = '/home/zatkins/so/data/smurf_data/{}/outputs/'
data_fn = '{}_iv.npy'
Ts = {}
Terrs = {}
iv = {}
gc = {}

for b in (1, 3):
    Ts[b] = []
    iv[b] = {}
    for T in md[b][tbath]:
        if T in exclude_temps: 
            continue
        Ts[b].append(T)
        dirs, ctime = md[b][tbath][T]
        iv[b][T] = np.load(fbase.format(dirs) + data_fn.format(ctime), allow_pickle=True).item()
    gc[b] = mgc[b][tbath]
    Terrs[b] = np.full(len(Ts[b]), Terr)

# get pixels based on detector type inputs
a_pix = []
a_smurf = []
for smurf, detinfo in smurf2det.items():
    o, d = detinfo
    if o != optband or d != dettype:
        continue

    smurfband, smurfchan = smurf
    if smurfband not in gc:
        continue
    if smurfchan not in gc[smurfband]:
        continue
    if smurf in bad_chans:
        continue

    a_pix.append(smurf2pix[smurf])
    a_smurf.append(smurf)

# get array
a = det.Array(setupname, arrayname, pixel_ids=a_pix)

# extract data and put into dict for Array.get_efficiency
bmin, bmax = band_int_lims[optband]
suffix = band_suffix[optband]

def extract(a_pix, get_power_kwargs):
    d = {}
    
    for i in range(len(a_pix)):
        pix = a_pix[i]
        b, c = a_smurf[i]

        d[pix] = {}
        d[pix]['Ts'] = Ts[b]
        d[pix]['p_sats'] = []
        for T in Ts[b]:
            d[pix]['p_sats'].append(1e-12*meda.get_psat(iv[b][T], b, c))
        d[pix]['p_sats'] = np.array(d[pix]['p_sats'])
        d[pix]['T_errs'] = Terrs[b]
        d[pix]['get_power_kwargs'] = get_power_kwargs
            
    return d

# get data and SPB
d = extract(a_pix, dict(band = suffix, min = bmin, max = bmax))

# get efficiency with passband and scatter
print('get efficiency')
a.get_efficiency(d)
a.plot_efficiency('wafer_scatter')
a.plot_efficiency('histogram')
a.plot_efficiency('radial_scatter', fmt = 'o', capsize = 5)
