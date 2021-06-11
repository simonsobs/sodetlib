import sys
import numpy as np
from tqdm import tqdm
import os

sys.path.append('/home/jlashner/repos/sodetlib/scratch/kaiwen/det_map/')
import detector_map as dm
import matplotlib.pyplot as plt

class ChannelMapping:
    def __init__(self, tunefile, uxm_dir, meta_dir):
        self.band2pos_file = os.path.join(uxm_dir, 'band2posn.csv')
        self.N_vna_map_file = os.path.join(uxm_dir, 'N_vna_map.csv')
        self.S_vna_map_file = os.path.join(uxm_dir, 'S_vna_map.csv')
        self.posnfile = os.path.join(uxm_dir, "band2posn.csv")

        self.design_file = os.path.join(meta_dir, "umux_32_map.pkl")
        self.waferfile = os.path.join(meta_dir, "UFM_Si.csv")

        self.N_vna2pad = dm.vna_freq_to_muxpad(self.N_vna_map_file,
                                               self.design_file) 
        self.S_vna2pad = dm.vna_freq_to_muxpad(self.S_vna_map_file,
                                               self.design_file) 

        self.smurf_tune = load_tune(tunefile)


def match_peaks(peaks, template, shift=0, tolerance=5e5):
    peaks = peaks.copy() - shift
    assignments = np.full_like(peaks, -1, dtype=int)
    template_mask = np.ones_like(template, dtype=bool)

    for i, r in enumerate(peaks):
        candidates = np.where(
            (np.abs(template - r) < tolerance) & template_mask
        )[0]
        if len(candidates) == 0:
            continue
        if len(candidates) == 1:
            template_idx = candidates[0]
        elif len(candidates) > 1:
            distances = np.abs(template[candidates] - r)
            template_idx = candidates[np.argmin(distances)]
        assignments[i] = template_idx
        template_mask[template_idx] = False

    return assignments


def find_shift_and_match(peaks, template, tolerance=.5, shift_min=-1,
                         shift_max=1, shift_step=.02, return_data=True):
    shifts = np.arange(shift_min, shift_max, shift_step)
    assigned_frac = np.zeros_like(shifts)
    for i, s in tqdm(enumerate(shifts), total=len(shifts)):
        assignment = match_peaks(peaks, template, shift=s, tolerance=tolerance)
        assigned_frac[i] = np.sum(assignment != -1) / len(assignment)

    shift_idx = np.argmax(assigned_frac)
    shift = shifts[shift_idx]
    assignment = match_peaks(peaks, template, shift=shift, tolerance=tolerance)
    print(f"Found shift of {shift:0.2f} MHz")
    print(f"Assigned frac: {np.max(assigned_frac)}")
    return shift, assignment, shifts, assigned_frac


def load_tune(tunefile, highband='S'):
    tune = np.load(tunefile, allow_pickle=True).item()
    peaks = {'N': {'chans': [], 'freqs': [], 'bands': []},
             'S': {'chans': [], 'freqs': [], 'bands': []}}
    if highband == 'S':
        lowband = 'N'
    else:
        lowband = 'S'

    for band, band_dict in tune.items():
        for _, chan_dict in band_dict['resonances'].items():
            if band >= 4:
                p = peaks[highband]
                offset = 2000
            else:
                p = peaks[lowband]
                offset = 0
            if chan_dict['channel'] != -1:
                p['chans'].append(chan_dict['channel'])
                p['freqs'].append(chan_dict['freq'] - offset)
                p['bands'].append(band)

    peaks['N']['chans'] = np.array(peaks['N']['chans'])
    peaks['S']['chans'] = np.array(peaks['S']['chans'])
    peaks['N']['freqs'] = np.array(peaks['N']['freqs'])
    peaks['S']['freqs'] = np.array(peaks['S']['freqs'])
    peaks['N']['bands'] = np.array(peaks['N']['bands'])
    peaks['S']['bands'] = np.array(peaks['S']['bands'])

    return peaks


def map_tune(tunefile, uxm_dir, highband='S'):
    # Extract tune
    tune = np.load(tunefile, allow_pickle=True).item()
    bands = sorted(tune.keys())
    abschans = []
    freqs = []
    for band in bands:
        for _, chan_dict in tune[band]['resonances'].items():
            if chan_dict['channel'] != -1:
                abschans.append(band * 512 + chan_dict['channel'])
                freqs.append(chan_dict['freq'])

    abschans = np.array(abschans)
    if highband == 'S':
        nchan_mask = abschans < (4*512)
        schan_mask = abschans >= (4*512)
    elif highband == 'N':
        schan_mask = abschans < (4*512)
        nchan_mask = abschans >= (4*512)
    tune_freqs = np.array(freqs) * 1e6

    # Load uxm files
    band2pos_file = os.path.join(uxm_dir, 'band2posn.csv')
    N_vna_map_file = os.path.join(uxm_dir, 'N_vna_map.csv')
    S_vna_map_file = os.path.join(uxm_dir, 'S_vna_map.csv')

    band2pos = pd.read_csv(band2pos_file)
    N_vna_map = pd.read_csv(N_vna_map_file)
    N_freqs = np.array(N_vna_map["UFM Frequency"])
    S_vna_map = pd.read_csv(S_vna_map_file)
    S_freqs = np.array(S_vna_map["UFM Frequency"])

def plot_match(freqs1, freqs2, assignment, band=0):
    if isinstance(band, int):
        band = [band]

    fig, axes = plt.subplots(
        len(band), 1, figsize=(30, 2*len(band)), squeeze=False
    )

    size = 4000

    f1_mask = (assignment >= 0)
    f2_mask = np.zeros_like(freqs2, dtype=bool)
    f2_mask[assignment[f1_mask]] = True

    cs1 = np.zeros_like(freqs1)
    cs2 = np.zeros_like(freqs2)
    for i in range(len(assignment)):
        c = i/5 % 1
        cs1[i] = c
        cs2[assignment[i]] = c

    for i, b in enumerate(band):
        ax = axes[i, 0]
        f0, f1 = 4000 + b*500, 4500 + (b*1) * 500
        bm1 = (f0 < freqs1) & (freqs1 < f1)
        bm2 = (f0 < freqs2) & (freqs2 < f1)

        m1 = f1_mask & bm1
        m2 = f2_mask & bm2

        ax.scatter(freqs1[m1], [1 for _ in freqs1[m1]], s=size, marker='|',
                   c=cs1[m1])
        ax.scatter(freqs2[m2], [0 for _ in freqs2[m2]], s=size, marker='|',
                   c=cs2[m2])

        ax.scatter(freqs1[~f1_mask & bm1], [1 for _ in freqs1[~f1_mask & bm1]],
                   s=size*1.6, marker='|', c='grey', alpha=0.5)
        ax.scatter(freqs2[~f2_mask & bm2], [0 for _ in freqs2[~f2_mask & bm2]],
                   s=size*1.6, marker='|', c='grey', alpha=0.5)

        ax.set(ylabel=f"Band {b}")
        ax.set_yticks([])
    return fig, ax
