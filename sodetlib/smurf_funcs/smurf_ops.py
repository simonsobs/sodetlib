"""
Module for general smurf operations.
"""
from sodetlib.util import cprint, TermColors
import numpy as np
import os
import time
from scipy import signal
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pysmurf.client.util.pub import set_action



@set_action()
def find_subbands(S, cfg, spur_width=5):
    """
    Do a noise sweep to find the coarse position of resonators.
    Return active bands and a dictionary of active subbands.
    ----------
    S : pysmurf.client.SmurfControl
        Smurf control object
    cfg : DetConfig
        sodetlib config object
    spur_width: float
        Will throw out any resonators which are within ``spur_width`` MHz
        from a multiple of 500 MHz to avoid picking up spurs.

    Returns
    -------
    bands : int array
        Active bands

    subband_dict : dict
        A dictionary of {band:[list of subbands]} for each resonator in MHz.
    """
    bands = np.array([])
    subband_dict = {}

    amc = S.which_bays()
    bands = []
    if 0 in amc:
        bands += [0, 1, 2, 3]
    if 1 in amc:
        bands += [4, 5, 6, 7]
    if not bands:
        print('No active AMC')
        return bands, subband_dict

    for band in bands:
        band_cfg = cfg.dev.bands[band]
        freq, resp = S.full_band_resp(band)
        peaks = S.find_peak(freq, resp, make_plot=True,show_plot=False, band=band)
        fs_ = np.array(peaks*1.0E-6) + S.get_band_center_mhz(band)

        # Drops channels that are too close to 500 MHz multiple
        fs = [f for f in fs_ if (np.abs((f + 500/2) % 500 - 500/2) > spur_width)]
        bad_fs = list(set(fs_) - set(fs))
        bad_fs = [f for f in fs_ if np.abs((f + 500/2) % 500 - 500/2) <= spur_width]

        if bad_fs:
            cprint(f"Dropping frequencies {bad_fs} because they are too close "
                   "to a 500 MHz interval", style=TermColors.WARNING)

        subbands=sorted(list({S.freq_to_subband(band,f)[0] for f in fs}))
        subband_dict[band] = subbands

        subband_strings = []
        for i,b in enumerate(subbands):
            subband_strings.append(f"{b} ({fs[i]:.2f}MHz)")

        cprint(f"Subbands detected for band {band}:\n{subband_strings}",
                style=TermColors.OKBLUE)
        cfg.dev.update_band(band, {'active_subbands': subbands})

    return bands, subband_dict


def find_and_tune_freq(S, cfg, bands, new_master_assignment=True):
    """
    Find_freqs to identify resonance, measure eta parameters + setup channels
    using setup_notches, run serial gradient + eta to refine

    Parameters
    ----------
    S:  (pysmurf.client.SmurfControl)
        Pysmurf control instance
    cfg: (DetConfig)
        Detector config object
    bands : [int]
        bands to find tuned frequencies on. In range [0,7].

    Optional parameters
    ----------
    new_master_assignment : bool
        Whether to create a new master assignment (tuning)
        file. This file defines the mapping between resonator frequency
        and channel number. Default True.

    Optional parameters from cfg file
    ----------
    drive : int
        The drive amplitude.  If none given, takes from cfg.
    make_plot : bool
        make the plot frequency sweep. Default False.
    save_plot : bool
        save the plot. Default True.
    """
    num_resonators_on = 0
    default_subbands = np.arange(13, 115)
    for band in bands:
        band_cfg = cfg.dev.bands[band]
        subband = band_cfg.get('active_subbands', default_subbands)
        if subband is True:
            subband = default_subbands
        elif not subband:
            continue
        S.find_freq(band, tone_power=band_cfg['drive'],
                    make_plot=band_cfg['make_plot'],
                    save_plot=band_cfg['save_plot'],
                    subband=subband)
        if len(S.freq_resp[band]['find_freq']['resonance']) == 0:
            cprint(f'Find freqs could not find resonators in '
            f'band : {band} and subbands : {subband}', False)
            continue
        S.setup_notches(band, tone_power=band_cfg['drive'],
                    new_master_assignment=new_master_assignment)
        S.run_serial_gradient_descent(band)
        S.run_serial_eta_scan(band)

        num_resonators_on += len(S.which_on(band))

    tune_file = S.tune_file
    if not tune_file:
        cprint("Find freqs was unsuccessful! could not find resonators in the\
                specified bands + subbands", False)
        return False
    print(f"Total num resonators on: {num_resonators_on}")
    print(f"Tune file: {tune_file}")

    print("Updating config tunefile...")
    cfg.dev.update_experiment({'tunefile': tune_file})

    return num_resonators_on, tune_file


