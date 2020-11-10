"""
Module for general smurf operations.
"""
from sodetlib.util import cprint, TermColors
import numpy as np
import os
import time
from scipy import signal
import scipy.optimize as opt
from scipy import interpolate
import pickle as pkl
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pysmurf.client.util.pub import set_action


@set_action()
def take_squid_open_loop(S,cfg,bands,wait_time,Npts,NPhi0s,Nsteps,relock,
			frac_pp=None,lms_freq=None,reset_rate_khz=None,
			lms_gain=None):
    """
    Adapted from script by SWH: shawn@slac.stanford.edu (can still see original
    in pysmurf/scratch/shawn/measureFluxRampFvsV.py) by MSF: msilvafe@ucsd.edu.

    Takes data in open loop (only slow integral tracking) and steps through flux
    values to trace out a SQUID curve. This can be compared against the tracked
    SQUID curve which might not perfectly replicate this if these curves are
    poorly approximated by a sine wave (or ~3 harmonics of a fourier expansion).

    Parameters
    ----------
    bands: (int list)
        list of bands to take dc SQUID curves on
    wait_time: (float)
        how long you wait between flux step point in seconds
    Npts: (int)
        number of points you take at each flux bias step to average
    Nphi0s: (int)
        number of phi0's or periods of your squid curve you want to take at
        least 3 is recommended and more than 5 just takes longer without much
        benefit.
    Nsteps: (int)
        Number of flux points you will take total.
    relock: (bool)
        Whether or not to relock before starting flux stepping

    Returns
    -------
    raw_data : (dict)
        This contains the flux bias array, channel array, and frequency
        shift at each bias value for each channel in each band.
    """
    ctime = S.get_timestamp()
    fn_raw_data = f'{S.output_dir}/{ctime}_fr_sweep_data.npy'
    print(bands)

    #This calculates the amount of flux ramp amplitude you need for 1 phi0
    #and then sets the range of flux bias to be enough to achieve the Nphi0s
    #specified in the fucnction call.
    band_cfg = cfg.dev.bands[bands[0]]
    if frac_pp is None:
        frac_pp = band_cfg['frac_pp']
    if lms_freq is None:
        lms_freq = band_cfg['lms_freq_hz']
    if reset_rate_khz is None:
        reset_rate_khz = band_cfg['flux_ramp_rate_khz']
    frac_pp_per_phi0 = frac_pp/(lms_freq/(reset_rate_khz*1e3))
    bias_low=-frac_pp_per_phi0*NPhi0s
    bias_high=frac_pp_per_phi0*NPhi0s

    #This is the step size calculated from range and number of steps
    bias_step=np.abs(bias_high-bias_low)/float(Nsteps)

    channels = {}

    bias = np.arange(bias_low, bias_high, bias_step)

    # final output data dictionary
    raw_data = {}
    raw_data['bias'] = bias
    bands_with_channels_on=[]
    for band in bands:
        band_cfg = cfg.dev.bands[band]
        print(band_cfg)
        if lms_gain is None:
            lms_gain = band_cfg['lms_gain']
        channels[band] = S.which_on(band)
        if len(channels[band])>0:
            S.log(f'{len(channels[band])} channels on in band {band}, configuring band for simple, integral tracking')
            S.log(f'-> Setting lmsEnable[1-3] and lmsGain to 0 for band {band}.', S.LOG_USER)
            prev_lms_enable1 = S.get_lms_enable1(band)
            prev_lms_enable2 = S.get_lms_enable2(band)
            prev_lms_enable3 = S.get_lms_enable3(band)
            prev_lms_gain = S.get_lms_gain(band)
            S.set_lms_enable1(band, 0)
            S.set_lms_enable2(band, 0)
            S.set_lms_enable3(band, 0)
            S.set_lms_gain(band, lms_gain)

            raw_data[band]={}

            bands_with_channels_on.append(band)

    bands=bands_with_channels_on
    fs = {}

    sys.stdout.write('\rSetting flux ramp bias to 0 V\033[K before tune'.format(bias_low))
    S.set_fixed_flux_ramp_bias(0.)

    ### begin retune on all bands with tones
    for band in bands:
        fs[band] = []
        S.log('Retuning')
        if relock:
            S.relock(band)
        for i in range(2):
            S.run_serial_gradient_descent(band)
            S.run_serial_eta_scan(band)

        # toggle feedback if functionality exists in this version of pysmurf
        time.sleep(5)
        S.toggle_feedback(band)

        ### end retune

    small_steps_to_starting_bias=None
    if bias_low<0:
        small_steps_to_starting_bias=np.arange(bias_low,0,bias_step)[::-1]
    else:
        small_steps_to_starting_bias=np.arange(0,bias_low,bias_step)

    # step from zero (where we tuned) down to starting bias
    S.log('Slowly shift flux ramp voltage to place where we begin.', S.LOG_USER)
    for b in small_steps_to_starting_bias:
        sys.stdout.write('\rFlux ramp bias at {b} V')
        sys.stdout.flush()
        S.set_fixed_flux_ramp_bias(b,do_config=False)
        time.sleep(wait_time)

    ## make sure we start at bias_low
    sys.stdout.write(f'\rSetting flux ramp bias low at {bias_low} V')
    S.set_fixed_flux_ramp_bias(bias_low,do_config=False)
    time.sleep(wait_time)

    S.log('Starting to take flux ramp.', S.LOG_USER)

    for b in bias:
        sys.stdout.write(f'\rFlux ramp bias at {b} V')
        sys.stdout.flush()
        S.set_fixed_flux_ramp_bias(b,do_config=False)
        time.sleep(wait_time)
        for band in bands:
            fsamp=np.zeros(shape=(Npts,len(channels[band])))
            for i in range(Npts):
                fsamp[i,:]=S.get_loop_filter_output_array(band)[channels[band]]
            fsampmean=np.mean(fsamp,axis=0)
            fs[band].append(fsampmean)

    sys.stdout.write('\n')

    S.log('Done taking flux ramp data.', S.LOG_USER)

    for band in bands:
        fres = []
        for ch in channels[band]:
            fres.append(S.channel_to_freq(band,ch))
        fres=[S.channel_to_freq(band, ch) for ch in channels[band]]
        raw_data[band]['fres']=np.array(fres)
        raw_data[band]['channels']=channels[band]

        #stack
        lfovsfr=np.dstack(fs[band])[0]
        raw_data[band]['lfovsfr']=lfovsfr
        raw_data[band]['fvsfr']=np.array([arr+fres for (arr,fres) in zip(lfovsfr,fres)])

    # save dataset for each iteration, just to make sure it gets
    # written to disk
    print(f'Writing SQUID Curve Data to: {fn_raw_data}')
    np.save(fn_raw_data, raw_data)
    S.pub.register_file(fn_raw_data,'dc_squid_curve',format='npy')

    # done - zero and unset
    S.set_fixed_flux_ramp_bias(0,do_config=False)
    S.unset_fixed_flux_ramp_bias()
    S.set_lms_enable1(band, prev_lms_enable1)
    S.set_lms_enable2(band, prev_lms_enable2)
    S.set_lms_enable3(band, prev_lms_enable3)
    S.set_lms_gain(band, lms_gain)
    return raw_data

@set_action()
def find_subbands(S, cfg, spur_width=5):
    """
    Do a noise sweep to find the coarse position of resonators.
    Return active bands and a dictionary of active subbands.

    Parameters
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
        A dictionary containing the list of subbands in each band.
    """
    subband_dict = {}
    bands = []

    amc = S.which_bays()
    if 0 in amc:
        bands += [0, 1, 2, 3]
    if 1 in amc:
        bands += [4, 5, 6, 7]
    if not bands:
        print('No active AMC')
        return bands, subband_dict

    for band in bands:
        freq, resp = S.full_band_resp(band)
        peaks = S.find_peak(freq, resp, make_plot=True,show_plot=False, band=band)
        fs_ = np.array(peaks*1.0E-6) + S.get_band_center_mhz(band)

        # Drops channels that are too close to 500 MHz multiple
        fs = [f for f in fs_
              if (np.abs((f + 500/2) % 500 - 500/2) > spur_width)]
        bad_fs = list(set(fs_) - set(fs))
        bad_fs = [f for f in fs_
                  if np.abs((f + 500/2) % 500 - 500/2) <= spur_width]

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


def find_and_tune_freq(S, cfg, bands, new_master_assignment=True, amp_cut=0.1):
    """
    Find_freqs to identify resonance, measure eta parameters + setup channels
    using setup_notches, run serial gradient + eta to refine

    Parameters
    ----------
    S: pysmurf.client.SmurfControl
        Pysmurf control instance
    cfg: DetConfig
        Detector config object
    bands : List[int]
        bands to find tuned frequencies on. In range [0,7].
    new_master_assignment : bool, optional
        Whether to create a new master assignment (tuning)
        file. This file defines the mapping between resonator frequency
        and channel number. Default True.
    amp_cut : float
        The fractiona distance from the median value to decide whether there
        is a resonance.
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
                    make_plot=True,
                    save_plot=True,
                    subband=subband, amp_cut=amp_cut)
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


