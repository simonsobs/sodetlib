"""
Module with functions that optimize device parameters with respect to
psd noise.
"""
import numpy as np
import os
import time
import sys
from scipy import signal
import matplotlib.pyplot as plt
import scipy.optimize as opt
from scipy import interpolate
import pickle as pkl
from sodetlib.util import cprint, TermColors, get_psd, make_filename, \
    get_tracking_kwargs, StreamSeg

from pysmurf.client.util.pub import set_action


def optimize_nphi0(S, cfg, band):
    """
    Optimizes the Nphi0 parameter for a given band. Loops through Nphi in
    the range [3, 6] and finds the one which results in the lowest white noise
    level.

    Args:
        S: pysmurf.client.SmurfControl
            Smurf control object
        cfg : DetConfig
            Detector Config object
        band : int
            band number

    Returns:
        lms_freq_opt : float
            Optimal LMS frequency (Hz)
        frac_pp_opt : float
            Optimal frac_pp
        nphi0_opt : int
            Optimal value of Nphi0
    """

    band_cfg = cfg.dev.bands[band]

    fr_rate = band_cfg['reset_rate_khz'] * 1e3
    tracking_kwargs = {'band': band,
                       'reset_rate_khz': band_cfg['reset_rate_khz'],
                       'lms_freq_hz': None, 'nsamp': 2**17,
                       'fraction_full_scale': band_cfg['frac_pp'],
                       'lms_gain': band_cfg['lms_gain'],
                       'make_plot': True, 'channel': [],
                       'show_plot': False, 'save_plot': True,
                       'meas_lms_freq': True, 'return_data': True}

    cur_nphi0 = fr_rate / band_cfg['lms_freq_hz']
    frac_pp_per_phi0 = band_cfg['frac_pp'] / cur_nphi0

    chans = S.which_on(band)

    nphi0s = np.arange(3, 7)
    wls = np.full_like(nphi0s, np.nan, dtype=float)
    fspans = np.full_like(nphi0s, np.nan, dtype=float)
    for i, nphi0 in enumerate(nphi0s):
        frac_pp = frac_pp_per_phi0 * nphi0
        if frac_pp >= 1:
            break
        tracking_kwargs['fraction_full_scale'] = frac_pp
        tracking_kwargs['lms_freq_hz'] = nphi0 * fr_rate
        f, df, _ = S.tracking_setup(**tracking_kwargs)
        datafile = S.take_stream_data(10)
        median_noise, _ = analyze_noise_psd(S, band, datafile)
        cprint(f"Nphi0: {nphi0}\tmedian_noise: {median_noise}")
        wls[i] = median_noise
        fspans[i] = np.median((np.max(f, 0) - np.min(f, 0))[chans])

    idx = np.argmin(wls)
    nphi0_opt = nphi0s[idx]
    fname = os.path.join(S.output_dir,
                         f'{S.get_timestamp()}_wls_vs_nphi0.txt')
    with open(fname, 'w') as f:
        f.write("# Nphi0\tfrac_pp\twl\n")
        for nphi0, wl in zip(nphi0s, wls):
            f.write(f"{nphi0}\t{nphi0*frac_pp_per_phi0:0.4f}\t{wl:0.4f}\n")
    S.pub.register_file(fname, 'tracking_optimization', format='txt')

    fig, ax = plt.subplots()
    ax2 = ax.twinx()

    ax.plot(nphi0s, wls)
    ax.set(xlabel='Nphi0', ylabel="Median White Noise Level")
    ax2.plot(nphi0s, fspans)
    ax2.set(ylabel="f span")
    figname = os.path.join(S.plot_dir,
                           f"{S.get_timestamp()}_wls_vs_nphi0.png")
    fig.savefig(figname)
    S.pub.register_file(figname, 'tracking_optimization', plot=True)

    lms_freq_opt = nphi0_opt * fr_rate
    frac_pp_opt = frac_pp_per_phi0 * nphi0_opt

    cprint("Optimal parameters:", TermColors.HEADER)
    cprint(f"Nphi0: {nphi0_opt}", True)
    cprint(f"lms_freq: {lms_freq_opt}", True)
    cprint(f"frac_pp: {frac_pp_opt}", True)

    S.load_tune()
    S.relock(band)
    for _ in range(2):
        S.run_serial_gradient_descent(band)
        S.run_serial_eta_scan(band)

    tracking_kwargs['lms_freq_hz'] = lms_freq_opt
    tracking_kwargs['fraction_full_scale'] = frac_pp_opt
    f, df, _ = S.tracking_setup(**tracking_kwargs)

    cfg.dev.update_band(band, {'frac_pp': frac_pp_opt,
                               'lms_freq_hz': lms_freq_opt,
                               'nphi0': nphi0_opt})
    cfg.dev.dump(os.path.abspath(os.path.expandvars(cfg.dev_file)),
                 clobber=True)

    return lms_freq_opt, frac_pp_opt, nphi0_opt


def optimize_tracking(S, cfg, band, init_fracpp=None, phi0_number=5,
                      reset_rate_khz=None, lms_gain=None, relock=True,
                      tunefile=None, make_plots=False):
    """
    This starts with some default parameters and optimizes the amplitude
    of the flux ramp and tracking frequency used in the tracking algorithm
    to minimize the frequency error.

    Returns: Optimized lms_freq, frac_pp given optimized lms_freq and
        provided reset_rate_khz, bad_track_chans: a list of channels
        that aren't tracking well, and a dictionary summarizing the
        tracking performance (peak to peak, error, bad channel flag).

    Args:
        band: (int)
            500MHz band to run on
        init_fracpp: (float)
            Fraction of the full range of the flux ramp DAC to start optimizing
            your tracking parameters. Default from dev_cfg.
        phiO_number
            Number of phi0's in the amplitude of your ramp. This is
            equivalently the target number of periods in your squid
            curve per flux ramp reset. The lms frequency is the product
            for phi0_number*reset_rate_khz. This defaults to 5 but if
            you do not have enough dynamic range it will decrease this
            to maximum allowing number given your FR DAC dynamic range.
        reset_rate_khz:
            Rate of the flux ramp in kHz. Default is taken from device
            config file.
        lms_gain:
            Tracking frequency lock loop feedback loop gain. Default is
            taken from device config file.
        relock:
            If True, will run reload the most recent tune file, relock, and
            run serial gradient descent and eta scan. Defaults to True
        tunefile: str
            Tunefile to use when relocking. Defaults to None
    """
    band_cfg = cfg.dev.bands[band]
    if lms_gain is None:
        lms_gain = band_cfg['lms_gain']
    if reset_rate_khz is None:
        reset_rate_khz = band_cfg['flux_ramp_rate_khz']
    if init_fracpp is None:
        init_fracpp = band_cfg['frac_pp']
    if tunefile is None:
        tunefile = cfg.dev.exp['tunefile']

    print(reset_rate_khz)
    cprint("Loading tune and running initial tracking setup")
    if relock:
        S.load_tune(tunefile)
        S.relock(band)
        for _ in range(2):
            S.run_serial_gradient_descent(band)
            S.run_serial_eta_scan(band)

    tracking_kwargs = {'band': band, 'reset_rate_khz': reset_rate_khz,
                       'lms_freq_hz': None, 'nsamp': 2**17,
                       'fraction_full_scale': init_fracpp,
                       'lms_gain': lms_gain, 'make_plot': True, 'channel': [],
                       'show_plot': False, 'save_plot': True,
                       'meas_lms_freq': True, 'return_data': True}

    if make_plots:
        tracking_kwargs['channel'] = S.which_on(band)

    f, _, _ = S.tracking_setup(**tracking_kwargs)
    f_span = np.max(f, 0) - np.min(f, 0)

    for c in S.which_on(band):
        if not 0.03 < f_span[c] < 0.25:
            S.channel_off(band, c)

    # Reruns to get re-estimate of tracking freq
    S.tracking_setup(**tracking_kwargs)

    lms_meas = S.lms_freq_hz[band]
    lms_freq = phi0_number * reset_rate_khz * 1e3
    frac_pp = init_fracpp * lms_freq / lms_meas

    tracking_kwargs['meas_lms_freq'] = False
    tracking_kwargs['fraction_full_scale'] = frac_pp
    tracking_kwargs['lms_freq_hz'] = lms_freq

    S.load_tune(tunefile)
    S.relock(band)
    for _ in range(2):
        S.run_serial_gradient_descent(band)
        S.run_serial_eta_scan(band)

    f, df, _ = S.tracking_setup(**tracking_kwargs)
    df_std = np.std(df, 0)
    f_span = np.max(f, 0) - np.min(f, 0)

    bad_track_chans = []
    outdict = {}
    for c in S.which_on(band):
        outdict[c] = {}
        outdict[c]['df_pp'] = f_span[c]
        outdict[c]['df_err'] = df_std[c]
        outdict[c]['tracking'] = True

    fname = os.path.join(S.output_dir,
                         f'{S.get_timestamp()}_optimize_tracking.pkl')
    pkl.dump(outdict, open(fname, 'wb'))
    S.pub.register_file(fname, 'tracking_optimization', format='pkl')

    print(f'Number of bad tracking channels {len(bad_track_chans)}')
    print(f'Optimized frac_pp: {frac_pp} for lms_freq = {lms_freq}')

    cfg.dev.update_band(band, {'frac_pp': frac_pp,
                               'lms_freq_hz': lms_freq,
                               'optimized_tracking': True})
    cfg.dev.dump(os.path.abspath(os.path.expandvars(cfg.dev_file)),
                 clobber=True)

    return lms_freq, frac_pp, outdict


def lowpass_fit(x, scale, cutoff, fs=4e3):
    """
    This function takes in a scaling and cutoff frequency and outputs
    a 1st order butterworth lowpass filter per the scipy.butter
    package. There's a spline interpolation applied to match the
    outputs of scipy.freqz package to the input x array.

    Args
    x (float array):
    Input frequency array

    scale (float):
    Scaling of lowpass filter function to match data

    cutoff (float):
    f3dB frequency in Hz of the lowpass filter

    fs (float):
    Data sampling rate in Hz.

    """
    b, a = signal.butter(1, 2*cutoff/fs)
    w, h = signal.freqz(b, a)
    x_fit = (fs*0.5/np.pi)*w
    y_fit = scale*abs(h)
    splrep = interpolate.splrep(x_fit, y_fit, s=0)
    return interpolate.splev(x, splrep, der=0)


def identify_best_chan(S, band, f, df,  f_span_min=.04, f_span_max=.12):
    """
    Identifies the channel with the minimum noise and frequency tracking
    error.

    Args
        f (array [nchannels x nsamples]):
            Tracked frequency for each channel.
        df (array [nchannesl x nsamples]):
            Tracking frequency error for each channel.
        f_span_min (float):
            Minimum flux ramp modulation depth [MHz]. Defaults to 40 kHz.
        f_span_max (float):
            Maximum flux ramp modulation depth [MHz]. Defaults to 120 kHz.

    Returns
    Channel identified as the lowest product of noise and tracking error
    """
    df_std = np.std(df, 0)
    f_span = np.max(f, 0) - np.min(f, 0)

    chans_to_consider = [c for c in S.which_on(band)
                         if f_span_min < f_span[c] < f_span_max]

    datfile = S.take_stream_data(10, make_freq_mask=False)
    _, outdict = analyze_noise_psd(S, band, datfile, chans=chans_to_consider)
    chan_noise = []
    chan_df = []
    for ch in list(chans_to_consider):
        chan_noise.append(outdict[ch]['white noise'])
        chan_df.append(df_std[ch])

    best_chan = chans_to_consider[
        np.argmin(np.asarray(chan_df)*np.asarray(chan_noise))
    ]
    cprint(f'best_chan: {best_chan}\tfspan: {f_span[best_chan]}')
    return best_chan




@set_action()
def optimize_lms_gain(S, cfg, band, BW_target, tunefile=None,
                      reset_rate_khz=None, frac_pp=None,
                      lms_freq=None, meas_time=10,
                      make_plot=False):
    """
    Finds the drive power and uc attenuator value that minimizes the median noise within a band.


    Parameters
    ----------
    band: (int)
        band to optimize noise on
    tunefile: (str)
        filepath to the tunefile for the band to be optimized
    reset_rate: (float)
        flux ramp reset rate in kHz used for tracking setup
    frac_pp: (float)
        fraction full scale of the FR DAC used for tracking_setup
    lms_freq: (float)
        tracking frequency used for tracking_setup
    meas_time: float
        Duration for data streaming to analyze PSD.
    make_plot: (bool)
        If true will make tracking plots for each channel.
        Defaults to False

    Returns
    -------
    opt_lms_gain: (int)
    Optimized value of the lms gain parameter

    outdict: (dictionary)
    dictionary containing a key for each lms_gain swept through.
    For each lms gain there is the frequency and PSD array
    as well as the single pole low pass filter fit parameters.

    """
    # Set the timestamp for plots and data outputs
    ctime = S.get_timestamp()

    # Initialize params from device config
    band_cfg = cfg.dev.bands[band]
    if tunefile is 'devcfg':
        tunefile = cfg.dev.exp['tunefile']
        S.load_tune(tunefile)
    if tunefile is None:
        S.load_tune()
    if frac_pp is None:
        frac_pp = band_cfg['frac_pp']
    if lms_freq is None:
        lms_freq = band_cfg['lms_freq_hz']
    if reset_rate_khz is None:
        reset_rate_khz = band_cfg['flux_ramp_rate_khz']

    # Retune and setup tracking on band to optimize
    S.relock(band)
    for i in range(2):
        S.run_serial_gradient_descent(band)
        S.run_serial_eta_scan(band)

    tracking_kwargs = {
        'reset_rate_khz': band_cfg['flux_ramp_rate_khz'],
        'lms_freq_hz': band_cfg['lms_freq_hz'],
        'fraction_full_scale': band_cfg['frac_pp'],
        'make_plot': make_plot, 'show_plot': False,
        'channel': [], 'nsamp': 2**18,
        'feedback_start_frac': 0.02,
        'feedback_end_frac': 0.94, 'lms_gain': 7,
        'return_data': True
    }
    if make_plot:
        tracking_kwargs['channel'] = S.which_on(band)

    f, df, _ = S.tracking_setup(band, **tracking_kwargs)
    best_chan = identify_best_chan(S, band, f, df, f_span_min=0.03,
                                   f_span_max=0.25)

    print(f'Channel chosen for lms_gain optimization: {best_chan}')

    # Store old downsample factor and filter parameters to reset at end
    prev_ds_factor = S.get_downsample_factor()
    prev_filt_param = S.get_filter_disable()
    # Turn off downsampling + filtering to measure tracking BW
    S.set_downsample_factor(1)
    S.set_filter_disable(1)

    # Setup parameters for PSDs
    nperseg = 2**12
    fs = S.get_flux_ramp_freq()*1e3
    detrend = 'constant'

    # Initialize output parameters
    outdict = {}
    f3dBs = []

    # Sweep over lms gain from 8 to 2 and calculate f3dB at each.
    lms_gain_sweep = np.arange(8, 1, -1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    fig.suptitle('$f_{3dB}$ vs lms_gain', fontsize=32)
    alpha = 0.8

    for lms_gain in lms_gain_sweep:
        outdict[lms_gain] = {}
        S.set_lms_gain(band, lms_gain)
        # for i in range(2):
        #    S.run_serial_gradient_descent(band)
        #    S.run_serial_eta_scan(band)
        tracking_kwargs['lms_gain'] = lms_gain
        S.tracking_setup(band, **tracking_kwargs)

        datafile = S.take_stream_data(meas_time, make_freq_mask=False)
        timestamp, phase, mask = S.read_stream_data(datafile)
        phase *= S.pA_per_phi0/(2*np.pi)
        ch_idx = mask[band, best_chan]
        f, Pxx = signal.welch(phase[ch_idx], detrend=detrend,
                              nperseg=nperseg, fs=fs)
        Pxx = np.sqrt(Pxx)

        outdict[lms_gain]['f'] = f
        outdict[lms_gain]['Pxx'] = Pxx
        def fit_func(x, scale, cutoff): return lowpass_fit(
            x, scale, cutoff, fs=fs)
        try:
            pars, covs = opt.curve_fit(
                fit_func, f, Pxx, bounds=([0, 0], [1e3, fs/2]))
            outdict[lms_gain]['fit_params'] = pars
            f3dBs.append(pars[1])

            ax1.loglog(f, Pxx, alpha=alpha, label=f'lms_gain: {lms_gain}')
            ax1.loglog(f, lowpass_fit(f, pars[0], pars[1]), '--',
                       label=f'fit f3dB = {np.round(pars[1],2)} Hz')
            alpha = alpha*0.9
        except RuntimeError as e:
            print(e)
            outdict[lms_gain]['fit_params'] = None
            f3dBs.append(fs/2)
        print(f'lms_gain = {lms_gain}: f_3dB fit = {pars[1]} Hz')

    # Identify the lms_gain that produces a f3dB closest to the target
    f3dBs = np.asarray(f3dBs)
    idx_min = np.argmin(np.abs(f3dBs - BW_target))
    if f3dBs[idx_min] < BW_target:
        opt_lms_gain = lms_gain_sweep[idx_min-1]
    else:
        opt_lms_gain = lms_gain_sweep[idx_min]
    print(f'Optimum lms_gain is: {opt_lms_gain}')

    S.set_lms_gain(band, opt_lms_gain)
    tracking_kwargs['lms_gain'] = opt_lms_gain
    S.tracking_setup(band, **tracking_kwargs)

    # Save plots and data and register them with the ocs publisher
    ax1.set_ylim([10, 100])
    ax1.legend(fontsize=14)
    ax1.set_xlabel('Frequency [Hz]', fontsize=18)
    ax1.set_ylabel('PSD', fontsize=18)

    ax2.plot(lms_gain_sweep, f3dBs, 'o--', label='Data')
    ax2.set_xlabel('lms_gain', fontsize=18)
    ax2.set_ylabel('$f_{3dB}$ [Hz]', fontsize=18)
    ax2.axvline(opt_lms_gain, ls='--', color='r',
                label='Optimized LMS Gain')
    ax2.axhline(BW_target, ls='--', color='k', label='Target $f_{3dB}$')
    ax2.legend(fontsize=14)
    plotpath = f'{S.plot_dir}/{ctime}_f3dB_vs_lms_gain_b{band}.png'
    plt.savefig(plotpath)
    S.pub.register_file(plotpath, 'opt_lms_gain', plot=True)

    datpath = f'{S.output_dir}/{ctime}_f3dB_vs_lms_gain_b{band}.pkl'
    pkl.dump(outdict, open(datpath, 'wb'))
    S.pub.register_file(datpath, 'opt_lms_gain_data', format='pkl')

    cfg.dev.update_band(band, {'lms_gain': opt_lms_gain,
                               'optimized_lms_gain': True})

    # Reset the downsampling filtering parameters
    S.set_downsample_factor(prev_ds_factor)
    S.set_filter_disable(prev_filt_param)
    return opt_lms_gain, outdict


@set_action()
def get_median_noise(S, cfg, band, meas_time=30, make_plots=False):
    """
    Takes PSD and returns the median noise of all active channels.

    Parameters
    ------------
    S:  pysmurf.client.SmurfControl
        Pysmurf control instance
    cfg: DetConfig
        Detector config object
    band: int
        band to get median noise for.

    Returns
    ---------
    median_noise: float
        Median noise for the specified band.
    """
    band_cfg = cfg.dev.bands[band]

    print("Serial Gradient Descent")
    S.run_serial_gradient_descent(band)
    print("Serail Eta Scan")
    S.run_serial_eta_scan(band)
    print("Tracking setup")

    tracking_kwargs = {
        'reset_rate_khz': 4, 'lms_freq_hz': band_cfg['lms_freq_hz'],
        'fraction_full_scale': band_cfg['frac_pp'],
        'make_plot': True, 'save_plot': True, 'show_plot': False,
        'channel': [], 'nsamp': 2**18,
        'feedback_start_frac': 0.02, 'feedback_end_frac': 0.94,
        'lms_gain': 7, 'return_data': True
    }
    if make_plots:
        tracking_kwargs['channel'] = S.which_on(band)


    f, _, _ = S.tracking_setup(band, **tracking_kwargs)
    f_span = np.max(f, 0) - np.min(f, 0)

    #for c in S.which_on(band):
    #    if not 0.03 < f_span[c] < 0.14:
    #        S.channel_off(band, c)

    datafile = S.take_stream_data(meas_time)
    median_noise, _ = analyze_noise_psd(S, band, datafile)
    return median_noise


@set_action()
def analyze_noise_psd(S, band, dat_file, chans=None, fit_curve=True,
                      max_phase_span=None):
    """
    Finds the white noise level, 1/f knee, and 1/f polynomial exponent of a
    noise timestream and returns the median white noise level of all channels
    and a dictionary of fitted values per channel.

    Parameters
    ----------
    band: int
        band to optimize noise on
    dat_file: str
        filepath to timestream data to analyze
    ctime: str
        ctime used for saved data/plot titles
    fit_curve: bool
        If true, will use the pysmurf ``analyze_psd`` function to calculate the
        white noise, n, and f_knee values.  If false, calculate the white noise
        value by taking the median of the PSD between 5 Hz and 100 Hz, and will
        set n=f_knee=None, which is much faster.
    max_phase_span : float
        If set, will cut channels based on the phase span when calculating the
        median.

    Returns
    -------
    median_noise: float
        median white noise level of all channels analyzed in pA/rtHz

    outdict: dict
        Dictionary containing noise information for each channel.
        Formatted like::

            outdict = {
                chan_number: {
                    'fknee': <fitted 1/f knee>,
                    'noise index': <fitted 1/f exponent>,
                    'whie noise': <white noise level in pA/rtHz>
                }
            }
    """
    if chans is None:
        chans = S.which_on(band)

    outdict = {}
    datafile = dat_file
    nperseg = 2**16
    detrend = 'constant'
    times, phase, mask = S.read_stream_data(datafile)
    f, Pxx = get_psd(S, times, phase, detrend=detrend, nperseg=nperseg)
    wl_mask = (f > 5) & (f < 50)
    wls = []
    cut_chans = []
    for chan in chans:
        if chan < 0:
            continue
        ch_idx = mask[band, chan]

        phase_span = np.max(phase[ch_idx]) - np.min(phase[ch_idx])
        if max_phase_span != None:
            if phase_span > max_phase_span:
                cut_chans.append(chan)
                continue

        if fit_curve:
            popt, pcov, f_fit, Pxx_fit = S.analyze_psd(f, Pxx[ch_idx])
            wl, n, f_knee = popt
            wlmean = np.nanmedian(Pxx[ch_idx][wl_mask])
        else:
            wl = np.nanmedian(Pxx[ch_idx][wl_mask])
            n = None
            f_knee = None

        if f_knee != 0:
            wls.append(wl)
        if f_knee == 0:
            wls.append(wlmean)

        outdict[chan] = {}
        outdict[chan]['fknee'] = f_knee
        outdict[chan]['noise index'] = n
        outdict[chan]['white noise'] = wl
        if fit_curve:
            outdict[chan]['wl_mean'] = wlmean

    cprint(f"Channels {cut_chans} were cut due to high phase span.", False)
    median_noise = np.median(np.asarray(wls))
    return median_noise, outdict


@set_action()
def optimize_bias(S, target_Id, vg_min, vg_max, amp_name, max_iter=30):
    """
    Scans through bias voltage for hemt or 50K amplifier to get the correct
    gate voltage for a target current.

    Parameters
    -----------
    S: pysmurf.client.SmurfControl
        PySmurf control object
    target_Id: float
        Target amplifier current
    vg_min: float
        Minimum allowable gate voltage
    vg_max: float
        Maximum allowable gate voltage
    amp_name: str
        Name of amplifier. Must be either "hemt" or "50K'.
    max_iter: int, optional
        Maximum number of iterations to find voltage. Defaults to 30.

    Returns
    --------
    success (bool):
        Returns a boolean signaling whether voltage scan has been successful.
        The set voltages can be read with S.get_amplifier_biases().
    """
    if amp_name not in ['hemt', '50K']:
        raise ValueError("amp_name must be either 'hemt' or '50K'")

    for _ in range(max_iter):
        amp_biases = S.get_amplifier_biases(write_log=True)
        Vg = amp_biases[f"{amp_name}_Vg"]
        Id = amp_biases[f"{amp_name}_Id"]
        delta = target_Id - Id
        # Id should be within 0.5 from target without going over.
        if 0 <= delta < 0.5:
            return True

        if amp_name == 'hemt':
            step = np.sign(delta) * (0.1 if np.abs(delta) > 1.5 else 0.01)
        else:
            step = np.sign(delta) * (0.01 if np.abs(delta) > 1.5 else 0.001)

        Vg_next = Vg + step
        if not (vg_min < Vg_next < vg_max):
            cprint(f"Vg adjustment would go out of range ({vg_min}, {vg_max}). "
                   f"Unable to change {amp_name}_Id to desired value", False)
            return False

        if amp_name == 'hemt':
            S.set_hemt_gate_voltage(Vg_next)
        else:
            S.set_50k_amp_gate_voltage(Vg_next)
        time.sleep(0.2)
    cprint(f"Max allowed Vg iterations ({max_iter}) has been reached. "
           f"Unable to get target Id for {amp_name}.", False)
    return False


@set_action()
def optimize_power_per_band(S, cfg, band, tunefile=None, dr_start=None,
                            frac_pp=None, lms_freq=None, make_plots=False,
                            lms_gain = None, meas_time=10, fixed_drive=False):
    """
    Finds the drive power and uc attenuator value that minimizes the median
    noise within a band.

    Parameters
    ----------
    band: int
        band to optimize noise on
    tunefile: str
        filepath to the tunefile for the band to be optimized
    dr_start: int
        drive power to start all channels with, default is 12
    frac_pp: float
        fraction full scale of the FR DAC used for tracking_setup
    lms_freq: float
        tracking frequency used for tracking_setup
    make_plots: bool
        If true, will make tracking plots
    meas_time : float
        Measurement time for noise PSD in seconds. Defaults to 10 sec.
    fixed_drive: bool
        If true, will not try to vary drive to search for global minimum.

    Returns
    -------
    min_med_noise : float
        The median noise at the optimized drive power
    atten : int
        Optimized uc attenuator value
    cur_dr : int
        Optimized dr value
    """
    band_cfg = cfg.dev.bands[band]
    if tunefile is None:
        tunefile = cfg.dev.exp['tunefile']
    if dr_start is None:
        dr_start = band_cfg['drive']
    if frac_pp is None:
        frac_pp = band_cfg['frac_pp']
    if lms_freq is None:
        lms_freq = band_cfg['lms_freq_hz']
    if lms_gain is None:
        lms_gain = band_cfg['lms_gain']

    S.load_tune(tunefile)

    drive = dr_start
    attens = np.arange(30, -2, -2)
    checked_drives = []
    found_min = False

    tracking_kwargs = {
        'reset_rate_khz': 4, 'lms_freq_hz': lms_freq,
        'fraction_full_scale': frac_pp,
        'make_plot': True, 'save_plot': True, 'show_plot': False,
        'channel': [], 'nsamp': 2**18,
        'feedback_start_frac': 0.02, 'feedback_end_frac': 0.94,
        'lms_gain': lms_gain, 'return_data': True
    }

    # Looping over drive powers
    while not found_min:
        cprint(f"Setting Drive to {drive}")
        ctime = S.get_timestamp()
        S.set_att_uc(band, 30)
        S.relock(band=band, tone_power=drive)

        print("Running setup notches")
        S.setup_notches(band, tone_power=drive, new_master_assignment=False)

        medians = []
        initial_median = None

        for atten in attens:
            cprint(f'Setting UC atten to: {atten}')
            S.set_att_uc(band, atten)

            band_cfg = cfg.dev.bands[band]

            print("Serial Gradient Descent")
            S.run_serial_gradient_descent(band)
            print("Serial Eta Scan")
            S.run_serial_eta_scan(band)
            print("Tracking setup")

            if make_plots:
                tracking_kwargs['channel'] = S.which_on(band)

            f, _, _ = S.tracking_setup(band, **tracking_kwargs)

            datafile = S.take_stream_data(meas_time)
            m, _ = analyze_noise_psd(S, band, datafile, fit_curve=False)

            medians.append(m)
            # Checks to make sure noise doesn't go too far over original median
            if initial_median is None:
                initial_median = m
            if m > 4 * initial_median:
                cprint(f"Median noise is now 4 times what it was at atten=30, "
                       f"so exiting loop at uc_atten = {atten}", style=False)
                break
        outfile = os.path.join(f'{S.output_dir}',
                               f'{S.get_timestamp()}_b{band}_noise_vs_input_power.txt')
        np.savetxt(outfile, np.array([attens[:len(medians)], medians]).T)

        # Summary plots
        plt.figure()
        plt.plot(attens[:len(medians)], medians)
        plt.title(f'Drive = {drive} in Band {band}', fontsize=18)
        plt.xlabel('UC Attenuator Value', fontsize=14)
        plt.ylabel('Median Channel Noise [pA/rtHz]', fontsize=14)
        plotname = os.path.join(S.plot_dir,
                                f'{ctime}_noise_vs_uc_atten_b{band}.png')
        plt.savefig(plotname)
        S.pub.register_file(plotname, 'noise_vs_atten', plot=True)
        plt.close()

        medians = np.asarray(medians)
        min_arg = np.argmin(medians)
        checked_drives.append(drive)
        if (0 < min_arg < len(medians)-1) or fixed_drive:
            found_min = True
            min_median = medians[min_arg]
            min_atten = attens[min_arg]
            min_drive = drive
            if not (0 < min_arg < len(medians) - 1):
                cprint("Minimum is on the boundary! May not be a global minimum!",
                       style=TermColors.WARNING)
        else:
            if min_arg == 0:  # Atten = 30
                drive -= 1
            else:             # atten = 0
                drive += 1

            if drive in checked_drives:
                cprint(f"Drive {drive} has already been checked!! "
                       f"Exiting loop unsuccessfully", False)
                found_min = False
                break

    if found_min:
        cprint(f'found optimum dr = {drive}, and optimum uc_att = {min_atten}',
               style=True)
        cprint(f'optimal noise: {min_median}', True)
        S.set_att_uc(band, min_atten)
        S.load_tune(tunefile)
        S.setup_notches(band, tone_power=drive, new_master_assignment=False)
        S.relock(band=band, tone_power=drive)
        S.run_serial_gradient_descent(band)
        S.run_serial_eta_scan(band)

        cfg.dev.update_experiment({'tunefile': S.tune_file})
        cfg.dev.update_band(band, {
            'uc_att': min_atten, 'drive': drive, 'optimized_drive': True
        })
        return min_median, min_atten, drive


@set_action()
def plot_optimize_attens(S, summary):
    """
    Plots the results from the optimize_attens functions.
    """
    wls = summary['wl_medians']
    grid = summary['atten_grid']
    shape = wls[0].shape

    xs = np.reshape(grid[:, 0], shape)
    ys = np.reshape(grid[:, 1], shape)

    ucs = summary['uc_attens']
    dcs = summary['dc_attens']

    for i, band in enumerate(summary['bands']):
        fig, ax = plt.subplots()
        fig.patch.set_facecolor('white')
        if len(ucs) == 1:  # Sweep is dc only
            uc = ucs[0]
            _wls = wls[i, 0, :]
            ax.plot(dcs, _wls)
            ax.set(xlabel="DC atten", ylabel="white noise (pA/rt(Hz))")
            ax.set(title=f"Band {band}, uc-atten {uc}")

        elif len(dcs) == 1:  # Sweep is uc only
            dc = dcs[0]
            _wls = wls[i, :, 0]
            ax.plot(ucs, _wls)
            ax.set(xlabel="UC atten", ylabel="white noise (pA/rt(Hz))")
            ax.set(title=f"Band {band}, dc-atten {dc}")

        else:
            wl = wls[i].copy()
            wl[wl > 100] = np.nan
            cbar = plt.contourf(xs, ys, wl,  levels=100)
            ax.set(xlabel="UC atten", ylabel="DC atten",
                   title=f"Band {band} Atten Sweep")
            fig.colorbar(cbar, label='Median White Noise [pA/rt(Hz)]')

        fname = make_filename(S, f'atten_sweep_b{band}.png', plot=True)
        fig.savefig(fname)


@set_action()
def optimize_attens(S, cfg, bands, meas_time=10, uc_attens=None,
                    dc_attens=None, tone_power=None, silence_logs=True,
                    tracking_kwargs=None, skip_setup_notches=False,
                    update_cfg=True, set_to_opt=True):
    """
    UC and DC attenuation optimization function, built to work efficiently
    with multiple bands.

    Args
    ----
    S : SmurfControl
        Pysmurf control object
    cfg : DetConfig
        Det Config instance
    meas_time : float
        Measurement time (sec) for white noise analysis
    tone_power : int
        Tone power to use for scan.
    silence_logs : bool
        If true will send pysmurf logs to file instead of stdout to declutter
        logs.
    tracking_kwargs : dict
        Custom tracking kwargs to pass to tracking setup
    skip_setup_notches : bool
        If true, will skip the initial setup notches at the start of the
        optimization. This is not recommended unless you are just testing the
        base functionality
    update_cfg : bool
        If true, will update the cfg object with the new atten values
    set_to_opt : bool
        If true, will set atten to the optimal values afterwards and run setup
        functions.
    """
    if isinstance(bands, (int, float)):
        bands = [bands]
    bands = np.array(bands)

    if uc_attens is None:
        uc_attens = np.arange(30, -2, -2)
    else:
        uc_attens = np.array(uc_attens)

    if dc_attens is None:
        dc_attens = np.arange(30, -2, -2)
    else:
        dc_attens = np.array(dc_attens)

    wl_medians = np.full((len(bands), len(uc_attens), len(dc_attens)), np.inf)
    datfiles = []
    atten_grid = []
    start_times = []
    stop_times = []

    start_time = time.time()

    logs_silenced = False
    logfile = None
    if silence_logs and (S.log.logfile == sys.stdout):
        logfile = make_filename(S, 'optimize_atten.log')
        print(f"Writing pysmurf logs to {logfile}")
        S.set_logfile(logfile)
        logs_silenced = True

    cprint("-" * 60)
    cprint("Atten optimization plan")
    cprint(f"bands: {bands}")
    cprint(f"uc_attens: {uc_attens}")
    cprint(f"dc_attens: {dc_attens}")
    cprint(f"logfile: {logfile}")
    cprint("-" * 60)

    tks = {}
    for b in bands:
        cprint(f"Setting up band {b}...")
        S.set_att_uc(b, uc_attens[0])
        S.set_att_dc(b, dc_attens[0])
        tks[b] = get_tracking_kwargs(S, cfg, b, kwargs=tracking_kwargs)
        tks[b].update({
            'return_data': False, 'make_plot': False, 'save_plot': False
        })
        if not skip_setup_notches:
            S.setup_notches(b, tone_power=tone_power,
                            new_master_assignment=False)

    for i, uc_atten in enumerate(uc_attens):
        cprint(f"Setting uc_atten to {uc_atten}")
        for b in bands:
            print(f"Serial fns for band {b}")
            S.set_att_uc(b, uc_atten)
            S.set_att_dc(b, dc_attens[0])
            # Should tracking setup go in dc_atten loop? Need to test.
            # S.tracking_setup(b, **tks[b])

        for j, dc_atten in enumerate(dc_attens):
            for b in bands:
                S.set_att_dc(b, dc_atten)
                print("Tracking Setup")
                S.run_serial_gradient_descent(b)
                S.run_serial_eta_scan(b)
                S.tracking_setup(b, **tks[b])
            # Take data
            atten_grid.append([uc_atten, dc_atten])
            start_times.append(time.time())
            datfile = S.take_stream_data(meas_time, make_freq_mask=False)
            datfiles.append(datfile)
            stop_times.append(time.time())

            # Analyze data
            seg = StreamSeg(*S.read_stream_data(datfile))
            f, Pxx = get_psd(S, seg.times, seg.sig)
            fmask = (f > 5) & (f < 50)
            # Array of whitenoise level for each readout chan
            wls = np.nanmedian(Pxx[:, fmask], axis=1)
            for k, b in enumerate(bands):
                rchans = seg.mask[b]  # Readout channels for band b
                rchans = rchans[rchans != -1]
                wl_medians[k, i, j] = np.nanmedian(wls[rchans])
            print(f"Median noise for uc={uc_atten}, dc={dc_atten}: "
                  f"{wl_medians[:, i, j]}")

    stop_time = time.time()
    atten_grid = np.array(atten_grid)

    opt_ucs = np.zeros_like(bands)
    opt_dcs = np.zeros_like(bands)
    for i, band in enumerate(bands):
        wls = wl_medians[i]
        uc_idx, dc_idx = np.unravel_index(np.argmin(wls, axis=None), wls.shape)
        opt_ucs[i] = uc_attens[uc_idx]
        opt_dcs[i] = dc_attens[dc_idx]
        cprint(f"Band {band}:")
        cprint(f"  Min White Noise: {wls[uc_idx, dc_idx]}")
        cprint(f"  UC Atten: {uc_attens[uc_idx]}")
        cprint(f"  DC Atten: {dc_attens[dc_idx]}")
        if update_cfg:
            cfg.dev.update_band(band, {
                'uc_att': uc_attens[uc_idx],
                'dc_att': dc_attens[dc_idx],
            })

    if logs_silenced:  # Returns logs to stdout
        S.set_logfile(None)

    summary = {
        'starts': start_times,
        'stops': stop_times,
        'uc_attens': uc_attens,
        'dc_attens': dc_attens,
        'atten_grid': atten_grid,
        'bands': bands,
        'dat_files': datfiles,
        'wl_medians': wl_medians,
        'opt_ucs': opt_ucs,
        'opt_dcs': opt_dcs,
    }
    fname = make_filename(S, 'optimize_atten_summary.npy')
    np.save(fname, summary, allow_pickle=True)
    S.pub.register_file(fname, 'optimize_atten_summary', format='npy')

    for i, band in enumerate(bands):
        wls = wl_medians[i]
        uc_idx, dc_idx = np.unravel_index(np.argmin(wls, axis=None), wls.shape)
        cprint(f"Band {band}:")
        cprint(f"  Min White Noise: {wls[uc_idx, dc_idx]}")
        cprint(f"  UC Atten: {uc_attens[uc_idx]}")
        cprint(f"  DC Atten: {dc_attens[dc_idx]}")
        if update_cfg:
            cfg.dev.update_band(band, {
                'uc_att': uc_attens[uc_idx],
                'dc_att': dc_attens[dc_idx],
            })
    if update_cfg:
        print(f"Updating cfg and dumping to {cfg.dev_file}")
        cfg.dev.dump(cfg.dev_file, clobber=True)

    plot_optimize_attens(S, summary)

    cprint(f"Finished atten scan. Summary saved to {fname}", True)
    cprint(f"Total duration: {stop_time - start_time} sec", True)

    if set_to_opt:
        for i, b in enumerate(bands):
            cprint("Setting attens to opt values and re-running setup notches "
                   f"for band {b}...")
            S.set_att_uc(b, opt_ucs[i])
            S.set_att_dc(b, opt_dcs[i])
            S.setup_notches(b, tone_power=tone_power,
                            new_master_assignment=False)
            S.run_serial_gradient_descent(b)
            S.run_serial_eta_scan(b)
            S.tracking_setup(b, **tks[b])

    return summary, fname


@set_action()
def optimize_uc_atten(S, cfg, band, drive=None, tunefile=None,
                      frac_pp=None, lms_freq=None, make_plots=False,
                      lms_gain=None, meas_time=10,
                      fit_curve=True, run_setup_notches=False):
    """
    Finds the drive power and uc attenuator value that minimizes the median
    noise within a band.

    Parameters
    ----------
    drive: int
        Drive to run uc_atten optimization. If None will pull from device cfg.
    band: int
        band to optimize noise on
    tunefile: str
        filepath to the tunefile for the band to be optimized
    frac_pp: float
        fraction full scale of the FR DAC used for tracking_setup
    lms_freq: float
        tracking frequency used for tracking_setup
    make_plots: bool
        If true, will make tracking plots
    meas_time : float
        Measurement time for noise PSD in seconds. Defaults to 10 sec.
    fit_curve: bool
        If True, will fit
    run_setup_notches : bool
        If True, will run setup notches after each uc_atten step.

    Returns
    -------
    min_med_noise : float
        The median noise at the optimized drive power
    atten : int
        Optimized uc attenuator value
    """
    band_cfg = cfg.dev.bands[band]
    if tunefile is None:
        tunefile = cfg.dev.exp['tunefile']
    if drive is None:
        drive = band_cfg['drive']
    if frac_pp is None:
        frac_pp = band_cfg['frac_pp']
    if lms_freq is None:
        lms_freq = band_cfg['lms_freq_hz']
    if lms_gain is None:
        lms_gain = band_cfg['lms_gain']

    tracking_kwargs = {
        'reset_rate_khz': 4, 'lms_freq_hz': lms_freq,
        'fraction_full_scale': frac_pp,
        'make_plot': True, 'save_plot': True, 'show_plot': False,
        'channel': [], 'nsamp': 2**18,
        'feedback_start_frac': 0.02, 'feedback_end_frac': 0.94,
        'lms_gain': lms_gain, 'return_data': True
    }

    S.load_tune(tunefile)

    attens = np.arange(30, -2, -2)

    medians = np.full(len(attens), np.nan)
    initial_median = None

    S.set_att_uc(band, 30)
    S.relock(band=band, tone_power=drive)
    S.setup_notches(band, tone_power=drive, new_master_assignment=False)

    for i, atten in enumerate(attens):
        cprint(f'Setting UC atten to: {atten}')
        S.set_att_uc(band, atten)

        if run_setup_notches:
            S.setup_notches(band, tone_power=drive,
                            new_master_assignment=False)

        print("Serial Gradient Descent")
        S.run_serial_gradient_descent(band)
        print("Serial Eta Scan")
        S.run_serial_eta_scan(band)
        print("Tracking setup")

        if make_plots:
            tracking_kwargs['channel'] = S.which_on(band)

        f, _, _ = S.tracking_setup(band, **tracking_kwargs)

        datafile = S.take_stream_data(meas_time)
        m, _ = analyze_noise_psd(S, band, datafile, fit_curve=fit_curve)

        medians[i] = m

        cprint(f"uc atten: {atten}\tmedian:{m}")

        # Checks to make sure noise doesn't go too far over original median
        if initial_median is None:
            initial_median = m
        if m > 4 * initial_median:
            cprint(f"Median noise is now 4 times what it was at atten=30, "
                   f"so exiting loop at uc_atten = {atten}", style=False)
            break

    min_idx = np.nanargmin(medians)
    uc_atten_opt = attens[min_idx]
    min_median = medians[min_idx]

    cprint(f"Optimal UC atten: {uc_atten_opt}", True)
    cprint(f'optimal noise: {min_median}', True)
    if not 2 < min_idx < len(attens) - 2:
        cprint("Optimal UC atten is close to the boundary. Consider doing a "
               "full drive optimization with a VNA ot find the drive and "
               "synth_scale combination that gives you the right amount of "
               "input power", style=False)

    # Summary files and plots
    fname = make_filename(S, f'b{band}_noise_vs_input_power.txt')
    np.savetxt(fname, np.array([attens, medians]).T)

    plt.figure()
    plt.plot(attens, medians)
    plt.title(f'Drive = {drive} in Band {band}', fontsize=18)
    plt.xlabel('UC Attenuator Value', fontsize=14)
    plt.ylabel('Median Channel Noise [pA/rtHz]', fontsize=14)

    fname = make_filename(S, 'noise_vs_uc_atten_b{band}.png', plot=True)
    plt.savefig(fname)
    S.pub.register_file(fname, 'noise_vs_atten', plot=True)
    plt.close()

    cprint("Rerunning setup-notches with optimal uc_atten")
    S.set_att_uc(band, uc_atten_opt)
    S.load_tune(tunefile)
    S.setup_notches(band, tone_power=drive, new_master_assignment=False)
    S.relock(band=band, tone_power=drive)
    S.run_serial_gradient_descent(band)
    S.run_serial_eta_scan(band)

    cfg.dev.update_experiment({'tunefile': S.tune_file})
    cfg.dev.update_band(band, {
        'uc_att': uc_atten_opt, 'optimized_uc_atten': True
    })
    return min_median, uc_atten_opt

