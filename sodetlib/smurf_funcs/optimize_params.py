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
import sodetlib.util as su
import sodetlib.smurf_funcs as sf
from pysmurf.client.util.pub import set_action


def optimize_tracking(S, cfg, bands, init_fracpp=None, phi0_number=5,
                      reset_rate_khz=None, show_plot=True, update_cfg=True):
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
    """
    # Channel data which will go into the output file
    out = {}
    bands = np.atleast_1d(bands)
    with np.errstate(invalid='ignore'):
        for band in bands:
            su.cprint(f"Optimizing tracking setup for band {band}")
            S.log("Loading tune and running initial tracking setup")
            tk = su.get_tracking_kwargs(
                S, cfg, band, kwargs={
                    'lms_freq_hz': None, 'meas_lms_freq': True}
            )

            if init_fracpp is not None:
                tk['fraction_full_scale'] = init_fracpp
            if reset_rate_khz is not None:
                tk['reset_rate_khz'] = reset_rate_khz
            else:  # This  sets this var based on the dev cfg
                reset_rate_khz = tk['reset_rate_khz']

            # Runs first tracking quality to do initial cut of bad chans
            rs, f, df, sync = sf.smurf_ops.tracking_quality(
                S, cfg, band, tracking_kwargs=tk, nphi0=1,
            )

            good_chans = np.where(rs > 0.95)[0]
            amp_scale_array = S.get_amplitude_scale_array(band)
            num_good, num_total = len(good_chans), np.sum(amp_scale_array != 0)
            su.cprint(f"{num_good}/{num_total} channels have r-squared > 0.95")
            if num_good/num_total < 0.5:
                su.cprint("This is less than 50%, so something is wrong with the tune.",
                          False)
                su.cprint("Not continuing with optimization...", False)
                return
            else:
                su.cprint(
                    "This is more than 50%, shutting off bad channels.", True)
                bad_chans = np.where((rs < 0.95) & (amp_scale_array != 0))[0]
                for c in bad_chans:
                    S.channel_off(band, c)

            # Reruns to get re-estimate of tracking freq
            rs, f, df, sync = sf.smurf_ops.tracking_quality(
                S, cfg, band, tracking_kwargs=tk, nphi0=1,
            )

            # Calculates actual tracking params
            lms_meas = S.lms_freq_hz[band]
            lms_freq = phi0_number * tk['reset_rate_khz'] * 1e3
            frac_pp = tk['fraction_full_scale'] * lms_freq / lms_meas

            tk['meas_lms_freq'] = False
            tk['fraction_full_scale'] = frac_pp
            tk['lms_freq_hz'] = lms_freq

            # Turns channels back on
            S.log("Relocking and running serial ops")
            S.relock(band)
            S.run_serial_gradient_descent(band)
            S.run_serial_eta_scan(band)

            chans = S.which_on(band)
            rs, f, df, sync = sf.smurf_ops.tracking_quality(
                S, cfg, band, tracking_kwargs=tk, show_plots=show_plot
            )

            nchans = len(S.which_on(band))
            nbad_chans = len(np.where(rs < 0.9)[0])
            su.cprint(
                f"{nbad_chans}/{nchans} Channels not tracking on band {band}")
            su.cprint(
                f'Optimized frac_pp: {frac_pp} for lms_freq = {lms_freq}')

            out[band] = {
                'chans': chans,
                'r2s': rs,
                'f': f,
                'df': df,
                'sync': sync,
                'frac_pp': frac_pp,
                'lms_freq_hz': lms_freq,
                'flux_ramp_rate_khz': reset_rate_khz
            }
        if update_cfg:
            cfg.dev.update_band(
                band, {'frac_pp': frac_pp, 'lms_freq_hz': lms_freq,
                       'flux_ramp_rate_khz': reset_rate_khz}
            )

    fname = make_filename(S, 'optimize_tracking.pkl')
    pkl.dump(out, open(fname, 'wb'))
    S.pub.register_file(fname, 'tracking_optimization', format='pkl')

    if update_cfg:
        cfg.dev.dump(cfg.dev_file, clobber=True)

    return out


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
    su.cprint(f'best_chan: {best_chan}\tfspan: {f_span[best_chan]}')
    return best_chan


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
    f, Pxx = su.get_psd(S, times, phase, detrend=detrend, nperseg=nperseg)
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

    su.cprint(f"Channels {cut_chans} were cut due to high phase span.", False)
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
            su.cprint(f"Vg adjustment would go out of range ({vg_min}, {vg_max}). "
                      f"Unable to change {amp_name}_Id to desired value", False)
            return False

        if amp_name == 'hemt':
            S.set_hemt_gate_voltage(Vg_next)
        else:
            S.set_50k_amp_gate_voltage(Vg_next)
        time.sleep(0.2)
    su.cprint(f"Max allowed Vg iterations ({max_iter}) has been reached. "
              f"Unable to get target Id for {amp_name}.", False)
    return False


@set_action()
def plot_optimize_attens(S, summary, wlmax=1000, vmin=None, vmax=None):
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

    fig, axes = plt.subplots(2, 4, figsize=(18, 6),
                             gridspec_kw={'hspace': 0.4, 'wspace': 0.4})
    fig.patch.set_facecolor('white')

    if len(ucs) == 1:
        fig.suptitle(f"Median White Noise, UC Atten = {ucs[0]}")
    elif len(dcs) == 1:
        fig.suptitle(f"Median White Noise, DC Atten = {dcs[0]}")

    if vmin is None:
        vmin = np.min(wls) * 0.9
    if vmax is None:
        vmax = min(np.max(wls) * 1.1, 500)  # We don't rly care if wl > 800

    for i, band in enumerate(summary['bands']):
        ax = axes[band // 4, band % 4]

        if len(ucs) == 1:  # Sweep is dc only
            uc = ucs[0]
            _wls = wls[i, 0, :]
            ax.plot(dcs, _wls)
            ax.set(xlabel="DC atten", ylabel="white noise (pA/rt(Hz))")
            ax.set(title=f"Band {band}")
            ax.set(ylim=(vmin, vmax))

        elif len(dcs) == 1:  # Sweep is uc only
            dc = dcs[0]
            _wls = wls[i, :, 0]
            ax.plot(ucs, _wls)
            ax.set(xlabel="UC atten", ylabel="white noise (pA/rt(Hz))")
            ax.set(title=f"Band {band}")
            ax.set(ylim=(vmin, vmax))

        else:  # Do full 2d heat map
            im = ax.pcolor(ucs, dcs, wls[band].T, vmin=vmin, vmax=vmax)
            ax.set(xlabel="UC atten", ylabel="DC atten", title=f"Band {band}")
            if i == 0:
                fig.colorbar(im, label='Median White Noise [pA/rt(Hz)]',
                             ax=axes.ravel().tolist())

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
        logfile = su.make_filename(S, 'optimize_atten.log')
        print(f"Writing pysmurf logs to {logfile}")
        S.set_logfile(logfile)
        logs_silenced = True

    su.cprint("-" * 60)
    su.cprint("Atten optimization plan")
    su.cprint(f"bands: {bands}")
    su.cprint(f"uc_attens: {uc_attens}")
    su.cprint(f"dc_attens: {dc_attens}")
    su.cprint(f"logfile: {logfile}")
    su.cprint("-" * 60)

    tks = {}
    for b in bands:
        su.cprint(f"Setting up band {b}...")
        S.set_att_uc(b, uc_attens[0])
        S.set_att_dc(b, dc_attens[0])
        tks[b] = su.get_tracking_kwargs(S, cfg, b, kwargs=tracking_kwargs)
        tks[b].update({
            'return_data': False, 'make_plot': False, 'save_plot': False
        })
        if not skip_setup_notches:
            S.setup_notches(b, tone_power=tone_power,
                            new_master_assignment=False)

    for i, uc_atten in enumerate(uc_attens):
        su.cprint(f"Setting uc_atten to {uc_atten}")
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
            seg = su.StreamSeg(*S.read_stream_data(datfile))
            f, Pxx = su.get_psd(S, seg.times, seg.sig)
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
        su.cprint(f"Band {band}:")
        su.cprint(f"  Min White Noise: {wls[uc_idx, dc_idx]}")
        su.cprint(f"  UC Atten: {uc_attens[uc_idx]}")
        su.cprint(f"  DC Atten: {dc_attens[dc_idx]}")
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
    fname = su.make_filename(S, 'optimize_atten_summary.npy')
    np.save(fname, summary, allow_pickle=True)
    S.pub.register_file(fname, 'optimize_atten_summary', format='npy')

    for i, band in enumerate(bands):
        wls = wl_medians[i]
        uc_idx, dc_idx = np.unravel_index(np.argmin(wls, axis=None), wls.shape)
        su.cprint(f"Band {band}:")
        su.cprint(f"  Min White Noise: {wls[uc_idx, dc_idx]}")
        su.cprint(f"  UC Atten: {uc_attens[uc_idx]}")
        su.cprint(f"  DC Atten: {dc_attens[dc_idx]}")
        if update_cfg:
            cfg.dev.update_band(band, {
                'uc_att': uc_attens[uc_idx],
                'dc_att': dc_attens[dc_idx],
            })
    if update_cfg:
        print(f"Updating cfg and dumping to {cfg.dev_file}")
        cfg.dev.dump(cfg.dev_file, clobber=True)

    plot_optimize_attens(S, summary)

    su.cprint(f"Finished atten scan. Summary saved to {fname}", True)
    su.cprint(f"Total duration: {stop_time - start_time} sec", True)

    if set_to_opt:
        for i, b in enumerate(bands):
            su.cprint("Setting attens to opt values and re-running setup notches "
                      f"for band {b}...")
            S.set_att_uc(b, opt_ucs[i])
            S.set_att_dc(b, opt_dcs[i])
            S.setup_notches(b, tone_power=tone_power,
                            new_master_assignment=False)
            S.run_serial_gradient_descent(b)
            S.run_serial_eta_scan(b)
            S.tracking_setup(b, **tks[b])

    return summary, fname
