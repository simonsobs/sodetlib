"""
Module with functions that optimize device parameters with respect to
psd noise.
"""
import numpy as np
import os
import time
from scipy import signal
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pysmurf.client.util.pub import set_action

@set_action()
def get_median_noise(S, cfg, band, meas_time=30):
    """
    Takes PSD and returns the median noise of all active channels.

    Parameters
    ------------
    S:  (pysmurf.client.SmurfControl)
        Pysmurf control instance
    cfg: (DetConfig)
        Detector config object
    band : (int)
        band to get median noise for.

    Returns
    ---------
    median_noise: (float)
        Median noise for the specified band.
    """
    band_cfg = cfg.dev.bands[band]
    S.run_serial_gradient_descent(band)
    S.run_serial_eta_scan(band)
    S.tracking_setup(
        band, reset_rate_khz=4, fraction_full_scale=band_cfg['frac_pp'],
        make_plot=band_cfg.get('make_plot', False),
        save_plot=band_cfg.get('save_plot', False),
        channel=S.which_on(band), nsamp=2**18, lms_gain=band_cfg['lms_gain'],
        lms_freq_hz=band_cfg['lms_freq_hz'], feedback_start_frac=1/12,
        feedback_end_frac=0.98, show_plot=False
    )
    datafile = S.take_stream_data(meas_time)
    median_noise, _ = analyze_noise_psd(S, band, datafile)
    return median_noise


@set_action()
def analyze_noise_psd(S, band, dat_file):
    """
    Finds the white noise level, 1/f knee, and 1/f polynomial exponent of a
    noise timestream and returns the median white noise level of all channels
    and a dictionary of fitted values per channel.

    Parameters
    ----------
    band : int
        band to optimize noise on
    dat_file : str
        filepath to timestream data to analyze
    ctime : str
        ctime used for saved data/plot titles

    Returns
    -------
    median_noise : float
        median white noise level of all channels analyzed in pA/rtHz

    outdict : dict of{int:dict of{str:float}}
        dictionary with each key a channel number and each channel number
        another dictionary containing the fitted 1/f knee, 1/f exponent, and
        white noise level in pA/rtHz
    """
    outdict = {}
    datafile = dat_file
    nperseg = 2**16
    detrend = 'constant'
    timestamp, phase, mask = S.read_stream_data(datafile)
    phase *= S.pA_per_phi0/(2.*np.pi)
    num_averages = S.config.get('smurf_to_mce')['num_averages']
    fs = S.get_flux_ramp_freq()*1.0E3/num_averages
    wls = []
    for chan in S.which_on(band):
        if chan < 0:
            continue
        ch_idx = mask[band, chan]
        f, Pxx = signal.welch(phase[ch_idx], nperseg=nperseg,fs=fs, detrend=detrend)
        Pxx = np.sqrt(Pxx)
        popt, pcov, f_fit, Pxx_fit = S.analyze_psd(f, Pxx)
        wl,n,f_knee = popt
        wls.append(wl)
        outdict[chan] = {}
        outdict[chan]['fknee']=f_knee
        outdict[chan]['noise index']=n
        outdict[chan]['white noise']=wl
    median_noise = np.median(np.asarray(wls))
    return median_noise, outdict


@set_action()
def optimize_bias(S, target_Id, vg_min, vg_max, amp_name, max_iter=30):
    """
    Scans through bias voltage for hemt or 50K amplifier to get the correct
    gate voltage for a target current.

    Parameters
    -----------
    S (pysmurf.client.SmurfControl):
        PySmurf control object
    target_Id (float):
        Target amplifier current
    vg_min (float):
        Minimum allowable gate voltage
    vg_max (float):
        Maximum allowable gate voltage
    amp_name (str):
        Name of amplifier. Must be either "hemt" or "50K'.
    max_iter (int):
        Maximum number of iterations to find voltage. Defaults to 30.

    Returns
    --------
    success (bool):
        Returns a boolean signaling whether voltage scan has been successful.
        The set voltages can be read with S.get_amplifier_biases().
    """
    if amp_name not in ['hemt', '50K']:
        raise ValueError(cprint(f"amp_name must be either 'hemt' or '50K'", False))

    for _ in range(max_iter):
        amp_biases = S.get_amplifier_biases(write_log=True)
        Vg = amp_biases[f"{amp_name}_Vg"]
        Id = amp_biases[f"{amp_name}_Id"]
        delta = target_Id - Id
        # Id should be within 0.5 from target without going over.
        if 0 <= delta < 0.5:
            return True

        if amp_name=='hemt':
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
                            frac_pp=None, lms_freq=None, make_plots=True,
                            meas_time=None, fixed_drive=False):
    """
    Finds the drive power and uc attenuator value that minimizes the median
    noise within a band.

    Parameters
    ----------
    band: (int)
        band to optimize noise on
    tunefile: (str)
        filepath to the tunefile for the band to be optimized
    dr_start: (int)
        drive power to start all channels with, default is 12
    frac_pp: (float)
        fraction full scale of the FR DAC used for tracking_setup
    lms_freq: (float)
        tracking frequency used for tracking_setup
    make_plots: (bool)
        If true, will make median noise plots

    Returns
    -------
    min_med_noise : float
        The median noise at the optimized drive power
    atten : int
        Optimized uc attenuator value
    cur_dr : int
        Optimized dr value
    meas_time : float
        Measurement time for noise PSD in seconds.
    fixed_drive: bool
        If true, will not try to vary drive to search for global minimum.
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

    S.load_tune(tunefile)
    drive = dr_start
    attens = np.arange(30, -2, -2)
    checked_drives = []
    found_min = False

    # Looping over drive powers
    while not found_min:
        cprint(f"Setting Drive to {drive}")
        ctime = S.get_timestamp()
        S.set_att_uc(band, 30)
        S.relock(band=band, drive=drive)

        medians = []
        initial_median = None

        for atten in attens:
            cprint(f'Setting UC atten to: {atten}')
            S.set_att_uc(band, atten)

            kwargs = {}
            if meas_time is not None:
                kwargs['meas_time'] = meas_time

            m = get_median_noise(S, cfg, band, **kwargs)
            medians.append(m)
            # Checks to make sure noise doesn't go too far over original median
            if initial_median is None:
                initial_median = m
            if m > 4 * initial_median:
                cprint(f"Median noise is now 4 times what it was at atten=30, "
                       f"so exiting loop at uc_atten = {atten}", style=False)
                break

        if make_plots:
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

        medians= np.asarray(medians)
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
            drive += 1 if min_arg == 0 else -1
            if drive in checked_drives:
                cprint(f"Drive {drive} has already been checked!! "
                       f"Exiting loop unsuccessfully", False)
                found_min = False
                break

    if found_min:
        cprint(f'found optimum dr = {drive}, and optimum uc_att = {min_atten}',
               style=True)
        S.set_att_uc(band, min_atten)
        S.load_tune(tunefile)
        S.relock(band=band,drive=drive)
        S.run_serial_gradient_descent(band)
        S.run_serial_eta_scan(band)

        cfg.dev.update_band(band, {
            'uc_att': min_atten, 'drive': drive
        })
        return min_median, min_atten, drive
