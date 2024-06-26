import time
import numpy as np
import sodetlib as sdl
from sodetlib.operations.tracking import relock_tracking_setup
from sodetlib.operations import uxm_setup

import matplotlib.pyplot as plt
import os
if not os.environ.get('NO_PYSMURF', False):
    from pysmurf.client.base.smurf_control import SmurfControl



@sdl.set_action()
def reload_tune(S, cfg, bands, setup_notches=False,
                new_master_assignment=False, tunefile=None, update_cfg=True):
    """
    Reloads an existing tune, runs setup-notches and serial grad descent
    and eta scan.

    Args
    -----
    S : SmurfControl
        Pysmurf instance
    cfg : DetConfig
        Det config instance
    bands : list, optional
        List of bands to run
    setup_notches : bool
        Whether to run setup_notches
    new_master_assignment : bool
        Whether to create a new master assignment
    tunefile : str
        Tunefile to load. Defaults to the tunefile in the device cfg.
    update_cfg : bool
        If true, will update the device cfg with the tunefile created by
        setup notches.
    """

    if tunefile is None:
        tunefile = cfg.dev.exp['tunefile']

    S.load_tune(tunefile)

    for band in bands:
        sdl.stop_point(S)
        bcfg = cfg.dev.bands[band]
        S.set_att_uc(band, bcfg['uc_att'])
        S.set_att_dc(band, bcfg['dc_att'])
        sdl.pub_ocs_log(S, f"Relocking tune: Band {band}")
        if setup_notches:
            S.log(f"Setup notches, new_master_assignment={new_master_assignment}")
            S.setup_notches(band, new_master_assignment=new_master_assignment)
        else:
            S.relock(band)

    if setup_notches and update_cfg:
        # Update tunefile
        cfg.dev.update_experiment({'tunefile': S.tune_file}, update_file=True)

    run_grad_descent_and_eta_scan(S, cfg, bands=bands, update_tune=False)

    return True, None


@sdl.set_action()
def run_grad_descent_and_eta_scan(
    S, cfg, bands=None, update_tune=False, force_run=False, max_iters=None,
    gain=None): 
    """
    This function runs serial gradient and eta scan for each band.
    Critically, it pulls in gradient descent tune parameters from the device
    config and uses them to setup the grad descent operation before running.

    Args
    ----
    S : SmurfControl
        Pysmurf instance
    cfg : DetConfig
        Det config instance
    bands : list, optional
        List of bands to run on. Defaults to all 8
    update_tune : bool
        If this is set to True, the new resonance frequency and eta parameters
        will be loaded into the smurf tune, and a new tunefile will be written
        based on the new measurements.
    force_run : bool
        If True, will reset the ``etaScanInProgress`` variable to force it
        to re-run. This might be necessary if serial gradient descent failed out.
    """

    if bands is None:
        bands = cfg.dev.exp['active_bands']
    bands = np.atleast_1d(bands)

    for b in bands:
        sdl.stop_point(S)
        in_progress_reg = S._cryo_root(b) + 'etaScanInProgress'
        if S._caget(in_progress_reg):
            if force_run:
                S._caput(in_progress_reg, 0)
            else:
                raise RuntimeError(
                    "Failed during grad descent -- scan already in progress"
                )

        bcfg = cfg.dev.bands[b]

        if max_iters is None:
            max_iters = bcfg['gradientDescentMaxIters']
        if gain is None:
            gain = bcfg['gradientDescentGain']

        S.set_gradient_descent_step_hz(b, bcfg['gradientDescentStepHz'])
        S.set_gradient_descent_max_iters(b, max_iters)
        S.set_gradient_descent_gain(b, gain)
        S.set_gradient_descent_converge_hz(b, bcfg['gradientDescentConvergeHz'])
        S.set_gradient_descent_beta(b, bcfg['gradientDescentBeta'])

        S.log(f"Running grad descent and eta scan on band {b}")

        init_scale_array = S.get_amplitude_scale_array(b)
        init_center_freq_array = S.get_center_frequency_array(b)
        S.run_serial_gradient_descent(b)

        if (S.get_amplitude_scale_array(b) != init_scale_array).any():
            S.log("Grad descent errored out! Resetting state")
            S.set_amplitude_scale_array(b, init_scale_array)
            S.set_center_frequency_array(b, init_center_freq_array)
            S._caput(in_progress_reg, 0)
            raise RuntimeError(
                "Failed during grad descent -- check smurf-streamer logs"
            )

        S.run_serial_eta_scan(b)

        if update_tune:
            # This basically does the inverse of what's in the relock function
            band_center_mhz = S.get_band_center_mhz(b)
            subband_offset = S.get_tone_frequency_offset_mhz(b)
            center_freq_array = S.get_center_frequency_array(b)
            eta_phase_array = S.get_eta_phase_array(b)
            eta_mag_array = S.get_eta_mag_array(b)
            res_freqs = band_center_mhz + subband_offset + center_freq_array

            for res in S.freq_resp[b]['resonances'].values():
                ch = res['channel']
                f0 = res['freq']
                # Pysmurf does this to ignore channels with bad freqs so I 
                # guess we will too
                for ll, hh in S._bad_mask:
                    if ll < f0 < hh:
                        ch = -1

                res['freq'] = res_freqs[ch]
                res['offset'] = center_freq_array[ch]
                res['eta_phase'] = eta_phase_array[ch]
                res['eta_scaled'] = eta_mag_array[ch]

    if update_tune:
        S.log("Saving tune!")
        S.save_tune()
        cfg.dev.exp['tunefile'] = S.tune_file
        cfg.dev.update_file()


def get_full_band_sweep(S, cfg, band, chan):
    """
    This runs full_band_ampl_sweep to get the resonator response around a single
    channel. This is useful for debugging bad channels.

    Args
    -----
    S : SmurfControl
        Pysmurf instance
    cfg : DetConfig
        Det Config instance
    band : int
        smurf-band of channel
    channel : int
        smurf-channel of channel

    Returns
    ----------
    fs : np.ndarray
        Array of frequencies that were swept over
    resp : np.ndarray
        Complex transmission across the frequency range
    """
    sb = S.get_subband_from_channel(band, chan)
    tone_power = cfg.dev.bands[band]['tone_power']
    center_freq_array = S.get_center_frequency_array(band)
    amp_scale_array = S.get_amplitude_scale_array(band)
    eta_phase_array = S.get_eta_phase_array(band)
    eta_mag_array = S.get_eta_mag_array(band)
    fb_enable = S.get_feedback_enable(band)
    fb_enable_arr = S.get_feedback_enable_array(band)

    try:
        fs, resp = S.full_band_ampl_sweep(band, [sb], tone_power, 2, n_step=256)
    finally:
        S.set_center_frequency_array(band, center_freq_array)
        S.set_amplitude_scale_array(band, amp_scale_array)
        S.set_eta_phase_array(band, eta_phase_array)
        S.set_eta_mag_array(band, eta_mag_array)
        S.set_feedback_enable(band, fb_enable)
        S.set_feedback_enable_array(band, fb_enable_arr)
    return fs, resp


def plot_channel_resonance(S, cfg, band, chan):
    """
    Measures and plots resonator properties for a single smurf channel.
    This will perform a scan around the specified channel, and will plot the
    resonance dip along with where smurf thinks the resonance frequency is.
    This is very useful for debugging why a channel isn't reading out properly.

    Args
    -----
    S : (SmurfControl)
        pysmurf instance
    cfg : (DetConfig)
        DetConfig instance
    band : (int)
        smurf band number
    chan : (int)
        smurf channel number
    """
    res_freq = S.channel_to_freq(band, chan)
    fs, resp = get_full_band_sweep(S, cfg, band, chan)

    fs = fs.ravel() + S.get_band_center_mhz(band)
    resp = resp.ravel()
    fs = fs[np.abs(resp) > 0]
    resp = resp[np.abs(resp) > 0]

    fig, axes = plt.subplots(2, 1, figsize=(10, 10))

    # Res-dip plot
    ax = axes[0]
    ax.plot(fs, np.abs(resp))
    ax = ax.twinx()
    ax.plot(fs, np.unwrap(np.angle(resp)), color='C1')
    ax.axvline(res_freq, color='grey', ls='--')

    # Circ plot
    eta_phase = S.get_eta_phase_array(band)[chan]
    eta = np.exp(1.0j * eta_phase * (2*np.pi) / 360)
    Q = np.real(resp * eta)
    I = -np.imag(resp * eta)

    ax = axes[1]
    ax.axvline(0, color='grey')
    ax.axhline(0, color='grey')
    ax.scatter(I, Q, c=np.abs(resp))
    ax.scatter(np.real(resp), np.imag(resp), c=np.abs(resp), alpha=0.3)

    return fig, axes


@sdl.set_action()
def uxm_relock(
    S, cfg, bands=None, show_plots=False,
    setup_notches=False, new_master_assignment=False, reset_rate_khz=None,
    nphi0=None, skip_setup_amps=False):
    """
    Relocks resonators by running the following steps:

        1. Reset state (all off, disable waveform, etc.)
        2. Set amps and check tolerance
        3. load tune, setup_notches, serial grad descent and eta scan
        4. Tracking setup
        5. Measure noise

    Args
    -----
    S : SmurfControl
        Pysmurf instance
    cfg : DetConfig
        Det config instance
    bands : list, optional
        List of bands to run on. Defaults to all 8
    setup_notches : bool
        If True, will run setup notches instead of relocking
    new_master_assignment : bool
        Whether to run setup_notches with new_master_assignment.
        Defaults to False
    reset_rate_khz : float, optional
        Flux Ramp Reset Rate to set (kHz), defaults to the value in the dev cfg
    nphi0 : int, optional
        Number of phi0's to ramp through. Defaults to the value that was used
        during setup.
    show_plots : bool
        If True will show plots
    skip_setup_amps : bool
        If True will skip amplifier setup.

    Returns
    --------
    summary : dict
        Dictionary containing a summary of all run operations.
    """
    if bands is None:
        bands = cfg.dev.exp['active_bands']
    bands = np.atleast_1d(bands)

    exp = cfg.dev.exp

    #############################################################
    # 1. Reset to known state
    #############################################################
    S.all_off()  # Turn off Flux ramp, tones, and biases
    S.set_rtm_arb_waveform_enable(0)
    S.set_filter_disable(0)
    S.set_downsample_factor(exp['downsample_factor'])
    if exp['coupling_mode'] == 'dc':
        S.set_mode_dc()
    else:
        S.set_mode_ac()

    for band in bands:
        band_cfg = cfg.dev.bands[band]
        S.set_synthesis_scale(band, exp['synthesis_scale'])
        S.set_att_uc(band, band_cfg['uc_att'])
        S.set_att_dc(band, band_cfg['dc_att'])
        S.set_band_delay_us(band, band_cfg['band_delay_us'])
        S.amplitude_scale[band] = band_cfg['tone_power']

    summary = {}
    summary['timestamps'] = []
    #############################################################
    # 2. Setup amps
    #############################################################
    sdl.stop_point(S)
    if not skip_setup_amps:
        summary['timestamps'].append(('setup_amps', time.time()))
        sdl.set_session_data(S, 'timestamps', summary['timestamps'])
        success, summary['amps'] = uxm_setup.setup_amps(S, cfg)

        if not success:
            return False, summary
    else:
        print("Skipping amp setup")

    #############################################################
    # 3. Load tune
    #############################################################
    sdl.stop_point(S)
    summary['timestamps'].append(('load_tune', time.time()))
    sdl.set_session_data(S, 'timestamps', summary['timestamps'])
    success, summary['reload_tune'] = reload_tune(
        S, cfg, bands, setup_notches=setup_notches,
        new_master_assignment=new_master_assignment)

    sdl.set_session_data(S, 'reload_tune', summary['reload_tune'])
    if not success:
        return False, summary

    #############################################################
    # 4. Tracking Setup
    #############################################################
    sdl.stop_point(S)
    summary['timestamps'].append(('tracking_setup', time.time()))
    sdl.set_session_data(S, 'timestamps', summary['timestamps'])

    tr = relock_tracking_setup(
        S, cfg, bands, show_plots=show_plots,
        reset_rate_khz=reset_rate_khz, nphi0=nphi0
    )
    summary['tracking_setup_results'] = tr

    #############################################################
    # 5. Noise
    #############################################################
    sdl.stop_point(S)
    summary['timestamps'].append(('noise', time.time()))
    sdl.set_session_data(S, 'timestamps', summary['timestamps'])
    am, summary['noise'] = sdl.noise.take_noise(S, cfg, 30,
                                                show_plot=show_plots,
                                                save_plot=True)
    summary['noise']['am'] = am

    sdl.set_session_data(S, 'noise', {
        'band_medians': summary['noise']['band_medians']
    })

    summary['timestamps'].append(('end', time.time()))
    sdl.set_session_data(S, 'timestamps', summary['timestamps'])

    return True, summary
