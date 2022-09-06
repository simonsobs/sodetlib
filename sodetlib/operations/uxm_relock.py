import time
import numpy as np
import sodetlib as sdl
from sodetlib.operations.tracking import relock_tracking_setup
from sodetlib.operations import uxm_setup


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
    """

    if tunefile is None:
        tunefile = cfg.dev.exp['tunefile']

    S.load_tune(tunefile)

    for band in bands:
        bcfg = cfg.dev.bands[band]
        S.set_att_uc(band, bcfg['uc_att'])
        S.set_att_dc(band, bcfg['dc_att'])
        sdl.pub_ocs_log(S, f"Relocking tune: Band {band}")
        if setup_notches:
            S.log(f"Setup notches, new_master_assignment={new_master_assignment}")
            S.setup_notches(band, new_master_assignment=new_master_assignment)
        else:
            S.relock(band)

        S.run_serial_gradient_descent(band)
        S.run_serial_eta_scan(band)

    if setup_notches and update_cfg:
        # Update tunefile
        cfg.dev.update_experiment({'tunefile': S.tune_file}, update_file=True)

    return True, None


@sdl.set_action()
def uxm_relock(S, cfg, bands=None, disable_bad_chans=True, show_plots=False,
               setup_notches=False, new_master_assignment=False,
               reset_rate_khz=None, nphi0=None):
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
    disable_bad_chans : bool
        If true, will disable tones for bad-tracking channels

    Returns
    --------
    summary : dict
        Dictionary containing a summary of all run operations.
    """
    if bands is None:
        bands = np.arange(8)
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
    summary['timestamps'].append(('setup_amps', time.time()))
    sdl.set_session_data(S, 'timestamps', summary['timestamps'])
    success, summary['amps'] = uxm_setup.setup_amps(S, cfg)

    if not success:
        return False, summary

    #############################################################
    # 3. Load tune
    #############################################################
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
    summary['timestamps'].append(('tracking_setup', time.time()))
    sdl.set_session_data(S, 'timestamps', summary['timestamps'])

    tr = relock_tracking_setup(
        S, cfg, bands, show_plots=show_plots,
        reset_rate_khz=reset_rate_khz, nphi0=nphi0
    )
    summary['tracking_setup_results'] = tr

    sdl.set_session_data(S, 'tracking_setup_results', {
        'bands': tr.bands, 'channels': tr.channels,
        'r2': tr.r2, 'f_ptp': tr.f_ptp, 'df_ptp': tr.df_ptp,
        'is_good': tr.is_good,
    })

    # Check that the number of good tracing channels is larger than the
    # min_good_tracking_frac
    if tr.ngood / tr.nchans < cfg.dev.exp['min_good_tracking_frac']:
        return False, summary

    #############################################################
    # 5. Noise
    #############################################################
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
