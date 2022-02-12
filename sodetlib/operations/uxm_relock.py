import time
import numpy as np
import sodetlib as sdl


@sdl.set_action()
def reload_amps(S, cfg):
    """
    Reloads amplifier biases from dev cfg and checks that drain-currents fall
    within tolerance.

    Args
    ----
    S : SmurfControl
        Pysmurf instance
    cfg : DetConfig
        Det config instance
    """
    summary = {}

    exp = cfg.dev.exp
    sdl.pub_ocs_log(S, 'setting amp voltage')
    S.set_50k_amp_gate_voltage(cfg.dev.exp['amp_50k_Vg'])
    S.set_hemt_gate_voltage(cfg.dev.exp['amp_hemt_Vg'])
    S.C.write_ps_en(0b11)
    time.sleep(exp['amp_enable_wait_time'])

    biases = S.get_amplifier_biases()
    summary['biases'] = biases

    exp = cfg.dev.exp

    tol_hemt, tol_50k = exp['amp_hemt_Id_tolerance'], exp['amp_50k_Id_tolerance']
    in_range_hemt = np.abs(biases['hemt_Id'] - exp['amp_hemt_Id']) < tol_hemt
    in_range_50k = np.abs(biases['50k_Id'] - exp['amp_50k_Id']) < tol_50k

    if not (in_range_50k and in_range_hemt):
        S.log("Hemt or 50K Amp drain current not within tolerance")
        S.log(f"Target hemt Id: {exp['amp_hemt_Id']}")
        S.log(f"Target 50K Id: {exp['amp_50k_Id']}")
        S.log(f"tolerance: {(tol_hemt, tol_50k)}")
        S.log(f"biases: {biases}")

        summary['success'] = False
        sdl.pub_ocs_log(S, 'Failed to set amp voltages')
        sdl.pub_ocs_data(S, {'amp_summary': summary})
        return False, summary

    summary['success'] = True 
    sdl.pub_ocs_log(S, 'Succuessfully to set amp voltages')
    sdl.pub_ocs_data(S, {'amp_summary': summary})
    return True, summary


@sdl.set_action()
def reload_tune(S, cfg, bands, setup_notches=True,
                new_master_assignment=False, tunefile=None):
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
        S.set_att_uc(band)
        sdl.pub_ocs_log(S, f"Relocking tune: Band {band}")
        if setup_notches:
            S.log(f"Setup notches, new_master_assignment={new_master_assignment}")
            S.setup_notches(band, new_master_assignment=new_master_assignment)
        else:
            S.relock(band)

        S.run_serial_gradient_descent(band)
        S.run_serial_eta_scan(band)

    return True, None


@sdl.set_action()
def relock_tracking_setup(S, cfg, bands, reset_rate_khz=None, nphi0=None,
                          show_plots=False, disable_bad_chans=True):
    """
    Sets up tracking for smurf. This assumes you already have optimized
    lms_freq and frac-pp for each bands in the device config. This function
    will chose the flux-ramp fraction-full-scale by averaging the optimized
    fractions across the bands you're running on.

    This function also allows you to set reset_rate_khz and nphi0. The
    fraction-full-scale, and lms frequencies of each band will be automatically
    adjusted based on their pre-existing optimized values.

    Additional keyword args specified will be passed to S.tracking_setup.

    Args
    -----
    S : SmurfControl
        Pysmurf instance
    cfg : DetConfig
        Det config instance
    reset_rate_khz : float, optional
        Flux Ramp Reset Rate to set (kHz), defaults to the value in the dev cfg
    nphi0 : int, optional
        Number of phi0's to ramp through. Defaults to the value that was used
        during setup.
    disable_bad_chans : bool
        If true, will disable tones for bad-tracking channels

    Returns
    --------
    res : dict
        Dictionary of results of all tracking-setup calls, with the bands number
        as key.
    """
    bands = np.atleast_1d(bands)
    nbands = len(bands)
    exp = cfg.dev.exp

    # Arrays containing the optimized tracking parameters for each band
    frac_pp0 = np.zeros(nbands)
    lms_freq0 = np.zeros(nbands)  # Hz
    reset_rate_khz0 = exp['flux_ramp_rate_khz']  # This is assumed to be the same for each band
    init_nphi0 = exp['nphi0']

    for i, b in enumerate(bands):
        bcfg = cfg.dev.bands[b]
        frac_pp0[i] = bcfg['frac_pp']
        lms_freq0[i] = bcfg['lms_freq_hz'] 

    # Choose frac_pp to be the mean of all running bands.
    # This is the frac-pp at the flux-ramp-rate used for optimization
    fpp0 = np.mean(frac_pp0)

    # Adjust fpp, lms_freq, and flux-ramp-rate depending on desired
    # flux-ramp-rate and nphi0
    fpp, lms_freqs = fpp0, lms_freq0
    if nphi0 is not None:
        fpp *= nphi0 / init_nphi0
        lms_freqs *= fpp / fpp0
    if reset_rate_khz is not None:
        lms_freqs *= reset_rate_khz / reset_rate_khz0
    else:
        reset_rate_khz = reset_rate_khz0

    # Runs tracking setup
    summary = {
        'bands': bands,
        'tracking_results': []
    }
    tk = sdl.get_tracking_kwargs(S, cfg, bands[0])
    tk['show_plot'] = show_plots
    tk['reset_rate_khz'] = reset_rate_khz
    tk['fraction_full_scale'] = fpp
    success = True
    for i, band in enumerate(bands):
        tk.update({'lms_freq_hz': lms_freqs[i]})
        f, df, sync = S.tracking_setup(band, **tk)

        r2 = sdl.compute_tracking_quality(S, f, df, sync)
        f_ptp = np.ptp(f, axis=0)
        df_ptp = np.ptp(df, axis=0)

        # Cuts
        f_ptp_range = exp['f_ptp_range']
        df_ptp_range = exp['df_ptp_range']
        good_chans = (r2 > exp['r2_min'])  \
            & (f_ptp_range[0] < f_ptp) & (f_ptp < f_ptp_range[1])  \
            & (df_ptp_range[0] < df_ptp) & (df_ptp < df_ptp_range[1])
        num_good = np.sum(good_chans)
        num_tot = len(good_chans)

        summary['tracking_results'].append({
            'f': f, 'df': df, 'sync': sync, 'r2': r2, 'good_chans': good_chans
        })

        S.log(f"Band {band}: {num_good} / {num_tot} good tracking chans")
        if num_good / num_tot < exp['min_good_tracking_frac']:
            S.log(f"Not enough good channels on band {band}!!")
            S.log(f"Something is probably wrong")
            success = False

        if disable_bad_chans:
            asa = S.get_amplitude_scale_array(band)
            asa[~good_chans] = 0
            S.set_amplitude_scale_array(asa)

    return success, summary


@sdl.set_action()
def uxm_relock(S, cfg, bands=None, disable_bad_chans=True, show_plots=False,
               setup_notches=False, new_master_assignment=False,
               reset_rate_khz=None, nphi0=None):
    """
    Relocks resonators by running the following steps

        1. Reset state (all off, disable waveform, etc.)
        2. Set amps and check tolerance 
        3. load tune, setup_notches, serial grad descent and eta scan
        6. Tracking setup
        7. Measure noise

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
    success, summary['reload_amps'] = reload_amps(S, cfg)
    if not success:
        return False, summary

    #############################################################
    # 3. Load tune
    #############################################################
    summary['timestamps'].append(('load_tune', time.time()))
    success, summary['reload_tune'] = reload_tune(
        S, cfg, bands, setup_notches=setup_notches,
        new_master_assignment=new_master_assignment)
    if not success:
        return False, summary

    #############################################################
    # 4. Tracking Setup
    #############################################################
    summary['timestamps'].append(('tracking_setup', time.time()))
    success, summary['tracking_setup'] = relock_tracking_setup(
        S, cfg, bands, show_plots=show_plots,
        disable_bad_chans=disable_bad_chans,
        reset_rate_khz=reset_rate_khz, nphi0=nphi0
    )
    if not success:
        return False, summary

    #############################################################
    # 5. Noise
    #############################################################
    summary['timestamps'].append(('noise', time.time()))
    am, summary['noise'] = sdl.noise.take_noise(S, cfg, 30,
                                                show_plot=show_plots,
                                                save_plot=True)
    summary['noise']['am'] = am
    sdl.pub_ocs_data(S, {'noise_summary': {
        'band_medians': summary['noise']['noisedict']['band_medians']
    }})

    summary['timestamps'].append(('end', time.time()))

    return True, summary
