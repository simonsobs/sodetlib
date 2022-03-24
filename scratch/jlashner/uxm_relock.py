import time
import numpy as np
import sodetlib as sdl


sdl.set_action()
def reload_amps(S, cfg, id_tolerance=0.5):
    """
    Reloads amplifier biases from dev cfg and checks that drain-currents fall
    within tolerance.

    Args
    ----
    S : SmurfControl
        Pysmurf instance
    cfg : DetConfig
        Det config instance
    id_tolerance : float
        Max difference between target drain current and actual
        drain currents for this to be considered success (mA).
        Defaults to 0.5 mA.
    """
    summary = {}

    sdl.pub_ocs_log(S, 'setting amp voltage')
    S.set_50k_amp_gate_voltage(cfg.dev.exp['amp_50k_Vg'])
    S.set_hemt_gate_voltage(cfg.dev.exp['amp_hemt_Vg'])
    S.C.write_ps_en(3)
    time.sleep(0.1)
    biases = S.get_amplifier_biases()
    summary['biases'] = biases

    exp = cfg.dev.exp

    in_range_50k = np.abs(biases['50k_Id'] - exp['amp_50k_Id']) < id_tolerance
    in_range_hemt = np.abs(biases['hemt_Id'] - exp['amp_hemt_Id']) < id_tolerance

    if not (in_range_50k and in_range_hemt):
        S.log("Hemt or 50K Amp drain current not within tolerance")
        S.log(f"Target hemt Id: {exp['amp_hemt_Id']}")
        S.log(f"Target 50K Id: {exp['amp_50k_Id']}")
        S.log(f"tolerance: {id_tolerance}")
        S.log(f"biases: {biases}")

        summary['success'] = False
        sdl.pub_ocs_log(S, 'Failed to set amp voltages')
        sdl.pub_ocs_data(S, {'amp_summary': summary})
        return False, summary

    summary['success'] = True 
    sdl.pub_ocs_log(S, 'Succuessfully to set amp voltages')
    sdl.pub_ocs_data(S, {'amp_summary': summary})
    return True, summary


sdl.set_action()
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
        S.relock(band)
        if setup_notches:
            S.setup_notches(band, new_master_assignment=new_master_assignment)

        S.run_serial_gradient_descent(band)
        S.run_serial_eta_scan(band)

    return True, None


sdl.set_action()
def relock_tracking_setup(S, cfg, bands, reset_rate_khz=None, nphi0=None,
                          **kwargs):
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

    Returns
    --------
    res : dict
        Dictionary of results of all tracking-setup calls, with the bands number
        as key.
    """
    bands = np.atleast_1d(bands)
    nbands = len(bands)

    # Arrays containing the optimized tracking parameters for each band
    frac_pp0 = np.zeros(nbands)
    lms_freq0 = np.zeros(nbands)  # Hz
    reset_rate_khz0 = None  # This is assumed to be the same for each band

    for i, b in enumerate(bands):
        bcfg = cfg.dev.bands[b]
        frac_pp0[i] = bcfg['frac_pp']
        lms_freq0[i] = bcfg['lms_freq_hz'] 
        if reset_rate_khz0 is None:
            reset_rate_khz0 = bcfg['flux_ramp_rate_khz']

    # Nphi0 used during optimization (assumes this is the same for each band)
    init_nphi0 = np.round(lms_freq0[0] / reset_rate_khz0)

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
    res = {}
    tk = sdl.get_tracking_kwargs(S, cfg, bands[0], kwargs=kwargs)
    tk['reset_rate_khz'] = reset_rate_khz
    tk['fraction_full_scale'] = fpp
    for i, b in enumerate(bands):
        tk.update({'lms_freq_hz': lms_freqs[i]})
        res[b] = S.tracking_setup(b, **tk)

    return True, res


sdl.set_action()
def uxm_relock(S, cfg, bands=None, id_tolerance=0.5,
               new_master_assignment=False, setup_notches=True,
               show_plots=False):
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
    id_tolerance : float
        Tolerance in amp drain current
    new_master_assignment : bool
        Whether to run setup_notches with new_master_assignment.
        Defaults to False

    Returns
    --------
    summary : dict
        Dictionary containing a summary of all run operations.
    """
    if bands is None:
        bands = np.arange(8)
    bands = np.atleast_1d(bands)

    summary = {}

    # 1. Reset system to known state
    S.all_off()  # Turn off Flux ramp, tones, and biases
    S.set_rtm_arb_waveform_enable(0)
    S.set_filter_disable(0)
    S.set_downsample_factor(20)
    S.set_mode_dc()

    for band in bands:
        band_cfg = cfg.dev.bands[band]
        S.set_att_uc(band, band_cfg['uc_att'])
        S.set_att_dc(band, band_cfg['dc_att'])
        S.set_synthesis_scale(band, cfg.dev.exp['synthesis_scale'])
        S.amplitude_scale[band] = band_cfg['tone_power']

    # 2. set amplifiers
    success, summary['reload_amps'] = reload_amps(S, cfg)
    if not success:
        return False, summary

    # 3. load tune
    success, summary['reload_tune'] = reload_tune(
        S, cfg, bands, setup_notches=setup_notches,
        new_master_assignment=new_master_assignment)
    if not success:
        return False, summary

    success, summary['tracking_setup'] = relock_tracking_setup(
        S, cfg, bands, show_plot=show_plots)
    if not success:
        return False, summary

    _, summary['noise'] = sdl.noise.take_noise(S, cfg, 30,
                                                show_plot=show_plots,
                                                save_plot=True)
    sdl.pub_ocs_data(S, {'noise_summary': {
        'band_medians': summary['noise']['noisedict']['band_medians']
    }})

    return True, summary
