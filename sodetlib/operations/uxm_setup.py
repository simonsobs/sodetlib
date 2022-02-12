import numpy as np
import time
import sodetlib as sdl


def find_gate_voltage(S, target_Id, amp_name, vg_min=-2.0, vg_max=0,
                      max_iter=50, wait_time=0.5, id_tolerance=0.2):
    """
    Scans through bias voltage for hemt or 50K amplifier to get the correct
    gate voltage for a target current.

    Args
    -----
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
    wait_time : float
        Time to wait after setting the voltage at each step. Defaults
        to 0.5 sec
    id_tolerance : float
        Max difference between target drain current and actual drain currents
        for this to be considered success (mA). Defaults to 0.2 mA.

    Returns
    --------
    success : bool
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

        # Check if Id is within tolerance
        if np.abs(delta)/target_Id < id_tolerance:
            return True

        if amp_name == 'hemt':
            step = np.sign(delta) * (0.1 if np.abs(delta) > 1.5 else 0.01)
        else:
            step = np.sign(delta) * (0.01 if np.abs(delta) > 1.5 else 0.001)

        Vg_next = Vg + step
        if not (vg_min < Vg_next < vg_max):
            S.log(f"Vg adjustment would go out of range ({vg_min}, {vg_max}). "
                  f"Unable to change {amp_name}_Id to desired value", False)
            return False

        if amp_name == 'hemt':
            S.set_hemt_gate_voltage(Vg_next, override=True)
        else:
            S.set_50k_amp_gate_voltage(Vg_next, override=True)

        time.sleep(wait_time)

    S.log(f"Max allowed Vg iterations ({max_iter}) has been reached. "
          f"Unable to get target Id for {amp_name}.", False)

    return False


@sdl.set_action()
def setup_amps(S, cfg, update_cfg=True):
    """
    Initial setup for 50k and hemt amplifiers. Determines gate voltages
    required to reach specified drain currents. Will update detconfig on
    success.

    Args
    -----
    S : SmurfControl
        Pysmurf instance
    cfg : DetConfig
        DetConfig instance
    amp_hemt_Id : float
        Target hemt drain current (mA)
    amp_50k_Id : float
        Target 50k drain current (mA)
    vgmin_hemt : float
        Min hemt gate voltage (V)
    vgmin_50k : float
        Min 50k gate voltage (V)
    update_cfg : bool
        If true, will update the device cfg and save the file.
    wait_time : float
        Time to wait after setting the voltage at each step. Defaults
        to 0.5 sec
    id_tolerance : float, tuple
        Max difference between target drain current and actual drain currents
        for this to be considered success (mA). Defaults to 0.2 mA. If a tuple
        is passed, first element corresponds to HEMt amp, and second with
        50k.
    """
    sdl.pub_ocs_log(S, "Starting setup_amps")

    exp = cfg.dev.exp

    S.set_50k_amp_gate_voltage(exp['amp_50k_init_Vg'])
    S.set_hemt_gate_voltage(exp['amp_hemt_init_Vg'])
    S.C.write_ps_en(0b11)
    time.sleep(exp['amp_enable_wait_time'])

    # Data to be passed back to the ocs-pysmurf-controller clients
    summary = {
        'success':     False,
        'amp_50k_Id':  None,
        'amp_hemt_Id': None,
        'amp_50k_Vg':  None,
        'amp_hemt_Vg': None,
    }

    success = find_gate_voltage(
        S, exp['amp_hemt_Id'], 'hemt', wait_time=exp['amp_step_wait_time'],
        id_tolerance=exp['amp_hemt_Id_tolerance'])
    if not success:
        sdl.pub_ocs_log(S, "Failed determining hemt gate voltage")
        sdl.pub_ocs_data(S, {'setup_amps_summary': summary})
        S.C.write_ps_en(0)
        return False, summary

    success = find_gate_voltage(
        S, exp['amp_50k_Id'], '50k', wait_time=exp['amp_step_wait_time'],
        id_tolerance=exp['amp_50k_Id_tolerance'])
    if not success:
        sdl.pub_ocs_log(S, "Failed determining 50k gate voltage")
        sdl.pub_ocs_data(S, {'setup_amps_summary': summary})
        S.C.write_ps_en(0)
        return False, summary

    # Update device cfg
    biases = S.get_amplifier_biases()
    if update_cfg:
        cfg.dev.update_experiment({
            'amp_50k_Vg':  biases['50K_Vg'],
            'amp_hemt_Vg': biases['hemt_Vg'],
        }, update_file=True)

    summary = {'success': True, **biases}
    sdl.pub_ocs_data(S, {'setup_amps_summary': summary})
    return True, summary


@sdl.set_action()
def setup_phase_delay(S, cfg, bands, update_cfg=True, modify_attens=True):
    """
    Sets uc and dc attens to reasonable values and runs estimate phase delay
    for desired bands.

    Args
    -----
    S : SmurfControl
        Pysmurf instance
    cfg : DetConfig
        DetConfig instance
    uc_att : int
        UC atten to use for phase-delay estimation
    dc_att : int
        DC atten to use for phase-delay estimation
    update_cfg : bool
        If true, will update the device cfg and save the file.
    """
    sdl.pub_ocs_log(S, f"Estimating phase delay for bands {bands}")

    exp = cfg.dev.exp

    summary = {
        'bands': [],
        'band_delay_us': [],
    }
    for b in bands:
        if modify_attens:
            S.set_att_dc(b, exp['phase_delay_dc_att'])
            S.set_att_uc(b, exp['phase_delay_uc_att'])
        summary['bands'].append(int(b))
        band_delay_us, _ = S.estimate_phase_delay(b, make_plot=True, show_plot=False)
        band_delay_us = float(band_delay_us)
        summary['band_delay_us'].append(band_delay_us)
        if update_cfg:
            cfg.dev.bands[b].update({
                'band_delay_us': band_delay_us
            })

    if update_cfg:
        cfg.dev.update_file()

    sdl.pub_ocs_data(S, {'setup_phase_delay': summary})
    return True, summary


def estimate_uc_dc_atten(S, cfg, band, update_cfg=True):
    """
    Provides an initial estimation of uc and dc attenuation for a band in order
    to get the band response at a particular frequency to be within a certain
    range. The goal of this function is to choose attenuations such that the
    ADC isn't saturated, and the response is large enough to find resonators.

    For simplicity, this searches over the parameter space where uc and dc
    attens are equal, instead of searching over the full 2d space.

    For further optimization of attens with respect to median white noise, run
    the ``optimize_attens`` function from the
    ``sodetlib/operations/optimize.py``
    module.

    Args
    -----
    S : SmurfControl
        Pysmurf instance
    band : int
        Band to estimate attens for
    update_cfg : bool
        If true, will update the device cfg and save the file.  """
    att, step = 15, 7
    sb = S.get_closest_subband(-200, band)

    resp_range = (0.1, 0.4)
    # Just look at 5 subbands after specified freq.
    sbs = np.arange(sb, sb + 5)
    nread = 2

    success = False
    S.log(f"Estimating attens for band {band}")
    while True:
        S.set_att_uc(band, att)
        S.set_att_dc(band, att)

        _, resp = S.full_band_ampl_sweep(band, sbs, cfg.dev.exp['tone_power'], nread)
        max_resp = np.max(np.abs(resp))
        S.log(f"att: {att}, max_resp: {max_resp}")

        if resp_range[0] < max_resp < resp_range[1]:
            S.log(f"Estimated atten: {att}")
            success = True
            break

        if step == 0:
            S.log(f"Cannot achieve resp in range {resp_range}!")
            success = False
            break

        if max_resp < resp_range[0]:
            att -= step
            step = step // 2
        elif max_resp > resp_range[1]:
            att += step
            step = step // 2

    if success and update_cfg:
        cfg.dev.update_band(band, {
            'uc_att': att,
            'dc_att': att,
        }, update_file=True)

    return success


@sdl.set_action()
def setup_tune(S, cfg, bands, show_plots=False, update_cfg=True):
    """
    Find freq, setup notches, and serial gradient descent and eta scan

    Args
    -----
    S : SmurfControl
        Pysmurf instance
    cfg : DetConfig
        DetConfig instance
    tone_power : int, optional
        Tone power to use. Defaults to what exists in the pysmurf-cfg file.
    show_plots : bool
        If true, will show find_freq plots. Defaults to False
    amp_cut : float
        Amplitude cut for peak-finding in find_freq
    grad_cut : float
        Gradient cut for peak-finding in find_freq
    estimate_attens : bool
        If True, will try to find reasonable uc / dc attens for
        each band before running fund_freq. See the
        ``estimate_uc_dc_atten`` function for more details.
    update_cfg : bool
        If true, will update the device cfg and save the file.
    """
    bands = np.atleast_1d(bands)
    sdl.pub_ocs_log(S, f"Starting setup_tune for bands {bands}")

    exp = cfg.dev.exp

    summary = {}

    for band in bands:
        sdl.pub_ocs_log(S, f"Find Freq: band {band}")
        bcfg = cfg.dev.bands[band]
        S.find_freq(band, tone_power=bcfg['tone_power'], make_plot=True,
                    save_plot=True, show_plot=show_plots,
                    amp_cut=exp['res_amp_cut'], 
                    grad_cut=exp['res_grad_cut'])

    for band in bands:
        bcfg = cfg.dev.bands[band]
        sdl.pub_ocs_log(S, f"Setup Notches: band {band}")
        S.setup_notches(band, tone_power=bcfg['tone_power'], new_master_assignment=True)

    for band in bands:
        sdl.pub_ocs_log(S, f"Serial grad descent and eta scan: band {band}")
        S.run_serial_gradient_descent(band)
        S.run_serial_eta_scan(band)

    if update_cfg:
        cfg.dev.update_experiment({'tunefile': S.tune_file}, update_file=True)

    return True, summary


def setup_tracking_params(S, cfg, bands, update_cfg=True, show_plots=False,
                          disable_bad_chans=True):
    """
    Setups up tracking parameters by determining correct frac-pp and lms-freq
    for each band.

    Args
    -----
    S : SmurfControl
        Pysmurf instance
    cfg : DetConfig
        DetConfig instance
    bands : np.ndarray, int
        Band or list of bands to run on
    init_fracpp : float, optional
        Initial frac-pp value to use for tracking estimates
    nphi0 : int
        Number of phi0 periods to track on
    reset_rate_khz : float
        Flux ramp reset rate in khz
    update_cfg : bool
        If true, will update the device cfg and save the file.
    """

    bands = np.atleast_1d(bands)

    exp = cfg.dev.exp
    summary = {
        'success': None,
        'tracking_results': {}
    }
    success = True
    for band in bands:
        sdl.pub_ocs_log(S, f"Setting up trackng params: band {band}")
        tk = sdl.get_tracking_kwargs(
            S, cfg, band, kwargs={
                'lms_freq_hz':         None,
                'show_plot':           False,
                'meas_lms_freq':       True,
                'fraction_full_scale': exp['init_frac_pp'],
                'reset_rate_khz':      exp['flux_ramp_rate_khz'],
                'lms_gain': exp['lms_gain'],
                'feedback_gain': exp['feedback_gain'],
            }
        )
        f, df, sync = S.tracking_setup(band, **tk)
        r2 = sdl.compute_tracking_quality(S, f, df, sync)

        # Cut all but good chans to calc fpp / lms-freq
        asa_init = S.get_amplitude_scale_array(band)
        asa_good = asa_init.copy()
        asa_good[r2 < 0.95] = 0
        S.set_amplitude_scale_array(band, asa_good)

        # Calculate trracking parameters
        S.tracking_setup(band, **tk)
        lms_meas = S.lms_freq_hz[band]
        lms_freq = exp['nphi0'] * tk['reset_rate_khz'] * 1e3
        frac_pp = tk['fraction_full_scale'] * lms_freq / lms_meas

        # Re-enables all channels and re-run tracking setup with correct params
        S.set_amplitude_scale_array(band, asa_init)
        tk['meas_lms_freq'] = False
        tk['fraction_full_scale'] = frac_pp
        tk['lms_freq_hz'] = lms_freq
        tk['show_plot'] = show_plots
        f, df, sync = S.tracking_setup(band, **tk)

        # Make cuts based on tracking-quality, and p2p of tracked f and df
        r2 = sdl.compute_tracking_quality(S, f, df, sync)
        f_ptp = np.ptp(f, axis=0)
        df_ptp = np.ptp(df, axis=0)

        f_ptp_range = exp['f_ptp_range']
        df_ptp_range = exp['df_ptp_range']
        good_chans = (r2 > exp['r2_min'])  \
            & (f_ptp_range[0] < f_ptp) & (f_ptp < f_ptp_range[1])  \
            & (df_ptp_range[0] < df_ptp) & (df_ptp < df_ptp_range[1])
        num_good = np.sum(good_chans)
        num_tot = len(good_chans)

        summary['tracking_results'][band] = {
            'f': f, 'df': df, 'sync': sync, 'r2': r2, 'good_chans': good_chans
        }

        if num_good / num_tot < exp['min_good_tracking_frac']:
            S.log(f"Not enough good channels on band {band}!!")
            S.log(f"Something is probably wrong")
            success = False

        if disable_bad_chans:
            asa = S.get_amplitude_scale_array(band)
            asa[~good_chans] = 0
            S.set_amplitude_scale_array(asa)

        # Update det config
        if update_cfg:
            cfg.dev.update_band(band, {
                'frac_pp':            frac_pp,
                'lms_freq_hz':        lms_freq,
            }, update_file=True)

        ## Lets add some cuts on p2p(f) and p2p(df) here. What are good numbers
        ## for that?
        sdl.pub_ocs_data(S, {'setup_tracking_params_summary': summary})

    return success, summary


def uxm_setup(S, cfg, bands=None, show_plots=True, update_cfg=True):
    """
    The goal of this function is to do a pysmurf setup completely from scratch,
    meaning no parameters will be pulled from the device cfg.

    The following steps will be run:

        1. setup amps
        2. Estimate phase delay
        3. Setup tune
        4. setup tracking
        5. Measure noise

    Args
    -----
    S : SmurfControl
        Pysmurf instance
    cfg : DetConfig
        DetConfig instance
    show_plots : bool
        If true, will show find_freq plots. Defaults to False
    update_cfg : bool
        If true, will update the device cfg and save the file.
    modify_attens : 
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
        S.set_synthesis_scale(band, exp['synthesis_scale'])

    summary = {}
    summary['timestamps'] = []
    #############################################################
    # 2. Setup amps
    #############################################################
    summary['timestamps'].append(('setup_amps', time.time()))
    success, summary['setup_amps'] = setup_amps(S, cfg, update_cfg=update_cfg)
    if not success:
        sdl.pub_ocs_log(S, "UXM Setup failed on setup amps step")
        return False, summary

    #############################################################
    # 3. Estimate Attens
    #############################################################
    summary['timestamps'].append(('estimate_attens', time.time()))
    for band in bands:
        bcfg = cfg.dev.bands[band]
        if (bcfg['uc_att'] is None) or (bcfg['dc_att'] is None):
            success = estimate_uc_dc_atten(S, cfg, band, update_cfg=update_cfg)
            if not success:
                sdl.pub_ocs_log(S, f"Failed to estimate attens on band {band}")
                return False, summary

    #############################################################
    # 4. Estimate Phase Delay
    #############################################################
    summary['timestamps'].append(('setup_phase_delay', time.time()))
    success, summary['setup_phase_delay'] = setup_phase_delay(
        S, cfg, bands, update_cfg=update_cfg)
    if not success:
        S.log("UXM Setup failed on setup phase delay step")
        return False, summary

    #############################################################
    # 5. Setup Tune
    #############################################################
    summary['timestamps'].append(('setup_tune', time.time()))
    success, summary['setup_tune'] = setup_tune(
        S, cfg, bands, show_plots=show_plots, update_cfg=update_cfg,)
    if not success:
        S.log("UXM Setup failed on setup tune step")
        return False, summary

    #############################################################
    # 6. Setup Tracking
    #############################################################
    summary['timestamps'].append(('setup_tracking_params', time.time()))
    success, summary['setup_tracking_params'] = setup_tracking_params(
        S, cfg, bands, show_plots=show_plots, update_cfg=update_cfg
    )
    if not success:
        S.log("UXM Setup failed on setup tracking step")
        return False, summary

    #############################################################
    # 7. Noise
    #############################################################
    summary['timestamps'].append(('noise', time.time()))
    _, summary['noise'] = sdl.noise.take_noise(
        S, cfg, 30, show_plot=show_plots, save_plot=True
    )
    sdl.pub_ocs_data(S, {'noise_summary': {
        'band_medians': summary['noise']['noisedict']['band_medians']
    }})

    summary['timestamps'].append(('end', time.time()))

    return True, summary
