import numpy as np
import time
import sodetlib as sdl
from sodetlib.operations import tracking

import matplotlib.pyplot as plt


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
        Name of amplifier. Must be one of ['hemt', 'hemt1', 'hemt2', '50k', '50k1', '50k2'].
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
    all_amps = S.C.list_of_c02_and_c04_amps
    if amp_name not in all_amps:
        raise ValueError(f"amp_name must be one of {all_amps}")

    for _ in range(max_iter):
        amp_biases = S.get_amplifier_biases()
        Vg = amp_biases[f"{amp_name}_gate_volt"]
        Id = amp_biases[f"{amp_name}_drain_current"]
        delta = target_Id - Id

        S.log(delta)
        # Check if Id is within tolerance
        if np.abs(delta) < id_tolerance:
            return True

        if 'hemt' in amp_name:
            step = np.sign(delta) * (0.1 if np.abs(delta) > 1.5 else 0.01)
        else:
            step = np.sign(delta) * (0.01 if np.abs(delta) > 1.5 else 0.001)

        Vg_next = Vg + step
        if not (vg_min < Vg_next < vg_max):
            S.log(f"Gate voltage adjustment would go out of range ({vg_min}, {vg_max}). "
                  f"Unable to change {amp_name}_drain_current to desired value", False)
            return False

        if 'hemt' in amp_name:
            S.set_amp_gate_voltage(amp_name, Vg_next, override=True)
        else:
            S.set_amp_gate_voltage(amp_name, Vg_next, override=True)

        time.sleep(wait_time)

    S.log(f"Max allowed Vg iterations ({max_iter}) has been reached. "
          f"Unable to get target drain current for {amp_name}.", False)

    return False

@sdl.set_action()
def setup_amps(S, cfg, update_cfg=True, enable_300K_LNA=True):
    """
    Initial setup for 50k and hemt amplifiers. For C04/C05 cryocards, will first
    check if the drain voltages are set. Then checks if drain
    currents are in range, and if not will scan gate voltage to find one that
    hits the target current. Will update the device cfg if successful.

    The following parameters can be modified in the device cfg, where {amp}
    is one of ['hemt', 'hemt1', 'hemt2', '50k', '50k1', '50k2']:

        exp:
         - amp_enable_wait_time (float): Seconds to wait after enabling amps
           before scanning gate voltages
         - amp_{amp}_drain_current (float): Target drain current (mA)
         - amp_{amp}_drain_current_tolerance (float): Tolerance for drain current (mA)
         - amp_{amp}_drain_volt (float) : Drain voltage (V). C04/C05 cryocards only.

    Args
    -----
    S : SmurfControl
        Pysmurf instance
    cfg : DetConfig
        DetConfig instance
    update_cfg : bool
        If true, will update the device cfg and save the file.
    enable_300K_LNA:
        If true, will turn on the 300K LNAs.
    """
    sdl.pub_ocs_log(S, "Starting setup_amps")

    exp = cfg.dev.exp

    # Determine cryocard rev
    major, minor, patch = S.C.get_fw_version()
    if major == 4:
        cc_rev = 'c04'
        amp_list = S.C.list_of_c04_amps
    elif major == 0:
        if sum((major,minor,patch)) == 0:
               raise ValueError("Error communicatin with cryocard; "
                                + "is it connected?")
        else:
               raise ValueError(f"Unrecognized cryocard firmware version "
                                + "({major},{minor},{patch}).")
    else:
        cc_rev = 'c02'
        amp_list = S.C.list_of_c02_amps
        S.C.write_ps_en(0b11)
        time.sleep(exp['amp_enable_wait_time'])
        if S.C.read_ps_en() != 3:
            raise ValueError("Could not enable amps.")

    amp_list = list(set(amp_list) & set(exp['amps_to_bias']))
    if len(amp_list) == 0:
        raise ValueError(
            f"exp['amps_to_bias']: {exp['amps_to_bias']} contains no valid amps."
        )
    # Data to be passed back to the ocs-pysmurf-controller clients
    summary = {'success': False}
    for amp in amp_list:
        summary[f'{amp}_gate_volt'] = None
        summary[f'{amp}_drain_current'] = None
        summary[amp + '_enable'] = None
        if cc_rev == 'c04':
            summary[f'{amp}_drain_volt'] = None

    amp_biases = S.get_amplifier_biases()

    # For C04, first check drain voltages
    if cc_rev == 'c04':
        for amp in amp_list:
            Vd = amp_biases[f"{amp}_drain_volt"]
            if Vd != exp[f"amp_{amp}_drain_volt"]:
                S.set_amp_drain_voltage(amp, exp[f"amp_{amp}_drain_volt"]) 

    # Check drain currents / scan gate voltages
    delta_drain_currents = dict()
    for amp in amp_list:
        delta_Id = np.abs(amp_biases[f"{amp}_drain_current"] - exp[f"amp_{amp}_drain_current"])
        if delta_Id > exp[f'amp_{amp}_drain_current_tolerance']:
            S.log(f"{amp} current not within tolerance, scanning for correct gate voltage")
            S.set_amp_gate_voltage(amp, exp[f'amp_{amp}_init_gate_volt'],override=True)
            success = find_gate_voltage(
                S, exp[f"amp_{amp}_drain_current"], amp, wait_time=exp['amp_step_wait_time'],
                id_tolerance=exp[f'amp_{amp}_drain_current_tolerance']
            )
            if not success:
                sdl.pub_ocs_log(S, f"Failed determining {amp} gate voltage")
                sdl.set_session_data(S, 'setup_amps_summary', summary)
                S.C.write_ps_en(0)
                return False, summary
    # Turn on 300K LNAs
    if enable_300K_LNA:
        S.C.write_optical(0b11)
        
    # Update device cfg
    biases = S.get_amplifier_biases()
    if update_cfg:
        for amp in amp_list:
            cfg.dev.update_experiment({f'amp_{amp}_gate_volt': biases[f'{amp}_gate_volt']},
                                      update_file=True)

    summary = {'success': True, **biases}
    sdl.set_session_data(S, 'setup_amps_summary', summary)
    return True, summary


@sdl.set_action()
def setup_phase_delay(S, cfg, bands, update_cfg=True):
    """
    Runs estimate phase delay and updates the device cfg with the results.
    This will run with the current set of attenuation values.

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

    summary = {
        'bands': [],
        'band_delay_us': [],
    }
    for b in bands:
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

    sdl.set_session_data(S, 'setup_phase_delay', summary)
    return True, summary


def estimate_uc_dc_atten(S, cfg, band, update_cfg=True, tone_power=None):
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

    The following parameters can be modified in the device cfg:

        bands:
         - tone_power (int): Tone power to use for atten estimation

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
    sbs = np.arange(250, 255)
    nread = 2

    success = False
    S.log(f"Estimating attens for band {band}")

    if tone_power is None:
        tone_power = cfg.dev.bands[band]['tone_power']

    while True:
        S.set_att_uc(band, att)
        S.set_att_dc(band, att)

        S.log(f'tone_power: {tone_power}')
        _, resp = S.full_band_ampl_sweep(band, sbs, cfg.dev.bands[band]['tone_power'], nread)
        max_resp = np.max(np.abs(resp))
        S.log(f"att: {att}, max_resp: {max_resp}")

        if resp_range[0] < max_resp < resp_range[1]:
            S.log(f"Estimated atten: {att}")
            success = True
            break

        if max_resp < resp_range[0]:
            if att == 0:
                S.log(f"Cannot achieve resp in range {resp_range}! Try increasing tone power")
                success = False
                break
            att -= step

        elif max_resp > resp_range[1]:
            if att == 30:
                S.log(f"Cannot achieve resp in range {resp_range}! Try decreasing tone power")
                success = False
                break
            att += step
        step = int(np.ceil(step / 2))

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

    The following parameters can be modified in the device cfg:

        exp:
         - res_amp_cut (float): Amplitude cut for peak-finding in find-freq
         - res_grad_cut (float): Gradient cut for peak-finding in find-freq
        bands:
         - tone_power (int): Tone power to use for atten estimation

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


def _find_fixed_tone_freq(S, band, ntone, min_spacing, band_start=-250, band_end=250):
    """
    Place fixed tones in the largest gaps between resonators.

    Args
    ----
    S : SmurfControl
        Pysmurf instance
    band : int
        The band to populate with fixed tones.
    ntone : int
        The number of fixed tones to attempt to place in this band.
    min_spacing : float
        The minimum spacing in MHz between resonators to place a fixed tone in.
    band_start : float
        The lower edge of the band in MHz relative to band center.
    band_end : float
        The upper edge of the band in MHz relative to band center.
    """

    # get resonator and channel frequencies
    res_freq = np.sort([res["freq"] for res in S.freq_resp[band]["resonances"].values()])

    # only consider central part of the band
    band_center = S.get_band_center_mhz(band)
    freq_low = band_center + band_start
    freq_high = band_center + band_end
    res_freq = res_freq[(res_freq > freq_low) & (res_freq < freq_high)]

    # gaps between resonators
    gaps = np.diff(res_freq)

    # check there are enough for the number of tones
    num_gaps = (gaps >= min_spacing).sum()
    if num_gaps < ntone:
        S.log(
            f"Only {num_gaps} gaps greater than {min_spacing} MHz were found in band {band}. "
            "That number of fixed tones will be set."
        )
        ntone = num_gaps

    # return the central frequencies in selected gaps
    gap_ind = np.argsort(gaps)[:-(ntone + 1):-1]  # ntone largest gaps
    return 0.5 * (res_freq[gap_ind + 1] + res_freq[gap_ind])


@sdl.set_action()
def setup_fixed_tones(
    S, cfg, bands=None, fixed_tones_per_band=8, min_gap=10, tone_power=10, update_cfg=True
    ):
    """
    Identify gaps between resonators and attempt to place a number of fixed tones there.
    These will be saved for every band in the device config. A tune should be available
    before running this action.

    Args
    ----
    S : SmurfControl
        Pysmurf instance
    cfg : DetConfig
        DetConfig instance
    bands : list of int
        The bands to set fixed tones in. Get from config by default.
    fixed_tones_per_band : int
        The number of tones to place in each band. Defaults to 8.
    min_gap : float
        The minimum gap size in order to place a fixed tone, in MHz. Defaults to 10.
    tone_power : int
        The tone power to use for the fixed tones. Defaults to 10.
    update_cfg : bool
        Whether to save the fixed tone configuration to the device config. Defaults to True.

    Returns
    -------
    fixed_tones : dict
        Fixed tones configuration, as saved to device config.
    """

    if bands is None:
        bands = cfg.dev.exp['active_bands']

    # ensure we have a tune available
    try:
        _ = S.freq_resp[bands[0]]["resonances"]
    except KeyError as e:
        raise ValueError("No tune is available to set fixed tones from.") from e

    # identify available frequencies
    ft_freq = []
    for b in bands:
        ft_freq += list(_find_fixed_tone_freq(S, b, fixed_tones_per_band, min_gap))

        # turn off any existing fixed tones
        try:
            channels = cfg.dev.bands[b]["fixed_tones"]["channels"]
            amp = S.get_amplitude_scale_array(b)
            amp[channels] = 0.0
            S.set_amplitude_scale_array(b, amp)
            S.log(f"Turned off current fixed tones on band {b}.") 
        except KeyError:
            pass

    # set the fixed tones
    tunefile = cfg.dev.exp.get("tunefile", None)
    fixed_tones = {
        b: {
            "enabled": False,
            "freq_offsets": [],
            "channels": [],
            "tone_power": tone_power,
            "tunefile": tunefile,
        }
        for b in bands
    }
    for fr in ft_freq:
        # this function sets the tone power and disables feedback
        try:
            band, channel = S.set_fixed_tone(fr, tone_power=tone_power)
        except AssertionError as e:
            S.log(f"Failed to assign a fixed tone on frequency {fr} MHz.\n{e}.")
            continue
        # read back the assigned frequency offset
        foff = S.get_center_frequency_mhz_channel(band, channel)
        fixed_tones[band]["freq_offsets"].append(float(foff))
        fixed_tones[band]["channels"].append(int(channel))
        fixed_tones[band]["enabled"] = True

    # update config
    if update_cfg:
        for band, data in fixed_tones.items():
            cfg.dev.update_band(band, {"fixed_tones": data}, update_file=True)

    S.log("Done setting fixed tones.")

    return fixed_tones


@sdl.set_action()
def turn_off_fixed_tones(S, cfg, bands=None):
    """
    Read fixed tones from the device config and set their amplitude to 0.

    Args
    ----
    S : SmurfControl
        Pysmurf instance
    cfg : DetConfig
        DetConfig instance
    bands : list of int
        The bands to operate on. Get from config by default.
    """

    if bands is None:
        bands = cfg.dev.exp['active_bands']

    for band in bands:
        # try to read from config
        try:
            if not cfg.dev.bands[band]["fixed_tones"].get("enabled", True):
                S.log(f"Fixed tones are already disabled on band {band}.")
                continue
            channels = cfg.dev.bands[band]["fixed_tones"]["channels"]
        except KeyError as e:
            raise ValueError(
                "No fixed tones present in device config."
            ) from e

        amp = S.get_amplitude_scale_array(band)
        amp[channels] = 0
        S.set_amplitude_scale_array(band, amp)
        cfg.dev.bands[band]["fixed_tones"]["enabled"] = False


@sdl.set_action()
def turn_on_fixed_tones(S, cfg, bands=None):
    """
    Read fixed tones from the device config and ensure the corresponding channels
    are set to the specified amplitude and have feedback disabled.

    Args
    ----
    S : SmurfControl
        Pysmurf instance
    cfg : DetConfig
        DetConfig instance
    bands : list of int
        The bands to operate on. Get from config by default.
    """

    if bands is None:
        bands = cfg.dev.exp['active_bands']

    for band in bands:
        # try to read from config
        try:
            if not cfg.dev.bands[band]["fixed_tones"].get("enabled", False):
                S.log(f"Fixed tones are not enabled on band {band}.")
                continue
            foff = cfg.dev.bands[band]["fixed_tones"]["freq_offsets"]
            channels = cfg.dev.bands[band]["fixed_tones"]["channels"]
            tone_power = cfg.dev.bands[band]["fixed_tones"].get("tone_power", None)
            tunefile = cfg.dev.bands[band]["fixed_tones"].get("tunefile", None)
        except KeyError as e:
            raise ValueError(
                "No fixed tones present in device config. Use `setup_fixed_tones` to add some."
            ) from e
        if len(foff) != len(channels):
            raise ValueError(
                "Number of fixed tone channels in device config is different from number of "
                f"frequency offsets for band {band} ({len(channels)} != {len(foff)})."
            )
        cfg_tunefile = cfg.dev.exp.get("tunefile", None)
        if cfg_tunefile is None or tunefile is None:
            # warn but do nothing
            S.log(
                "Turning on fixed tones with no associated tunefile could lead "
                "to conflicts with resonator channels.", S.LOG_ERROR
            )
        elif tunefile != cfg_tunefile:
            raise ValueError(
                "The tunefile has changed since fixed tones were set up. Run `setup_fixed_tones "
                "to find available channels."
            )
        S.log(f"Enabling {len(channels)} fixed tones on band {band}.")

        # set the channel center frequency for each tone
        for df, chan in zip(foff, channels):
            S.set_center_frequency_mhz_channel(band, chan, df)

        # set amplitude and feedback
        amp = S.get_amplitude_scale_array(band)
        feedback = S.get_feedback_enable_array(band)

        amp[channels] = tone_power
        feedback[channels] = 0

        S.set_amplitude_scale_array(band, amp)
        S.set_feedback_enable_array(band, feedback)


@sdl.set_action()
def uxm_setup(S, cfg, bands=None, show_plots=True, update_cfg=True,
              modify_attens=True, skip_phase_delay=False,
              skip_setup_amps=False): 
    """
    The goal of this function is to do a pysmurf setup completely from scratch,
    meaning no parameters will be pulled from the device cfg.

    The following steps will be run:

        1. setup amps
        2. Estimate phase delay
        3. Setup tune
        4. setup tracking
        5. Measure noise

    The following device cfg parameters can be changed to modify behavior:

        exp:
         - downsample_factor (int): Downsample factor to use
         - coupling_mode (str): Determines whether to run in DC or AC mode. Can
           be 'dc' or 'ac'.
         - synthesis_scale (int): Synthesis scale to use
         - amp_enable_wait_time (float): Seconds to wait after enabling amps
           before scanning gate voltages
         - amp_hemt_Id (float): Target drain current for hemt amp (mA)
         - amp_50k_Id (float): Target drain current for 50k amp (mA)
         - amp_hemt_Id_tolerance (float): Tolerance for hemt drain current (mA)
         - amp_50k_Id_tolerance (float): Tolerance for 50k drain current (mA)
         - res_amp_cut (float): Amplitude cut for peak-finding in find-freq
         - res_grad_cut (float): Gradient cut for peak-finding in find-freq
        bands:
         - tone_power (int): Tone power to use for atten estimation

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
    modify_attens : bool
        If true or attenuations are not set in the device config, will run
        estimate_uc_dc_atten to find a set of attenuations that will work for
        estimate_phase_delay and find_freq. If False and attenuation values
        already exist in the device config, this will use those values.
    skip_phase_delay : bool
        If True, will skip the estimate_phase_delay step
    skip_setup_amps : bool
        If True, will skip amplifier setup
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
        S.set_synthesis_scale(band, exp['synthesis_scale'])

    summary = {}
    summary['timestamps'] = []

    #############################################################
    # 2. Setup amps
    #############################################################
    if not skip_setup_amps:
        summary['timestamps'].append(('setup_amps', time.time()))
        sdl.set_session_data(S, 'timestamps', summary['timestamps'])

        success, summary['setup_amps'] = setup_amps(S, cfg, update_cfg=update_cfg)
        if not success:
            sdl.pub_ocs_log(S, "UXM Setup failed on setup amps step")
            return False, summary

    #############################################################
    # 3. Estimate Attens
    #############################################################
    summary['timestamps'].append(('estimate_attens', time.time()))
    sdl.set_session_data(S, 'timestamps', summary['timestamps'])
    for band in bands:
        bcfg = cfg.dev.bands[band]
        if modify_attens or (bcfg['uc_att'] is None) or (bcfg['dc_att'] is None):
            success = estimate_uc_dc_atten(S, cfg, band, update_cfg=update_cfg)
            if not success:
                sdl.pub_ocs_log(S, f"Failed to estimate attens on band {band}")
                return False, summary
        else:
            S.set_att_uc(band, bcfg['uc_att'])
            S.set_att_dc(band, bcfg['dc_att'])

    #############################################################
    # 4. Estimate Phase Delay
    #############################################################
    if not skip_phase_delay:
        summary['timestamps'].append(('setup_phase_delay', time.time()))
        sdl.set_session_data(S, 'timestamps', summary['timestamps'])
        success, summary['setup_phase_delay'] = setup_phase_delay(
            S, cfg, bands, update_cfg=update_cfg)
        if not success:
            S.log("UXM Setup failed on setup phase delay step")
            return False, summary

    #############################################################
    # 5. Setup Tune
    #############################################################
    summary['timestamps'].append(('setup_tune', time.time()))
    sdl.set_session_data(S, 'timestamps', summary['timestamps'])
    success, summary['setup_tune'] = setup_tune(
        S, cfg, bands, show_plots=show_plots, update_cfg=update_cfg,)
    if not success:
        S.log("UXM Setup failed on setup tune step")
        return False, summary

    #############################################################
    # 6. Setup Tracking
    #############################################################
    summary['timestamps'].append(('setup_tracking', time.time()))
    sdl.set_session_data(S, 'timestamps', summary['timestamps'])

    # Ensure feedback is enabled
    for b in bands:
        S.set_feedback_enable(b, 1)

    tracking_res = tracking.setup_tracking_params(
        S, cfg, bands, show_plots=show_plots, update_cfg=update_cfg
    )
    summary['tracking_res'] = tracking_res

    #############################################################
    # 7. Noise
    #############################################################
    summary['timestamps'].append(('noise', time.time()))
    sdl.set_session_data(S, 'timestamps', summary['timestamps'])
    _, summary['noise'] = sdl.noise.take_noise(
        S, cfg, 30, show_plot=show_plots, save_plot=True
    )

    summary['timestamps'].append(('end', time.time()))
    sdl.set_session_data(S, 'timestamps', summary['timestamps'])

    return True, summary
