"""
det_ops.py is a module containing functions for detector characterization and
operation, such as:
- Tickle and bias step function
- Running IV's
- and more!!
"""
import numpy as np
import time
import os
import sys
from sodetlib.util import make_filename
from tqdm import tqdm
import sodetlib.smurf_funcs.smurf_ops as so
from sodetlib.analysis import det_analysis

from pysmurf.client.util.pub import set_action
from sodetlib.smurf_funcs import smurf_ops as so

def get_current_mode(S, bias_group):
    """
    Returns 1 if requested bias_group is in high_current_mode and 0
    otherwise.
    Args
    ----
    S:
        SmurfControl object
    bias_group : int
        The bias group to query
    """
    relay = S.get_cryo_card_relays()
    relay = S.get_cryo_card_relays()  # querey twice to ensure update
    if bias_group >= S._n_bias_groups:
        raise ValueError("Biasgroup must be between 0 and {S._n_bias_groups}")
    r = np.ravel(S._pic_to_bias_group[np.where(
        S._pic_to_bias_group[:, 1] == bias_group)])[0]
    return (relay >> r) & 1


@set_action()
def take_iv(
    S,
    cfg,
    bias_groups=None,
    overbias_voltage=8.0,
    overbias_wait=2.0,
    high_current_mode=True,
    cool_wait=30,
    cool_voltage=None,
    bias=None,
    bias_high=1.5,
    bias_low=0,
    bias_step=0.005,
    wait_time=0.1,
    do_analysis=True,
    phase_excursion_min=3.0,
    psat_level=0.9,
    make_channel_plots=False,
    make_summary_plots=True,
    save_plots=True,
):
    """
    Replaces the pysmurf run_iv function to be more appropriate for SO-specific
    usage, as well as easier to edit as needed.  Steps the TES bias down
    slowly. Starts at bias_high to bias_low with step size bias_step. Waits
    wait_time between changing steps. After overbiasing, can choose bias point
    to allow system to cool down.

    Args
    ----
    S : 
        SmurfControl object
    cfg :
        DetConfig object
    bias_groups : numpy.ndarray or None, optional, default None
        Which bias groups to take the IV on. If None, defaults to
        the groups in the config file.
    overbias_voltage : float, optional, default 8.0
        The voltage to set the TES bias in the overbias stage.
    overbias_wait : float, optional, default 2.0
        The time to stay in the overbiased state in seconds.
    high_current_mode : bool, optional, default True
        The current mode to take the IV in.
    cool_wait : float, optional, default 30.0
        The time to stay in the low current state after
        overbiasing before taking the IV.
    cool_voltage: float, optional, default None
        The voltage to bias at after overbiasing before taking the IV
        while the system cools.
    bias : float array or None, optional, default None
        A float array of bias values. Must go high to low.
    bias_high : float, optional, default 1.5
        The maximum TES bias in volts.
    bias_low : float, optional, default 0
        The minimum TES bias in volts.
    bias_step : float, optional, default 0.005
        The step size in volts.
    wait_time : float, optional, default 0.1
        The amount of time between changing TES biases in seconds.
    do_analysis: bool, optional, default True
        Whether to do the pysmurf IV analysis
    phase_excursion_min : float, optional, default 3.0
        The minimum phase excursion required for making plots.
    psat_level: float
        Default 0.9. Fraction of R_n to calculate Psat.
        Used if analyzing IV.
    make_channel_plots: bool
        Generates individual channel plots of IV, Rfrac, S_I vs. Rfrac, and RP.
    make_summary_plots: bool
        Make histograms of R_n and Psat.

    Returns
    -------
    output_path : str
        Full path to iv_info npy file.
    """

    # This is the number of bias groups that the cryocard has
    n_bias_groups = S._n_bias_groups
    if bias_groups is None:
        bias_groups = np.arange(12)  # SO UFMs have 12 bias groups
    bias_groups = np.array(bias_groups)
    if overbias_voltage != 0.:
        overbias = True
    else:
        overbias = False
    if bias is None:
        # Set actual bias levels
        bias = np.arange(bias_high, bias_low-bias_step, -bias_step)
    # Overbias the TESs to drive them normal
    if overbias:
        if cool_voltage is None:
            S.overbias_tes_all(bias_groups=bias_groups,
                               overbias_wait=overbias_wait,
                               tes_bias=np.max(bias), cool_wait=cool_wait,
                               high_current_mode=high_current_mode,
                               overbias_voltage=overbias_voltage)
        else:
            S.overbias_tes_all(bias_groups=bias_groups,
                               overbias_wait=overbias_wait,
                               tes_bias=cool_voltage, cool_wait=cool_wait,
                               high_current_mode=high_current_mode,
                               overbias_voltage=overbias_voltage)
            S.log('Starting to take IV.', S.LOG_USER)
    S.log('Starting TES bias ramp.', S.LOG_USER)
    bias_group_bool = np.zeros((n_bias_groups,))
    # only set things on the bias groups that are on
    bias_group_bool[bias_groups] = 1
    S.set_tes_bias_bipolar_array(bias[0] * bias_group_bool)
    time.sleep(wait_time)
    start_time = S.get_timestamp()  # get time the IV starts
    S.log(f'Starting IV at {start_time}')
    sid = so.stream_g3_on(S, make_freq_mask=False)  
    S.log(f'g3 stream id: {sid}')
    for b in bias:
        S.log(f'Bias at {b:4.3f}')
        S.set_tes_bias_bipolar_array(b * bias_group_bool)
        time.sleep(wait_time)
    datfile = S.get_data_file_name(as_string=True)
    sid = so.stream_g3_off(S)
    stop_time = S.get_timestamp()  # get time the IV finishes
    S.log(f'Finishing IV at {stop_time}')
    basename = str(sid)
    path = os.path.join(S.output_dir, basename + '_iv_bias_all')
    np.save(path, bias)
    # publisher announcement
    S.pub.register_file(path, 'iv_bias', format='npy')

    iv_info = {}
    iv_info['plot_dir'] = S.plot_dir
    iv_info['output_dir'] = S.output_dir
    iv_info['tune_file'] = S.tune_file
    iv_info['R_sh'] = S.R_sh
    iv_info['bias_line_resistance'] = S.bias_line_resistance
    iv_info['high_low_ratio'] = S.high_low_current_ratio
    iv_info['pA_per_phi0'] = S.pA_per_phi0
    iv_info['high_current_mode'] = high_current_mode
    iv_info['start_time'] = start_time
    iv_info['stop_time'] = stop_time
    iv_info['basename'] = basename
    iv_info['session id'] = sid
    iv_info['datafile'] = datfile
    iv_info['bias'] = bias
    iv_info['bias group'] = bias_groups
    if cfg.uxm is not None:
        iv_info['wafer_id'] = cfg.uxm.get('wafer_id')
    else:
        iv_info['wafer_id'] = None
    iv_info['version'] = 'v1'

    iv_info_fp = os.path.join(S.output_dir, basename + '_iv_info.npy')

    np.save(iv_info_fp, iv_info)
    S.log(f'Writing IV information to {iv_info_fp}.')
    S.pub.register_file(iv_info_fp, 'iv_info', format='npy')

    if do_analysis:
        timestamp, phase, mask, v_bias = det_analysis.load_from_sid(cfg, iv_info_fp)
        iv_analyze_fp = det_analysis.analyze_iv_and_save(
                S,
                cfg,
                iv_info_fp,
                phase,
                v_bias,
                mask,
                phase_excursion_min=phase_excursion_min,
                psat_level=psat_level,
                outfile=None,
        )

        iv_analyze = np.load(iv_analyze_fp, allow_pickle=True).item()

        if make_channel_plots:
            det_analysis.iv_channel_plots(
                iv_info,
                iv_analyze,
                plot_dir=iv_info["plot_dir"],
                show_plot=False,
                save_plot=save_plots,
                S=S,
            )

        if make_summary_plots:
            det_analysis.iv_summary_plots(
                iv_info,
                iv_analyze,
                plot_dir=iv_info["plot_dir"],
                show_plot=False,
                save_plot=save_plots,
                S=S,
            )

    return iv_info_fp


@set_action()
def take_tickle(S, cfg, bias_groups, tickle_freq=5., tickle_voltage=0.005,
                duration=3., high_current=False, silence_pysmurf=True):
    """
    Takes a tickle measurement on one or more bias groups. If multiple bias
    groups are specified, will play a tickle over each bias group in sequence,
    so we are able to identify which detectors belong to which bias group.

    Args
    ----
    bias_group : (int, list[int])
        bias group or list of bias groups to tickle.
    tickle_freq : float
        Frequency of tickle to play
    tickle_voltage : float
        voltage of tickle
    duration : float
        duration of tickle (sec)
    high_current: bool
        If True, will set to high-current mode and adjust the dc bias voltage
        such that the current is the same if not already in high-current mode.
        Note that running without high-current mode means that there is
        low-pass frequency with a low cutoff that will almost certainly screw
        up the analysis and give you incorrect resistance values.
    silence_pysmurf : bool
        If True, will sent the pysmurf logs to a logfile instead of stdout

    Returns
    -------
    summary : dictionary
        A dict containing a summary of the operation, containing all info
        needed to perform the analysis on either the smurf-server from .dat
        files or simons1 from g3 files.

    """
    if isinstance(bias_groups, (float, int)):
        bias_groups = [bias_groups]

    logs_silenced = False
    logfile = None
    if S.log.logfile != sys.stdout:
        logfile = S.log.logfile.name
    elif silence_pysmurf:
        logfile = make_filename(S, 'take_tickle.log')
        print(f"Writing pysmurf logs to {logfile}")
        S.set_logfile(logfile)
        logs_silenced = True

    init_biases = S.get_tes_bias_bipolar_array()
    bias_array = S.get_tes_bias_bipolar_array()
    bias_groups = np.array(bias_groups)
    start_times = np.zeros_like(bias_groups, dtype=np.float64)
    stop_times = np.zeros_like(bias_groups,  dtype=np.float64)
    dat_files = []

    if (bias_groups >= 12).any():
        raise ValueError("Can only run with bias groups < 12")

    for i, bg in enumerate(bias_groups):
        orig_hc_mode = get_current_mode(S, bg)
        orig_bias = S.get_tes_bias_bipolar(bg)
        new_bias = None
        if high_current and (not orig_hc_mode):
            new_bias = orig_bias / S.high_low_current_ratio
            print(f"Setting bias_group {bg} to high current mode.")
            print(f"Changing bias from {orig_bias} to {new_bias} to preserve "
                  f"dc-current.")
            S.set_tes_bias_high_current(bg)
            S.set_tes_bias_bipolar(bg, new_bias)
            bias_array[bg] = new_bias

        print(f"Playing sine wave on bias group {bg}")
        S.play_sine_tes(bg, tickle_voltage, tickle_freq)
        time.sleep(1)
        dat_file = S.stream_data_on(make_freq_mask=False)
        dat_files.append(dat_file)
        start_times[i] = time.time()
        time.sleep(duration)
        stop_times[i] = time.time()
        S.stream_data_off()
        S.set_rtm_arb_waveform_enable(0)
        S.set_tes_bias_bipolar(bg, init_biases[bg])

        if new_bias is not None:
            print(f"Restoring bg {bg} to low-current mode  and dc bias to "
                  f"{orig_bias}")
            S.set_tes_bias_low_current(bg)
            S.set_tes_bias_bipolar(bg, orig_bias)

        time.sleep(2)  # Gives some time for g3 file to finish

    if logs_silenced:  # Returns logs to stdout
        S.set_logfile(None)

    summary = {
        'tickle_freq': tickle_freq,
        'tone_voltage': tickle_voltage,
        'high_current': high_current,
        'bias_array': bias_array,
        'bias_groups': bias_groups,
        'start_times': start_times,
        'stop_times': stop_times,
        'dat_files': dat_files,
    }
    filename = make_filename(S, 'tickle_summary.npy')
    np.save(filename, summary, allow_pickle=True)
    S.pub.register_file(filename, 'tickle_summary', format='npy')
    print(f"Saved tickle summary to {filename}")
    return filename


@set_action()
def bias_detectors_from_sc(S, bias_points_fp, high_current_mode=False):
    """
    Biases detectors using the bias points calculated for various bias
    groups using sodetlib.det_analysis.find_bias_points. Will bias any
    bias groups that are specified in the file at bias_points_fp. 

    Args
    ----
    S:
        SmurfControl object
    bias_points_fp: str
        path to the .npy file generated by sodetlib.det_analysis.find_bias_points
    high_current_mode: bool, default False
        Whether or not you want to bias the bias groups in high current mode.

    Returns
    -------
    None

    """

    bias_points = np.load(bias_points_fp, allow_pickle=True).item()

    bias_groups = np.fromiter(bias_points['biases'].keys(), dtype=int)
    bias_values = np.fromiter(bias_points['biases'].values(), dtype=float)

    # For now, this is hard-coded. Should be pulled from uxm_config
    overbias_voltage = 19.9

    bias_array = S.get_tes_bias_bipolar_array()
    bias_array[bias_groups] = bias_values

    print(f'Overbiasing and setting TES biases on bias groups {bias_groups}.')
    S.overbias_tes_all(bias_groups=bias_groups,
                       overbias_voltage=overbias_voltage,
                       tes_bias=overbias_voltage,
                       high_current_mode=high_current_mode)

    S.set_tes_bias_bipolar_array(bias_array)

def so_play_tes_bipolar_waveform(S, bias_group, waveform, do_enable=True,
                                 continuous=True, **kwargs):
    """
    Play a bipolar waveform on the bias group.
    Args
    ----
    bias_group : int
                The bias group
    waveform : float array
                The waveform the play on the bias group.
    do_enable : bool, optional, default True
                Whether to enable the DACs (similar to what is required
                for TES bias).
    continuous : bool, optional, default True
                Whether to play the TES waveform continuously.
    """
    bias_order = S.bias_group_to_pair[:,0]

    dac_positives = S.bias_group_to_pair[:,1]
    dac_negatives = S.bias_group_to_pair[:,2]

    dac_idx = np.ravel(np.where(bias_order == bias_group))

    dac_positive = dac_positives[dac_idx][0]
    dac_negative = dac_negatives[dac_idx][0]

    # https://confluence.slac.stanford.edu/display/SMuRF/SMuRF+firmware#SMuRFfirmware-RTMDACarbitrarywaveforms
    # Target the two bipolar DACs assigned to this bias group:
    S.set_dac_axil_addr(0, dac_positive)
    S.set_dac_axil_addr(1, dac_negative)

    # Must enable the DACs (if not enabled already)
    if do_enable:
        S.set_rtm_slow_dac_enable(dac_positive, 2, **kwargs)
        S.set_rtm_slow_dac_enable(dac_negative, 2, **kwargs)

    # Load waveform into each DAC's LUT table.  Opposite sign so
    # they combine coherenty
    S.set_rtm_arb_waveform_lut_table(0, waveform)
    S.set_rtm_arb_waveform_lut_table(1, -waveform)

    # Enable waveform generation (3=on both DACs)
    S.set_rtm_arb_waveform_enable(3)

    # Continous mode to play the waveform continuously
    if continuous:
        S.set_rtm_arb_waveform_continuous(1)
    else:
        S.set_rtm_arb_waveform_continuous(0)


def setup_square_wave(S, bg, step_dur, step_size):
    # Setup waveform
    sig = np.ones(2048)
    cur_bias = S.get_tes_bias_bipolar(bg)
    sig *= cur_bias / (2*S._rtm_slow_dac_bit_to_volt)
    sig[1024:] += step_size / (2*S._rtm_slow_dac_bit_to_volt)
    ts = int(step_dur/(6.4e-9 * 2048))
    S.set_rtm_arb_waveform_timer_size(ts, wait_done=True)
    return sig


def play_squarewave(S, bg, sig, step_dur=0.75, toggle_streaming=False,
                    n_steps=4):
    '''
    Function to play a square wave of amplitude
    step_size on a single bias group (bg). The step will be
    played about whatever DC level the bias group is currently
    at. You can set this before this function with S.set_tes_bias_bipolar
    S.overbias or equivalent functions.
    Args
    --------
    S : class, Pysmurf control object.
    bg : int, bias group to play step function on.
    step_size : float, amplitude of the step in volts at the TES
                bias DAC.
    play_time : float, length of time to play square wave.
                Defaults is to play indefinitely.
    step_dur : float, time in sec for one step cycle to complete.
    Returns
    --------
    datfile : str, filepath to .dat file generated.
    start : int, starting ctime of the step data.
    stop : int, ending ctime of the step data.
    '''
    # Play waveform
    cur_bias = S.get_tes_bias_bipolar(bg)
    start = S.get_timestamp()
    if toggle_streaming:
        sid = so.stream_g3_on(S)
    time.sleep(step_dur/2)
    so_play_tes_bipolar_waveform(S,bg, sig)
    time.sleep(step_dur*(n_steps+1))
    # Instead of using the pysmurf stop function, set to the original dc value
    S.set_rtm_arb_waveform_enable(0)
    S.set_tes_bias_bipolar(bg, cur_bias)
    time.sleep(step_dur/2)
    if toggle_streaming:
        so.stream_g3_off(S)
    stop = S.get_timestamp()
    if toggle_streaming:
        return sid, start, stop
    else:
        return start, stop

def calc_step_size_pW2V(S, pW=0.1, Rfrac=0.5, Rn=8):
    '''
    Function to calculate bias step size in volts given a target
    Rfrac, normal resistance, and step amplitude in picowatts.
    Args
    --------
    pW : float, Desired bias step size in picowatts. Defaults
                    to 0.1.
    Rfrac : float, Fraction of normal resistance at desired bias
                    point must be a number > 0 and <= 1. Defaults
                    to 0.5.
    Rn : float, Normal resistance in mOhms. Defaults to 8.
    Returns
    -------
    V : float, Bias step amplitude in volts at TES bias DAC.
    '''
    R = Rfrac*Rn
    V_tes = np.sqrt(pW*1e-12*R*1e-3)
    V = S.bias_line_resistance*(V_tes*(R*1e-3+S.R_sh)/(R*1e-3*R_sh))
    return V


def calc_step_size_V2pW(S, V=0.1, Rfrac=0.5, Rn=8):
    '''
    Function to calculate bias step size in volts given a target
    Rfrac, normal resistance, and step amplitude in picowatts.
    Args
    --------
    V : float, Desired bias step size in volts commanded.
        Default to 0.1.
    Rfrac : float, Fraction of normal resistance at desired bias
        point must be a number > 0 and <= 1. Defaults
        to 0.5.
    Rn : float, Normal resistance in mOhms. Defaults to 8.
    Returns
    -------
    pW : float, bias step amplitude in pW on TES.
    '''
    R = Rfrac*Rn
    V_tes = (V/S.bias_line_resistance)*(R*1e-3*S.R_sh/(R*1e-3+S.R_sh))
    pW = V_tes**2/(R*1e-3)
    return pW


@set_action()
def take_step_vs_bias(S, cfg, bgs, step_size, bias_high, ob_volt,
                      bias_low, bias_step, wait_time=1.0, n_steps=3,
                      step_dur=0.5, cool_wait=180,bmap = None):
    '''
    Args
    -------
    S : class, pysmurf control instance.
    cfg : class, device config instance.
    bgs : int list, bias groups to take step data on.
    step_size : float, amplitude of bias step.
    bias_high : float, maximum voltage value in the transition.
    ob_volt : float, overbias amplitude in volts.
    bias_low : float, minimum voltage value in the transition.
    bias_step : float, step size between bias_high and bias_low through the
                transition.
    wait_time : float, time to wait after changing the bias point in sec.
                Default is 1 second.
    cool_wait : float, time to wait after overbiasing.
    play_time : float, time in sec to play the step function at each bias point
                Default is 5 seconds.
    step_dur : float, period of step function in seconds. Default is 0.75s.
    Returns
    -------
    out_file : str, filepath to file containing all info needed to analyze the
                bias step vs bias point data.
    '''
    ctime = S.get_timestamp()
    outfpath = f'{S.output_dir}/{ctime}_bias_step_info'
    out_dict = {}
    out_dict['info'] = {}
    out_dict['info']['step_size'] = step_size
    out_dict['info']['n_steps'] = n_steps
    out_dict['info']['step_dur'] = step_dur
    out_dict['info']['output_dir'] = S.output_dir
    start_downsample = S.get_downsample_factor()
    S.log('Turning off downsampling')
    S.set_downsample_factor(1)
    fs = S.get_flux_ramp_freq()*1e3/S.get_downsample_factor()
    out_dict['info']['fs'] = fs
    S.log('Turning off downsample filter')
    S.set_filter_disable(1)
    for bg in bgs:
        print(f'biasgroup {bg}')
        out_dict[bg] = {}
        out_dict[bg]['bias'] = []
        out_dict[bg]['start'] = []
        out_dict[bg]['stop'] = []
        S.log('Setting to high current mode to bypass low current'
              f'mode analog filter on biasgroup {bg}')
        S.set_tes_bias_high_current(bg)
        S.set_tes_bias_bipolar(bg, 0.0)
        # Convert step size to high current mode
        hlr = S.high_low_current_ratio
        step_size /= hlr
        S.log(f'Overbiasing on biasgroup {bg}')
        S.overbias_tes(bias_group=bg, tes_bias=bias_high/hlr, overbias_wait=5,
                       overbias_voltage=ob_volt, high_current_mode=True,
                       cool_wait=cool_wait)
        bias_vals = np.arange(bias_high/hlr, bias_low/hlr, -bias_step/hlr)
        S.log(f'Stepping through bias values from {bias_high/hlr}, to'
              f'{bias_low/hlr} in steps of {-bias_step/hlr} for a total'
              f'of {len(bias_vals)} steps')
        #sig = setup_square_wave(S, bg, step_dur, step_size)
        if bmap is None:
            sid = so.stream_g3_on(S)
        else:
            bgmap = np.where(bmap == bg)[0]
            sid = so.stream_g3_on(S,channel_mask=bgmap)
        out_dict[bg]['sid'] = sid
        time.sleep(2)
        for b_bias in tqdm(bias_vals):
            S.set_tes_bias_bipolar(bg, b_bias)
            time.sleep(wait_time)
            sig = setup_square_wave(S, bg, step_dur, step_size)
            start, stop = play_squarewave(S, bg, sig, step_dur=0.75,
                                          toggle_streaming=False,
                                          n_steps=4)
            out_dict[bg]['bias'].append(b_bias)
            out_dict[bg]['start'].append(start)
            out_dict[bg]['stop'].append(stop)
        so.stream_g3_off(S,)
        S.log('Setting back to low current mode')
        S.set_tes_bias_low_current(bg)
        np.save(outfpath, out_dict)
    S.log('Turning back on downsampling, and filtering')
    S.set_filter_disable(0)
    S.set_downsample_factor(start_downsample)
    np.save(outfpath, out_dict)
    S.pub.register_file(f'{outfpath}.npy', 'bias_step_info', format='npy')
    S.log(f'Saved bias step info to {outfpath}.npy')
    'm 'return out_dict
