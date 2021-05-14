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
from sodetlib.util import make_filename

from pysmurf.client.util.pub import set_action


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
def take_iv(S, bias_groups=None, wait_time=.1, bias=None,
            bias_high=1.5, bias_low=0, bias_step=.005,
            show_plot=False, overbias_wait=2., cool_wait=30,
            make_plot=True, save_plot=True, plotname_append='',
            channels=None, band=None, high_current_mode=True,
            overbias_voltage=8., grid_on=True,
            phase_excursion_min=3., bias_line_resistance=None,
            do_analysis=True, cool_voltage=None):
    """
    Replaces the pysmurf run_iv function to be more appropriate for SO-specific
    usage, as well as easier to edit as needed.  Steps the TES bias down
    slowly. Starts at bias_high to bias_low with step size bias_step. Waits
    wait_time between changing steps. After overbiasing, can choose bias point
    to allow system to cool down.

    Args
    ----
    bias_groups : numpy.ndarray or None, optional, default None
        Which bias groups to take the IV on. If None, defaults to
        the groups in the config file.
    wait_time : float, optional, default 0.1
        The amount of time between changing TES biases in seconds.
    bias : float array or None, optional, default None
        A float array of bias values. Must go high to low.
    bias_high : float, optional, default 1.5
        The maximum TES bias in volts.
    bias_low : float, optional, default 0
        The minimum TES bias in volts.
    bias_step : float, optional, default 0.005
        The step size in volts.
    overbias_wait : float, optional, default 2.0
        The time to stay in the overbiased state in seconds.
    cool_wait : float, optional, default 30.0
        The time to stay in the low current state after
        overbiasing before taking the IV.
    high_current_mode : bool, optional, default True
        The current mode to take the IV in.
    overbias_voltage : float, optional, default 8.0
        The voltage to set the TES bias in the overbias stage.
    grid_on : bool, optional, default True
        Grids on plotting.
    phase_excursion_min : float, optional, default 3.0
        The minimum phase excursion required for making plots.
    bias_line_resistance : float or None, optional, default None
        The resistance of the bias lines in Ohms. If None, loads value
        in config file
    do_analysis: bool, optional, default True
        Whether to do the pysmurf IV analysis
    cool_voltage: float, optional, default None
        The voltage to bias at after overbiasing before taking the IV
        while the system cools.
    Returns
    -------
    output_path : str
        Full path to IV raw npy file.
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
    datafile = S.stream_data_on()
    S.log(f'writing to {datafile}')
    for b in bias:
        S.log(f'Bias at {b:4.3f}')
        S.set_tes_bias_bipolar_array(b * bias_group_bool)
        time.sleep(wait_time)
    S.stream_data_off()
    stop_time = S.get_timestamp()  # get time the IV finishes
    S.log(f'Finishing IV at {stop_time}')
    basename, _ = os.path.splitext(os.path.basename(datafile))
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
    iv_info['datafile'] = datafile
    iv_info['bias'] = bias
    iv_info['bias group'] = bias_groups

    iv_info_fp = os.path.join(S.output_dir, basename + '_iv_info.npy')

    np.save(iv_info_fp, iv_info)
    S.log(f'Writing IV information to {iv_info_fp}.')
    S.pub.register_file(iv_info_fp, 'iv_info', format='npy')

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

    if silence_pysmurf:
        fname = make_filename(S, 'take_tickle.log')
        S.set_logfile(fname)

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
        S.stop_tes_bipolar_waveform(bg)
        S.set_tes_bias_bipolar(bg, init_biases[bg])

        if new_bias is not None:
            print(f"Restoring bg {bg} to low-current mode  and dc bias to "
                  f"{orig_bias}")
            S.set_tes_bias_low_current(bg)
            S.set_tes_bias_bipolar(bg, orig_bias)

        time.sleep(2)  # Gives some time for g3 file to finish

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

    bias_points = np.load(bias_points_fp, allow_pickle=True).item()

    bias_groups = np.fromiter(bias_points.keys(), dtype=int)
    bias_values = np.fromiter(bias_points.values(), dtype=float)

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
