import numpy as np
import sodetlib.smurf_funcs.smurf_ops as so
import time
from pysmurf.client.util.pub import set_action
import sodetlib.util as su
from tqdm.auto import tqdm

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

    # Enable waveform generation (1=on both DACs)
    S.set_rtm_arb_waveform_enable(3)

    # Continous mode to play the waveform continuously
    if continuous:
        S.set_rtm_arb_waveform_continuous(1)
    else:
        S.set_rtm_arb_waveform_continuous(0)

def play_bias_steps_dc(S, cfg, bias_groups, step_duration, step_voltage,
                       num_steps=5, do_enable=True):
    """
    Plays bias steps on a group of bias groups stepping with only one DAC
    """
    if bias_groups is None:
        bias_groups = np.arange(12)
    bias_groups = np.atleast_1d(bias_groups)

    dac_volt_array_low = S.get_rtm_slow_dac_volt_array()
    dac_volt_array_high = dac_volt_array_low.copy()

    dac_enable_array = S.get_rtm_slow_dac_enable_array()

    bias_order, dac_positives, dac_negatives = S.bias_group_to_pair.T

    for bg in bias_groups:
        bg_idx = np.ravel(np.where(bias_order == bg))
        dac_positive = dac_positives[bg_idx][0] - 1
        dac_volt_array_high[dac_positive] += step_voltage


    start = time.time()
    for _ in range(num_steps):
        S.set_rtm_slow_dac_volt_array(dac_volt_array_high)
        time.sleep(duration)
        S.set_rtm_slow_dac_volt_array(dac_volt_array_low)
        time.sleep(duration)
    stop = time.time()

    return start, stop

def play_bias_steps_waveform(S, cfg, bias_group, step_duration, step_voltage,
                             num_steps=5, dc_bias=None):
    if dc_bias is None:
        dc_bias = S.get_tes_bias_bipolar(bias_group)
    sig, timer_size = make_step_waveform(S, step_duration, step_voltage, dc_bias)
    S.set_rtm_arb_waveform_timer_size(timer_size, wait_done=True)
    so_play_tes_bipolar_waveform(S, bias_group, sig)
    start_time = time.time()
    time.sleep(step_duration * (num_steps+1))
    stop_time = time.time()
    S.set_rtm_arb_waveform_enable(0)
    return start_time, stop_time


def make_step_waveform(S, step_dur, step_voltage, dc_voltage):
    ""
    # Setup waveform
    sig = np.ones(2048)
    sig *= dc_voltage / (2*S._rtm_slow_dac_bit_to_volt)
    sig[1024:] += step_voltage / (2*S._rtm_slow_dac_bit_to_volt)
    timer_size = int(step_dur/(6.4e-9 * 2048))
    return sig, timer_size

@set_action()
def bias_steps_vs_bias(S, cfg, bias_groups=None, biases=None,
                       step_duration=0.1, step_voltage=0.05, num_steps=5,
                       cool_wait=180, overbias_voltage=19.9, overbias=True,
                       step_wait=1.0, bgmap=None, show_pb=True):
    """
    Runs bias steps vs DC bias. Plays steps one bg at a time using the waveform
    generator to minimize heating from bias.
    """
    if bias_groups is None:
        bias_groups = np.arange(12)
    if biases is None:
        biases = np.arange(10, 0, 0.5)

    biases = np.atleast_1d(biases)
    # Make sure biases are sorted in descending order
    biases.sort()
    biases = biases[::-1]

    bias_groups = np.atleast_1d(bias_groups)

    nbgs = len(bias_groups)
    nbiases = len(biases)
    nsegments = nbgs * nbiases
    start_times = np.full((nbgs, nbiases), np.nan)
    stop_times = np.full((nbgs, nbiases), np.nan)
    sids = np.zeros(nbgs, dtype=np.int)

    initial_ds_factor = S.get_downsample_factor()
    initial_filter_disable = S.get_filter_disable()
    S.set_downsample_factor(1)
    S.set_filter_disable(1)

    # Convert voltages to high-low-ratio units
    biases = biases / S.high_low_current_ratio
    step_voltage = step_voltage / S.high_low_current_ratio
    # Sets initial bias to zero, since apparently switching a single bias-line
    # to high-current switches them all
    S.log("Setting all bias-groups to 0 and high-current mode")
    for bg in range(12):
        S.set_tes_bias_bipolar(bg, 0)
    # Now set to high current mode
    for bg in range(12):
        S.set_tes_bias_high_current(bg)

    # Create bias group map array based on device cfg file if one is not
    # specified
    if bgmap is None:
        bgmap_file = cfg.dev.exp['bg_map']
        _bgmap = np.load(bgmap_file, allow_pickle=True).item()
        bgmap = np.full(4096, -1)
        chans_per_band = S.get_number_channels()
        for band, v in _bgmap.items():
            for chan, bg in v.items():
                bgmap[band * chans_per_band + chan] = bg

    # Array mask of enabled channels to use for the channel mask
    enabled_abs_chans = np.zeros(4096, dtype=bool)
    for band in S._bands:
        ecs = np.where(S.get_amplitude_scale_array(band > 0))[0] + 512 * band
        enabled_abs_chans[ecs] = 1

    pb = tqdm(total=nsegments, disable=not show_pb)
    try:  # Main loop
        for bg_idx, bg in enumerate(bias_groups):
            # Get channel map for bias-group
            channel_mask = np.where((bgmap == bg) & enabled_abs_chans)[0]

            # Overbias detectors
            if overbias:
                S.log(f"Overbiasing bg {bg}")
                S.overbias_tes(
                    bias_group=bg, tes_bias=biases[0], overbias_wait=5,
                    overbias_voltage=overbias_voltage, high_current_mode=True,
                    cool_wait=cool_wait
                )

            sid = so.stream_g3_on(S, channel_mask=channel_mask)
            S.log(f"Running bias sweep for BG {bg}")
            for bias_idx, dc_bias in enumerate(biases):
                S.log(f"BG={bg}, bias={dc_bias}")
                S.set_tes_bias_bipolar(bg, dc_bias) # Do this 
                time.sleep(step_wait)
                start, stop = play_bias_steps_waveform(
                    S, cfg,bg, step_duration, step_voltage,
                    num_steps=num_steps
                )
                start_times[bg_idx, bias_idx] = start
                stop_times[bg_idx, bias_idx] = stop
                pb.update()
            so.stream_g3_off(S)
            sids[bg_idx] = sid
    finally:
        # Tries to return to orig state on failure
        S.log("Turning off data stream and resetting ds factor and filter")
        so.stream_g3_off(S)
        S.set_downsample_factor(initial_ds_factor)
        S.set_filter_disable(initial_filter_disable)
        for bg in range(12):
            S.set_tes_bias_low_current(bg)
            S.set_tes_bias_bipolar(bg, 0)

    outputs = {
        'bias_groups': bias_groups,
        'dc_biases': biases,
        'start_times': start_times,
        'stop_times': stop_times,
        'sids': sids,
        'summary': {
            'step_duration': step_duration,
            'num_steps': num_steps,
            'step_voltage': step_voltage,
            'cool_wait': cool_wait,
            'step_wait': step_wait,
            # Saving these things for ease of access and redundancy
            'tunefile': S.tune_file,
            'high_low_current_ratio': S.high_low_current_ratio,
            'bias_line_resistance': S.bias_line_resistance,
            'R_sh': S.R_sh,
        }
    }
    output_path = su.make_filename(S, 'bias_steps_vs_bias.npz')
    np.savez(output_path, **outputs)
    S.pub.register_file(output_path, 'bias_steps')
    return output_path, outputs









