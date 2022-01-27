import numpy as np
import tqdm
import os
import mock

if os.environ.get("SODETLIB_IMPORT_PYSMURF", True):
    from pysmurf.client.util.pub import set_action
    from pysmurf.client.base.smurf_control import SmurfControl
else:
    set_action = lambda : (lambda f: f)
    SmurfControl = mo

@set_action()
def take_complex_impedance(S: SmurfControl, cfg, bias_groups=None, freqs=None,
                           tickle_voltage=0.005, meas_time=2.,
                           flux_ramp_rate_khz=10e3, bgmap_file=None,
                           skip_tracking_setup=False, feedback_gain=None):
    S.set_tes_bias_bipolar_array()

    if bias_groups is None:
        bias_groups = np.arange(12)
    bias_groups = np.atleast_1d(bias_groups)

    if freqs is None:
        freqs = np.logspace(1, np.log10(2e3), 10)
    freqs = np.atleast_1d(freqs)

    nsteps = len(bias_groups) * len(freqs)

    if bgmap_file is None:
        bgmap_file = cfg.dev.exp['bgmap_file']

    # Loads bias group mapping into an array. We may need to change this
    # if mapping file is updated
    _bgmap = np.load(bgmap_file, allow_pickle=True).item()
    bgmap = np.full(4096, -1)
    chans_per_band = S.get_number_channels()
    for band, v in _bgmap.items():
        for chan, bg in v.items():
            bgmap[band * chans_per_band + chan] = bg

    start_times = []
    stop_times = []

    # Record initial params so we can restore later
    bcfg = cfg.dev.bands[0]
    init_fr_freq = bcfg['flux_ramp_rate_khz'] * 1e3
    init_nphi0 = np.round(
        bcfg['lms_freq_hz'] / init_fr_freq
    )

    # Setup tracking for all bands to accomodate for higher freq data
    if not skip_tracking_setup:
        S.log(f"Setting fr freqs to {flux_ramp_rate_khz} khz")
        tracking_setup_mult_bands(S, cfg, S._bands, fr_rate=flux_ramp_rate_khz,
                                  nphi0=3, show_plots=False, feedback_gain=feedback_gain)

    # Gets array of all enabled channel to use when channel masking
    enabled_abs_chans = np.zeros(4096, dtype=bool)
    for band in S._bands:
        ecs = np.where(S.get_amplitude_scale_array(band) > 0)[0] + band * chans_per_band
        enabled_abs_chans[ecs] = 1

    orig_biases = S.get_tes_bias_bipolar_array()
    new_biases = orig_biases / S.high_low_current_ratio

    pb = tqdm(total=nsteps)
    sids = []

    init_ds_factor = S.get_downsample_factor()
    init_filter_disable = S.get_filter_disable()
    try:
        S.set_filter_disable(1)
        S.set_downsample_factor(1)

        # Sets all bias groups to high-current mode + lower voltage
        for bg in bias_groups:
            S.set_tes_bias_high_current(bg)
        S.set_tes_bias_bipolar_array(new_biases)
        time.sleep(30)

        for bg in bias_groups:
            channel_mask = np.where((bgmap == bg) & enabled_abs_chans)[0]
            sid = so.stream_g3_on(S, tag='tickle', channel_mask=channel_mask)
            sids.append(sid)
            S.log(f"Streaming Bias Group {bg} with sid {sid}")

            for freq in freqs:
                S.log(f"Tickle with bg={bg}, freq={freq}")
                S.play_sine_tes(bg, tickle_voltage, freq)
                start_times.append(time.time())
                time.sleep(meas_time)
                stop_times.append(time.time())
                S.set_rtm_arb_waveform_enable(0)
                S.set_tes_bias_bipolar(bg, new_biases[bg])
                pb.update(1)

            so.stream_g3_off(S)
    finally:
        S.log("Returning filter-disable and downsample factor to initial values")
        S.set_filter_disable(init_filter_disable)
        S.set_downsample_factor(init_ds_factor)

        # Return to low current mode 
        S.set_tes_bias_bipolar_array(orig_biases)
        for bg in bias_groups:
            S.set_tes_bias_low_current(bg)

    summary = {
        'start_times': start_times,
        'stop_times': stop_times,
        'sids': sids,
        'freqs': freqs,
        'tickle_voltage': tickle_voltage,
        'bias_groups': bias_groups,
        'dc_biases': orig_biases,
    }

    summary_file = su.make_filename(S, 'complex_impedance_summary.npy')
    np.save(summary_file, summary, allow_pickle=True)
    S.pub.register_file(summary_file, 'complex_impedance')

    # Restores all params
    S.log("Returning flux ramp to original frequencies")

    if not skip_tracking_setup:
        tracking_setup_mult_bands(S, cfg, S._bands, fr_rate=init_fr_freq,
                                  nphi0=init_nphi0, show_plots=False)

    return summary_file, summary

