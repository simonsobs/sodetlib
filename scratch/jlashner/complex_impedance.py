import numpy as np
from tqdm.auto import tqdm
import time
import sodetlib.smurf_funcs.smurf_ops as so
import sodetlib.util as su


def tracking_setup_mult_bands(S, cfg, bands, fr_rate=None,
                              nphi0=None, fpp0=None, show_plots=True,
                              num_chan_plots=0, feedback_start_frac=None):
    tks = [
        su.get_tracking_kwargs(S, cfg, b)
        for b in range(8)
    ]
    fpp0s = np.array([tk['fraction_full_scale'] for tk in tks])
    lmsf0s = np.array([tk['lms_freq_hz'] for tk in tks])
    fr_rate0 = tks[0]['reset_rate_khz'] * 1e3
    init_nphi0 = np.round(lmsf0s[0] / fr_rate0)

    # Just choose the frac-pp to be the mean of AMC0 fpp's because they are
    # a bit different between amcs
    if fpp0 is None:
        fpp0 = fpp0s[0]

    fpp, lmsfs = fpp0, lmsf0s
    if nphi0 is not None:
        fpp *= nphi0 / init_nphi0
        lmsfs *= fpp / fpp0s
    if fr_rate is not None:
        lmsfs *= fr_rate / fr_rate0
    else:
        fr_rate = fr_rate0

    for band in bands:
        tk = tks[band]
        tk.update({
            'lms_freq_hz': lmsfs[band],
            'fraction_full_scale': fpp,
            'reset_rate_khz': fr_rate / 1e3,
            'show_plot': show_plots,
        })
        tk['channel'] = S.which_on(band)[:num_chan_plots]
        if feedback_start_frac is not None:
            tk['feedback_start_frac'] = feedback_start_frac

        S.tracking_setup(band, **tk)



def take_complex_impedance(S, cfg, bias_groups=None, freqs=None,
                           tickle_voltage=0.005, meas_time=2.,
                           flux_ramp_rate_khz=10e3, bgmap_file=None):
    if bias_groups is None:
        bias_groups = np.arange(12)
    bias_groups = np.atleast_1d(bias_groups)
    if freqs is None:
        freqs = np.logspace(1, np.log10(2e3), 10)
    freqs = np.atleast_1d(freqs)
    nsteps = len(bias_groups) * len(freqs)
    if bgmap_file is None:
        bgmap_file = cfg.dev.exp['bg_map']

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
    init_ds_factor = S.get_downsample_factor()
    init_filter_disable = S.get_filter_disable()

    # Setup tracking for all bands to accomodate for higher freq data
    S.log(f"Setting fr freqs to {flux_ramp_rate_khz} khz")
    tracking_setup_mult_bands(S, cfg, S._bands, fr_rate=flux_ramp_rate_khz,
                              nphi0=3, show_plots=False)
    S.set_filter_disable(1)
    S.set_downsample_factor(1)

    # Gets array of all enabled channel to use when channel masking
    enabled_abs_chans = np.zeros(4096, dtype=bool)
    for band in S._bands:
        ecs = np.where(S.get_amplitude_scale_array(band) > 0)[0] + band * chans_per_band
        enabled_abs_chans[ecs] = 1

    orig_biases = S.get_tes_bias_bipolar_array()
    new_biases = orig_biases / S.high_low_current_ratio

    pb = tqdm(total=nsteps)
    sids = []
    for bg in bias_groups:
        S.set_tes_bias_bipolar(bg, new_biases[bg])
        S.set_tes_bias_high_current(bg)

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
        S.set_tes_bias_bipolar(bg, orig_biases[bg])

    # Restores all params
    S.log("Returning flux ramp to original frequencies")

    tracking_setup_mult_bands(S, cfg, S._bands, fr_rate=init_fr_freq,
                              nphi0=init_nphi0, show_plots=False)

    S.set_downsample_factor(init_ds_factor)
    S.set_filter_disable(init_filter_disable)

    summary = {
        'start_times': start_times,
        'stop_times': stop_times,
        'sids': sids,
        'freqs': freqs,
        'bias_groups': bias_groups,
        'dc_biases': orig_biases,
    }

    summary_file = su.make_filename(S, 'complex_impedance_summary.npy')
    np.save(summary_file, summary, allow_pickle=True)
    S.pub.register_file(summary_file, 'complex_impedance')
    return summary_file, summary

