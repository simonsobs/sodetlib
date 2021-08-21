import numpy as np
from tqdm.auto import tqdm
import time
import sodetlib.smurf_funcs.smurf_ops as so
import sodetlib.util as su


def set_flux_ramp_rate(S, cfg, band, flux_ramp_rate_khz, tracking_kwargs=None,
                       nphi0=5):
    tk = su.get_tracking_kwargs(S, cfg, band, kwargs=tracking_kwargs)
    init_nphi0 = tk['lms_freq_hz'] / (tk['reset_rate_khz']*1e3)
    tk['fraction_full_scale'] *= nphi0 / init_nphi0
    tk['lms_freq_hz'] *= nphi0 / init_nphi0
    tk['lms_freq_hz'] *= flux_ramp_rate_khz / tk['reset_rate_khz']
    tk['reset_rate_khz'] = flux_ramp_rate_khz
    S.tracking_setup(band, **tk)


def take_complex_impedance(S, cfg, bias_groups=None, freqs=None,
                           tickle_voltage=0.005, meas_time=2.,
                           flux_ramp_rate_khz=10e3):
    if bias_groups is None:
        bias_groups = np.arange(12)
    bias_groups = np.atleast_1d(bias_groups)
    if freqs is None:
        freqs = np.logspace(1, 3, 10)
    freqs = np.atleast_1d(freqs)
    nsteps = len(bias_groups) * len(freqs)

    start_times = []
    stop_times = []

    # Record initial params
    bcfg = cfg.dev.bands[0]
    init_fr_freq = bcfg['flux_ramp_rate_khz']
    init_nphi0 = np.round(
        bcfg['lms_freq_hz'] / bcfg['flux_ramp_rate_khz'] / 1e3
    )
    init_ds_factor = S.get_downsample_factor()
    init_filter_disable = S.get_filter_disable()

    # Sets FR Rate to 10 Khz, downsample factor to one, and disables filter
    S.log(f"Setting fr freqs to {flux_ramp_rate_khz} khz")
    for band in S._bands:
        set_flux_ramp_rate(S, cfg, band, flux_ramp_rate_khz, nphi0=3)
    S.set_downsample_factor(1)
    S.set_filter_disable(1)

    orig_biases = S.get_tes_bias_bipolar_array()
    new_biases = orig_biases / S.high_low_current_ratio

    pb = tqdm(total=nsteps)
    sid = so.stream_g3_on(S, tag='tickle')
    S.log(f"Streaming with stream-id: {sid}")
    for bg in bias_groups:
        S.set_tes_bias_bipolar(bg, new_biases[bg])
        S.set_tes_bias_high_current(bg)

        for freq in freqs:
            S.log(f"Tickle with bg={bg}, freq={freq}")
            S.play_sine_tes(bg, tickle_voltage, freq)
            start_times.append(time.time())
            time.sleep(meas_time)
            stop_times.append(time.time())
            S.set_rtm_arb_waveform_enable(0)
            S.set_tes_bias_bipolar(bg, new_biases[bg])
            pb.update(1)

        S.set_tes_bias_bipolar(bg, orig_biases[bg])
    so.stream_g3_off(S)

    # Restores all params
    S.log("Returning flux ramp to original frequencies")
    for band in S._bands:
        set_flux_ramp_rate(S, cfg, band, init_fr_freq, nphi0=init_nphi0)
    S.set_downsample_factor(init_ds_factor)
    S.set_filter_disable(init_filter_disable)

    summary = {
        'start_times': start_times,
        'stop_times': stop_times,
        'sid': sid,
        'freqs': freqs,
        'bias_groups': bias_groups,
        'dc_biases': orig_biases,
    }

    summary_file = su.make_filename(S, 'complex_impedance_summary.npy')
    np.save(summary_file, summary, allow_pickle=True)
    S.pub.register_file(summary_file, 'complex_impedance')
    return summary_file, summary



