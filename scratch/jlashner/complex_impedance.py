import numpy as np
from tqdm.auto import tqdm
import time
import sodetlib.smurf_funcs.smurf_ops as so
import sodetlib.smurf_utils as su


def set_flux_ramp_rate(S, cfg, band, flux_ramp_rate_khz, tracking_kwargs=None):
    tk = su.get_tracking_kwargs(S, cfg, band, kwargs=tracking_kwargs)
    tk['lms_freq_hkz'] *= flux_ramp_rate_khz / tk['flux_ramp_rate_khz']
    tk['flux_ramp_rate_khz'] = flux_ramp_rate_khz
    S.tracking_setup(band, **tk)


def take_complex_impedance(S, cfg, bias_groups, freqs=None,
                           tickle_voltage=0.005, meas_time=2.,
                           flux_ramp_rate_khz=10e3):
    bias_groups = np.atleast_1d(bias_groups)
    if freqs is None:
        freqs = np.logspace(1, 3, 10)
    freqs = np.atleast_1d(freqs)
    nsteps = len(bias_groups) * len(freqs)

    start_times = []
    stop_times = []

    fr_freqs = [bcfg['flux_ramp_rate_khz'] for bcfg in cfg.dev.bands]
    S.log(f"Setting fr freqs to {flux_ramp_rate_khz} khz")
    for band in S._bands:
        set_flux_ramp_rate(S, cfg, band, flux_ramp_rate_khz)

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

    S.log("Returning flux ramp to original frequencies")
    for band in S._bands:
        set_flux_ramp_rate_khz(S, cfg, band, fr_freqs[band])

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
    return summary_file, summary



