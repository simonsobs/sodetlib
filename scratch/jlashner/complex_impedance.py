import numpy as np
from tqdm.auto import tqdm
import time
import sodetlib.smurf_funcs.smurf_ops as so
import sodetlib.smurf_utils as su

def take_complex_impedance(S, cfg, bias_groups, freqs=None,
                           tickle_voltage=0.005, meas_time=2.):
    bias_groups = np.atleast_1d(bias_groups)
    if freqs is None:
        freqs = np.logspace(1, 3, 10)
    freqs = np.atleast_1d(freqs)
    nsteps = len(bias_groups) * len(freqs)

    start_times = []
    stop_times = []

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



