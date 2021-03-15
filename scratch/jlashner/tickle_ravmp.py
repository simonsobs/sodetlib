import numpy as np
import time
from sodetlib.util import make_filename

from pysmurf.client.util.pub import set_action


@set_action()
def take_tickle(S, cfg, bias_groups, tickle_freq=1., tone_voltage=0.005,
                duration=3.):
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
    tone_voltage : float
        voltage of tickle
    duration : float
        duration of tickle (sec)
    """
    if isinstance(bias_groups, (float, int)):
        bias_groups = [bias_groups]

    init_biases = S.get_tes_bias_bipolar_array()
    bias_groups = np.array(bias_groups)
    start_times = np.zeros_like(bias_groups, dtype=np.float32)
    stop_times = np.zeros_like(bias_groups,  dtype=np.float32)
    dat_files = []

    for i, bg in enumerate(bias_groups):
        print(f"Playing sine wave on bias group {bg}")
        S.play_sine_tes(bg, tone_voltage, tickle_freq)
        dat_file = S.stream_data_on(make_freq_mask=False)
        dat_files.append(dat_file)
        start_times[i] = time.time()
        time.sleep(duration)
        stop_times[i] = time.time()
        S.stream_data_off()
        S.stop_tes_bipolar_waveform(bg)
        S.set_tes_bias_bipolar(bg, init_biases[bg])
        time.sleep(2)  # Gives some time for g3 file to finish

    summary = {
        'tickle_freq': tickle_freq,
        'tone_voltage': tone_voltage,
        'bias_array': S.get_tes_bias_bipolar_array(),
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
