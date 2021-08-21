from tqdm.auto import tqdm
import numpy as np

def noise_vs_bias(S, cfg, dc_biases=None, bgs=None, meas_time=30.,
                  overbias_voltage=19.9, cool_wait=30.):

    if dc_biases is None:
        dc_biases = np.arange(10, 0, -0.5)
    if bgs is None:
        bgs = np.arange(12)
    bgs = np.atleast_1d(bgs)

    S.log(f"Sweeping over biases: {dc_biases}")
    S.log("Overbiasing TES")
    S.overbias_tes_all(bgs, cool_wait=30., tes_bias=np.max(dc_biases))
    start_times = []
    stop_times = []

    sid = so.stream_g3_on(S, tag='noise_vs_bias')
    biases = S.get_tes_bias_bipolar_array()
    for bias in tqdm(dc_biases):
        S.log(f"Setting bias to {bias}")
        biases[bgs] = bias
        S.set_tes_bias_bipolar_array(biases)
        start_times.append(time.time())
        time.sleep(meas_time)
        stop_times.append(time.time())

    start_times = np.array(start_times)
    stop_times = np.array(stop_times)

    so.stream_g3_off(S)
    summary = {
        'dc_biases': dc_biases,
        'bgs': bgs,
        'start_times': start_times,
        'stop_times': stop_times,
        'sid': sid,
    }
    summary_file = su.make_filename(S, 'noise_vs_bias_summary.npy')
    np.save(summary_file, summary, allow_pickle=True)
    S.pub.register_file(summary_file, 'noise_vs_bias')
    return summary_file, summary
