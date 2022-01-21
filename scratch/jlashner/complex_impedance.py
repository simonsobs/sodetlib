import numpy as np
from tqdm.auto import tqdm
import time
import sodetlib.smurf_funcs.smurf_ops as so
import sodetlib.util as su
import os

from pysmurf.client.util.pub import set_action


def get_current_mode_array(S):
    """
    Gets high-current-mode relay status for all bias groups
    """
    relay = S.get_cryo_card_relays()
    relay = S.get_cryo_card_relays()  # querey twice to ensure update
    bgs = np.arange(S._n_bias_groups)
    hcms = np.zeros_like(bgs, dtype=bool)
    for i, bg in enumerate(bgs):

        r = np.ravel(S._pic_to_bias_group[np.where(
            S._pic_to_bias_group[:, 1] == bg)])[0]
        hcms[i] = (relay >> r) & 1

    return hcms


@set_action()
def tracking_setup_mult_bands(S, cfg, bands, fr_rate=None,
                              nphi0=None, fpp0=None, show_plots=True,
                              num_chan_plots=0, feedback_start_frac=None,
                              feedback_gain=None):
    tks = [
        su.get_tracking_kwargs(S, cfg, b)
        for b in range(8)
    ]
    fpp0s = np.array([tk['fraction_full_scale'] for tk in tks])
    lmsf0s = np.array([tk['lms_freq_hz'] for tk in tks])
    fr_rate0 = tks[bands[0]]['reset_rate_khz'] * 1e3
    init_nphi0 = np.round(lmsf0s[bands[0]] / fr_rate0)

    # Just choose the frac-pp to be the mean of AMC0 fpp's because they are
    # a bit different between amcs
    if fpp0 is None: fpp0 = fpp0s[bands[0]]

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
        if feedback_gain is not None:
            tk['feedback_gain'] = feedback_gain

        print(tk)
        S.tracking_setup(band, **tk)


def get_metadata(S, cfg, meta=None):
    return {
        'tunefile': S.tune_file,
        'high_low_current_ratio': S.high_low_current_ratio,
        'R_sh': S.R_sh,
        'pA_per_phi0': S.pA_per_phi0,
        'rtm_bit_to_volt': S._rtm_slow_dac_bit_to_volt,
        'bias_line_resistance': S.bias_line_resistance,
        'high_current_mode': get_current_mode_array(S),
        'timestamp': now,
        'stream_id': cfg.stream_id,
        'action': S.pub._action,
        'action_timestamp': S.pub._action_ts,
        'bgmap_file': cfg.dev.exp.get('bgmap_file'),
        'iv_file': cfg.dev.exp.get('iv_file')
    }

def save_data(S, cfg, fname, data, register=True, generate_path=True):
    # Validate data
    for k in ["channels", "bands", "sid"]:
        if k not in data:
            raise ValueError(f"Key '{k}' is required in data")

    if 'meta' in data:
        raise ValueError(f"Key 'meta' is not allowed to already exist in data")

    now = time.time()
    _data = {
        **data,
        'meta': {
            'tunefile': S.tune_file,
            'high_low_current_ratio': S.high_low_current_ratio,
            'R_sh': S.R_sh,
            'pA_per_phi0': S.pA_per_phi0,
            'rtm_bit_to_volt': S._rtm_slow_dac_bit_to_volt,
            'bias_line_resistance': S.bias_line_resistance,
            'high_current_mode': get_current_mode_array(S),
            'timestamp': now,
            'stream_id': cfg.stream_id,
            'action': S.pub._action,
            'action_timestamp': S.pub._action_ts,
            'bgmap_file': cfg.dev.exp.get('bgmap_file'),
            'iv_file': cfg.dev.exp.get('iv_file')
        }
    }

    if generate_path:
        path = su.make_filename(S, fname, ctime=int(now))
    else:
        path = fname

    np.save(path, _data, allow_pickle=True)
    if register:
        S.pub.register_file(path, 'sodetlib_data', format='npy')
    return path


def write_bgmap(S, cfg, bands, chans, bgmap, sid, update_cfg=True):
    bgmap_data = {
        'bands': bands,
        'channels': chans,
        'sid': sid,
        'bgmap': bgmap,
    }
    ts = S.get_timestamp()
    bgmap_dir = '/data/smurf_data/bias_group_maps/'
    path = os.path.join(
        bgmap_dir, ts[:5], cfg.stream_id, f'{ts}_bg_map.npy'
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    save_data(S, cfg, path, bgmap_data, generate_path=False)

    if update_cfg:
        cfg.dev.update_experiment({'bgmap_file': path}, update_file=True)
    return path


class ComplexImpedanceAnalysis:
    def __init__(self, S, cfg, bias_groups, freqs):
        self._S = S
        self._cfg = cfg
        self.bias_groups = bias_groups
        self.freqs = freqs
        self.nsteps = len(bias_groups) * len(freqs)
        self.analyzed = False





@set_action()
def take_complex_impedance(S, cfg, bias_groups=None, freqs=None,
                           tickle_voltage=0.005, meas_time=2.,
                           flux_ramp_rate_khz=10e3, bgmap_file=None,
                           skip_tracking_setup=False, feedback_gain=None,
                           max_periods=20):
    if bias_groups is None:
        bias_groups = np.arange(12)
    bias_groups = np.atleast_1d(bias_groups)

    if freqs is None:
        freqs = np.logspace(1, np.log10(2e3), 10)
    freqs = np.atleast_1d(freqs)

    nsteps = len(freqs)* len(bias_groups)

    anal = ComplexImpedanceAnalysis(S, cfg, bias_groups, freqs)
    anal.run_kwargs = {
        'tickle_voltage': tickle_voltage, 'meas_time': meas_time,
        'flux_ramp_rate_khz': flux_ramp_rate_khz, 'bgmap_file': bgmap_file,
        'skip_tracking_setup': skip_tracking_setup, 'feedback_gain': feedback_gain
    }

    if bgmap_file is None:
        bgmap_file = cfg.dev.exp.get('bgmap_file')
        if bgmap_file is None:
            raise ValueError("bgmap_file not in dev cfg. Must be specified in "
                             "function argument")

    # Selects band/chans that are already enabled and that have bg in 
    # the specified bias groups
    bgmap_data = np.load(bgmap_file, allow_pickle=True).item()
    bs, cs, bgmap = [bgmap_data[k] for k in ['bands', 'channels', 'bgmap']]
    scale_array = np.array([S.get_amplitude_scale_array(b) for b in range(8)])
    m = (scale_array[bs, cs] > 0) & np.in1d(bgmap, bias_groups)
    bands = bs[m]
    channels = cs[m]
    chan_bgs =bgmap[m]

    start_times = []
    stop_times = []
    sids = []

    # Record initial params so we can restore later
    bcfg = cfg.dev.bands[0]
    init_fr_freq = bcfg['flux_ramp_rate_khz'] * 1e3
    init_nphi0 = np.round(
        bcfg['lms_freq_hz'] / init_fr_freq
    )
    init_biases = S.get_tes_bias_bipolar_array()

    # Setup tracking for all bands to accomodate for higher freq data
    if not skip_tracking_setup:
        S.log(f"Setting fr freqs to {flux_ramp_rate_khz} khz")
        tracking_setup_mult_bands(S, cfg, S._bands, fr_rate=flux_ramp_rate_khz,
                                  nphi0=3, show_plots=False,
                                  feedback_gain=feedback_gain)

    pb = tqdm(total=nsteps)

    init_ds_factor = S.get_downsample_factor()
    init_filter_disable = S.get_filter_disable()

    try:
        S.set_filter_disable(1)
        S.set_downsample_factor(1)

        # Sets all bias groups to high-current mode + lower voltage
        su.set_current_mode(S, bias_groups, 1)
        hcm_biases = S.get_tes_bias_bipolar_array()

        for bg in bias_groups:
            m = chan_bgs == bg
            channel_mask = bands[m] * 512 + channels[m]
            sid = so.stream_g3_on(S, tag='complex_impedance',
                                  channel_mask=channel_mask)
            sids.append(sid)
            S.log(f"Streaming Bias Group {bg} with sid {sid}")

            for freq in freqs:
                S.log(f"Tickle with bg={bg}, freq={freq}")
                S.play_sine_tes(bg, tickle_voltage, freq)
                start_times.append(time.time())
                sleep_time = min(meas_time, max_periods/freq)
                time.sleep(sleep_time)
                stop_times.append(time.time())
                S.set_rtm_arb_waveform_enable(0)
                S.set_tes_bias_bipolar(bg, hcm_biases[bg])
                pb.update(1)

            so.stream_g3_off(S)
    finally:
        S.log("Returning filter-disable and downsample factor to initial values")
        S.set_filter_disable(init_filter_disable)
        S.set_downsample_factor(init_ds_factor)

        su.set_current_mode(S, bias_groups, 0)

    summary = {
        'start_times': start_times,
        'stop_times': stop_times,
        'sids': sids,
        'freqs': freqs,
        'tickle_voltage': tickle_voltage,
        'bias_groups': bias_groups,
        'dc_biases': init_biases,
        'bands': bands,
        'channels': channels,
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

