import sodetlib as sdl
import numpy as np
import time
from pysmurf.client.base.smurf_control import SmurfControl

def check_packet_loss(Ss, cfgs, dur=10, fr_khz=4, nchans=2000, slots=None):
    """
    Takes a short G3 Stream on multiple slots simultaneously and checks for
    dropped samples. This function is strange since it requires simultaneous
    streaming on multiple slots to properly test, so it doesn't follow the
    standard sodetlib function / data format.

    Args
    -----
    Ss : dict[SmurfController]1
        Dict of pysmurf instances where the key is the slot-number
    cfgs : dict[DetConfig]
        Dict of DetConfigs where the key is the slot number
    dur : float
        Duration of data stream (sec)
    fr_khz : float
        Frequency of FR rate (khz)
    nchans : int
        Number of channels to stream
    slots : list
        Which slots to stream data on. If None, will stream on all slots in the
        Ss object.

    Returns
    --------
    ams : dict[AxisManagers]
        Dict of axis managers indexed by slot-number
    res : dict
        Dict where the key is the slot number, and the values are dicts
        containing frame counters, number of dropped frames, etc.
    """
    if slots is None:
        slots = Ss.keys()

    for s in slots:
        S = Ss[s]
        S.flux_ramp_setup(fr_khz, 0.4, band=0)
        sdl.stream_g3_on(
            S, channel_mask=np.arange(nchans), downsample_factor=1
        )

    time.sleep(dur)
    
    sids = {}
    for s in slots:
        sids[s] = sdl.stream_g3_off(Ss[s])

    ams = {}
    for s, sid in sids.items():
        ams[s] = sdl.load_session(cfgs[s].stream_id, sid)

    res = {}
    for s, am in ams.items():
        dropped_samps = np.sum(np.diff(am.primary['FrameCounter']) - 1)
        total_samps = len(am.primary['FrameCounter'])
        res[s] = {
            'sid': sids[s],
            'meta': sdl.get_metadata(Ss[s], cfgs[s]),
            'frame_counter': am.primary['FrameCounter'],
            'dropped_samples': dropped_samps,
            'dropped_frac': dropped_samps / total_samps,
        }

    return ams, res

@sdl.set_action()
def measure_bias_line_resistances(S: SmurfControl, cfg, vstep=0.001, bgs=None, sleep_time=2.0):
    if bgs is None:
        bgs = np.arange(12)
    bgs = np.atleast_1d(bgs)

    vbias = S.get_tes_bias_bipolar_array()
    vb_low = vbias.copy()
    vb_low[bgs] = 0
    vb_high = vbias.copy()
    vb_high[bgs] = vstep
    segs = []

    S.set_tes_bias_bipolar_array(vb_low)
    sdl.set_current_mode(S, bgs, 0, const_current=False)

    def take_step(bias_arr, sleep_time, wait_time=0.2):
        S.set_tes_bias_bipolar_array(bias_arr)
        t0 = time.time()
        time.sleep(sleep_time)
        t1 = time.time()
        return (t0, t1)

    sdl.stream_g3_on(S)
    time.sleep(0.5)

    segs.append(take_step(vb_low, sleep_time, wait_time=0.5))
    segs.append(take_step(vb_high, sleep_time, wait_time=0.5))

    S.set_tes_bias_bipolar_array(vb_low)
    time.sleep(0.5)
    sdl.set_current_mode(S, bgs, 1, const_current=False)

    segs.append(take_step(vb_low, sleep_time, wait_time=0.05))
    segs.append(take_step(vb_high, sleep_time, wait_time=0.05))

    sid = sdl.stream_g3_off(S)

    am = sdl.load_session(cfg.stream_id, sid)
    ts = am.timestamps
    sigs = []
    for (t0, t1) in segs:
        m = (t0 < ts) & (ts < t1)
        sigs.append(np.mean(am.signal[:, m], axis=1) * S.pA_per_phi0 / (2*np.pi))

    Rbl_low = vstep / (np.abs(sigs[1] - sigs[0]) * 1e-12)
    Rbl_high = vstep / (np.abs(sigs[3] - sigs[2]) * 1e-12)
    high_low_ratio = Rbl_low / Rbl_high

    cfg.dev.exp['bias_line_resistance'] = np.nanmedian(Rbl_low)
    cfg.dev.exp['high_low_current_ratio'] = np.nanmedian(high_low_ratio)
    cfg.dev.update_file()

    path = sdl.make_filename(S, 'measure_bias_line_info')
    data = {
        'Rbl_low_all': Rbl_low,
        'Rbl_high_all': Rbl_high,
        'high_low_ratio_all': high_low_ratio,
        'bias_line_resistance': np.nanmedian(Rbl_low),
        'high_current_mode_resistance': np.nanmedian(Rbl_high),
        'high_low_ratio': np.nanmedian(high_low_ratio),
        'sid': sid,
        'vstep': vstep,
        'bgs': bgs,
        'meta': sdl.get_metadata(S, cfg),
        'segs': segs,
        'path': path,
    }
    np.save(path, data, allow_pickle=True)
    S.pub.register_file(path, 'bias_line_stuff', format='npy')

    return am, data