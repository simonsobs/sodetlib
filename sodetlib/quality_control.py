import sodetlib as sdl
import numpy as np
import time

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
        S.set_downsample_mode('internal')
        S.set_downsample_factor(1)
        sdl.stream_g3_on(S, channel_mask=np.arange(nchans))

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