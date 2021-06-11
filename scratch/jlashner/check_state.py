"""
The goal of the check-state function is to determine of the system is in a
state where it is ready to take data. This will primarily be decided based on
white-noise levels.

Since the noise level and yield will presumably be different for each system,
this will require the existance of a `channel_state` configuration file which
details channel-specific parameters such as noise level, tracking info,
bias-group info, etc.

The path to a recent channel-state variable will be stored in the device cfg
dict with the `set_channel_state` dict by an experienced smurf-user, which
will serve as the "base" channel info for a single cooldown/device. Any user
can then run the `check_channel_state` function, which will get the current
channel state and compare with the set-state, which will allow one to easily
determine if they are ready to start streaming
"""
from pysmurf.client.util.pub import set_action
from sodetlib.smurf_funcs import smurf_ops
from sodetlib.util import get_wls_from_am, make_filename
import numpy as np

CHANS_PER_BAND = 512


class ChannelState:
    """
    Container for channel specific information (kind of like a tune file)
    """
    def __init__(self, S=None):
        self.S = S
        self.save_path = None

        self.abs_chans = np.arange(8*CHANS_PER_BAND)
        self.bands = self.abs_chans // CHANS_PER_BAND
        self.chans = self.abs_chans % CHANS_PER_BAND
        self.enabled = np.zeros(len(self.abs_chans), dtype=int)
        self.wls = np.full(np.nan, len(self.abs_chans))
        self.timestamp = None

        if S is not None:
            self.timestamp = S.get_timestamp()
            for b in range(8):
                m = self.bands == b
                self.enabled[m] = (S.get_amplitude_scale_array(b) > 0)

    def set_channel_wls(self, am):
        wls = get_wls_from_am(am)
        achans = CHANS_PER_BAND * am.ch_info.band + am.ch_info.channel
        self.wls[achans] = wls

    def save(self, path):
        general = {
            'timestamp': self.timestamp
        }
        np.savez(path, enabled=self.enabled, wls=self.wls, general=general)
        self.save_path = path

    @classmethod
    def from_file(cls, S, cfg, path):
        self = cls()
        npz = np.load(path, allow_pickle=True)
        self.enabled = npz['enabled']
        self.wls = npz['wls']
        self.save_path = path
        general = npz['general'].item()
        self.timestamp = general['timestamp']

    def desc(self):
        desc = f"""
timestamp: {self.timestamp}
channels enabled: {np.sum(self.enabled)}
median noise:
    Band 0: {np.nanmedian(self.wls[self.bands == 0])}
    Band 1: {np.nanmedian(self.wls[self.bands == 1])}
    Band 2: {np.nanmedian(self.wls[self.bands == 2])}
    Band 3: {np.nanmedian(self.wls[self.bands == 3])}
    Band 4: {np.nanmedian(self.wls[self.bands == 4])}
    Band 5: {np.nanmedian(self.wls[self.bands == 5])}
    Band 6: {np.nanmedian(self.wls[self.bands == 6])}
    Band 7: {np.nanmedian(self.wls[self.bands == 7])}
"""
        return desc


@set_action()
def check_channel_state(S, cfg, state=None):
    """
    Checks the the current channel state against the one saved in the device
    config and lets you know major differences.
    """
    if state is None:
        state = get_channel_state(S, cfg)


@set_action()
def set_channel_state(S, cfg, state):
    cfg.dev.update_experiment({'channel_state_file': state.save_path})

@set_action()
def get_channel_state(S, cfg, save=True):
    chan_state = ChannelState(S=S)
    sid = smurf_ops.take_g3_data(S, 30)
    print("Taking noise data")
    am = smurf_ops.load_session(cfg, sid)
    chan_state.set_channel_wls(am)
    path = make_filename(S, 'channel_state.npz')
    chan_state.save(path)
    S.pub.register_file(path)
    return chan_state
