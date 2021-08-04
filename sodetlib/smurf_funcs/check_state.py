"""
The goal of the check-state function is to determine of the system is in a
state where it is ready to take data. This will allow you to compare white
noise levels to a saved base_state, which will tell you if you're tuned,
but not necessarily if the detectors are biased to the correct state. For this
you still need to use a tickle function to make sure.

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
import os
import matplotlib.pyplot as plt

CHANS_PER_BAND = 512
NBANDS = 8
NCHANS = NBANDS * CHANS_PER_BAND


class ChannelState:
    """
    Container for channel specific information (kind of like a tune file)

    Args
    -----
        S : SmurfControl
            Pysmurf instance that can be used to set a timestamp and other
            pysmurf variables on creation
    """
    def __init__(self, S=None):
        self.save_path = None

        self.abs_chans = np.arange(NCHANS)
        self.bands = self.abs_chans // CHANS_PER_BAND
        self.chans = self.abs_chans % CHANS_PER_BAND
        self.enabled = np.zeros(NCHANS, dtype=int)
        self.wls = np.full(NCHANS, np.nan)
        self.band_medians = np.full(NBANDS, np.nan)
        self.timestamp = None

        if S is not None:
            self.timestamp = S.get_timestamp()
            for b in range(NBANDS):
                m = self.bands == b
                self.enabled[m] = (S.get_amplitude_scale_array(b) > 0)

    def set_channel_wls(self, am):
        """
        Sets the wls and band_medians arrays from an axis manager

        Args
        -----
            am : AxisManager
                Axis Manager with a noise timestream
        """
        wls, self.band_medians = get_wls_from_am(am)
        achans = CHANS_PER_BAND * am.ch_info.band + am.ch_info.channel
        self.wls[achans] = wls

    def save(self, path):
        """
        Saves the state to an npz file.

        Args
        -----
            path: path
                File where state is saved
        """
        general = {
            'timestamp': self.timestamp
        }
        np.savez(
            path, enabled=self.enabled, wls=self.wls,
            band_medians=self.band_medians, general=general
        )
        self.save_path = path

    @classmethod
    def from_file(cls, path):
        """
        Creates a ChannelState instance from a state npz file.

        Args
        -----
            path: path
                File where state is saved
        """
        self = cls()
        npz = np.load(path, allow_pickle=True)
        self.enabled = npz['enabled']
        self.wls = npz['wls']
        self.band_medians = npz['band_medians']
        self.save_path = path
        general = npz['general'].item()
        self.timestamp = general['timestamp']
        return self

    def desc(self):
        """
        Returns a short description of the channel state
        """
        desc = f"timestamp: {self.timestamp}\n"
        desc += f"channels enabled: {np.sum(self.enabled)}\n"
        desc += "Band medians:\n"
        for band, med in enumerate(self.band_medians):
            desc += f" - Band {band}: {med}\n"
        return desc


def compare_state_with_base(cfg, state):
    """
    Creates a plot comparing a specified state object with a base state

    Args
    -----
        cfg : DetConfig
            Sodetlib det-config object
        state : ChannelState
            state object to be compared with the base state.
    """
    base = get_base_state(cfg)
    fig, axes = plot_state_noise(state, alpha=0.5, state_label='current')
    plot_state_noise(base, alpha=0.5, axes=axes, state_idx=1, state_label='base')
    return fig, axes


def get_base_state(cfg):
    """
    Returns the base state specified in the device cfg

    Args
    -----
        cfg : DetConfig
            Sodetlib det-config object
    """
    base_state_file = cfg.dev.exp.get('channel_base_state_file')
    if base_state_file is None:
        raise Exception("No base state has been set!")
    base = ChannelState.from_file(base_state_file)
    return base


@set_action()
def set_base_state(S, cfg, state):
    """
    Sets a given state to be the "base state". It will be saved to the
    directory ``/data/so/channel_states`` with the same base-name and saved
    to the device config file. It is moved out of the smurf output directory
    to prevent it from accidentally being deleted with the rest of the smurf
    outputs.

    Args
    -----
        S : SmurfControl
            Pysmurf instance
        cfg : DetConfig
            Sodetlib det-config instance
        state : ChannelState
            State to set as the base
    """
    if state.save_path is None:
        path = make_filename(S, 'channel_state.npz')
        state.save(path)
        S.pub.register_file(path)

    state_dir = '/data/so/channel_states'
    if not os.path.exists(state_dir):
        os.makedirs(state_dir)

    new_path = os.path.join(state_dir, os.path.basename(state.save_path))
    state.save(new_path)
    cfg.dev.update_experiment({'channel_base_state_file': state.save_path},
                              update_file=True)


@set_action()
def get_channel_state(S, cfg):
    """
    Gets a channel state from a pysmurf control object. This will take a noise
    timestream and use that to generate the state object.

    Args
    -----
        S : SmurfControl
            Pysmurf instance
        cfg : DetConfig
            Sodetlib det-config instance
    """
    chan_state = ChannelState(S=S)

    print("Taking noise data for 30 seconds...")
    sid = smurf_ops.take_g3_data(S, 30)
    am = smurf_ops.load_session(cfg, sid)
    chan_state.set_channel_wls(am)

    path = make_filename(S, 'channel_state.npz')
    chan_state.save(path)
    S.pub.register_file(path, 'channel_state')

    return chan_state


def plot_state_noise(state, nbins=40, axes=None, alpha=1, state_idx=0, state_label=None):
    """
    Creates a plot displaying the channel noise levels for each band.

    Args
    -----
        state : ChannelState
            State to plot
        nbins : int
            Number of bins in each histogram
        axes : np.ndarray([Axis])
            If specified, will plot on existing set of axes instead of creating
            a new one.
        alpha : float
            Alpha value of hist
        state_idx : int
            Index of the state being plotted. This will change the color and
            position of the text box so two states can clearly be plotted
            on the same axes.
    """
    if axes is None:
        fig, axes = plt.subplots(4, 2, figsize=(16, 8),
                                 gridspec_kw={'hspace': 0})
        ymax = 0

    else:
        fig = axes[0][0].get_figure()
        ymax = 0
        for i in range(8):
            ax = axes[i%4, i//4]
            ymax = max(ymax, ax.get_ylim()[1] / 1.1)

    text_pos = [
        (0.02, 0.65), (0.7, 0.65)
    ]
    if state_label is None:
        state_label = str(state_idx)

    bbox_props = {
        'facecolor': f'C{state_idx}', 'alpha': 0.5
    }
    bins = np.logspace(1, 4, nbins)

    for band in range(8):
        ax = axes[band % 4, band // 4]
        m = state.bands == band
        hist = ax.hist(state.wls[m], bins=bins, alpha=alpha, color=f'C{state_idx}')
        ymax = max(np.max(hist[0]), ymax)

        med = np.nanmedian(state.wls[m])
        nchans = np.sum(hist[0])
        text = f"State: {state_label}\n"
        text += f"Median: {med:0.2f}\n"
        text += f"Chans pictured: {nchans:0.0f}"
        ax.text(*text_pos[state_idx], text, transform=ax.transAxes, bbox=bbox_props)
        ax.axvline(med, color=f'C{state_idx}')
        ax.set(xscale='log', ylabel=f'Band {band}')

    for band in range(8):
        ax = axes[band % 4, band // 4]
        ax.set(ylim=(0, ymax * 1.1))
    return fig, axes
