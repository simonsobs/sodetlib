"""
Module for miscellaneous functions and classes that are useful in many sodetlib
scripts.
"""
import numpy as np
from scipy import signal
import time
import os
from collections import namedtuple
import sodetlib
from sodetlib import det_config
from sodetlib.constants import *
from sotodlib.tod_ops.fft_ops import calc_psd
from sotodlib.core import AxisManager

if not os.environ.get('NO_PYSMURF', False):
    try:
        import epics
        import pysmurf
        from pysmurf.client.command.cryo_card import cmd_make
    except Exception:
        os.environ['NO_PYSMURF'] = True

StreamSeg = namedtuple("StreamSeg", "times sig mask")

class TermColors:
    HEADER = '\n\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def cprint(msg, style=TermColors.OKBLUE):
    if style is True:
        style = TermColors.OKGREEN
    elif style is False:
        style = TermColors.FAIL
    print(f"{style}{msg}{TermColors.ENDC}")


def make_filename(S, name, ctime=None, plot=False):
    """
    Creates a timestamped filename in the pysmurf outputs or plot directory.
    """
    if ctime is None:
        ctime = S.get_timestamp()

    if plot:
        ddir = S.plot_dir
    else:
        ddir = S.output_dir

    return os.path.join(ddir, f'{ctime}_{name}')


def _encode_data(data):
    """
    Encodes a data object into one that is serializable with
    json so that it can be sent over UDP.
    """
    if isinstance(data, list):
        return [_encode_data(d) for d in data]
    elif isinstance(data, dict):
        return {str(k): _encode_data(d) for k, d in data.items()}
    elif isinstance(data, np.ndarray):
        return data.tolist()
    else:
        return data


def set_session_data(S, key, val):
    """
    Adds data to the OCS Session data object. If this is being run directly
    from an OCS PysmurfController operation, the _ocs_session object will be
    set in the pysmurf instance. If not, publish message so pysmurf monitor can
    relay data to the correct controller agent.
    """
    session = getattr(S, '_ocs_session', None)
    if session is not None:
        session.data[key] = _encode_data(val)
    else:
        S.pub.publish(_encode_data({key: val}), msgtype='session_data')



def pub_ocs_log(S, msg, log=True):
    """
    Passes a string to the OCS pysmurf controller to be logged to be passed
    around the OCS network.
    """
    if log:
        S.log(msg)

    session = getattr(S, '_ocs_session', None)
    if session is not None:
        session.add_message(msg)

    S.pub.publish(msg, msgtype='session_log')


class OCSAbortError(Exception):
    """Error that's thrown when an operation is aborted through OCS""" 


def stop_point(S):
    """
    This function will throw an error if the OCS session status is 'stopping'.
    This allows the pysmurf controller to gracefully tell long-running sodetlib
    functions to stop when possible without killing the agent.
    """
    session = getattr(S, '_ocs_session', None)
    if session is not None:
        if session.status == 'stopping':
            raise OCSAbortError()



def load_bgmap(bands, channels, bgmap_file):
    """
    Loads bias-group assignments and channel polarity for
    set of channels.

    Args
    ----
    bands : np.ndarray
        Array of len <nchans> containing the smurf-band of each channel
    channels : np.ndarray
        Array of len <nchans> containing the smurf-channel of each channel
    bgmap_file : str
        Path to the bgmap file.

    Returns
    --------
    bgmap : np.ndarray
        Array of lenght <nchans> containing bias-group assignments of each
        channel. Unassigned channels contain -1.
    polarity : np.ndarray
        Array of lenght <nchans> containing the polarity of each channel,
        or whether the squid current steps up or down with bias-current.
    """
    bgmap_full = np.load(bgmap_file, allow_pickle=True).item()
    idxs = map_band_chans(
        bands, channels, bgmap_full['bands'], bgmap_full['channels']
    )
    bgmap = bgmap_full['bgmap'][idxs]
    polarity = bgmap_full['polarity'][idxs]
    bgmap[idxs == -1] = -1

    return bgmap, polarity


def map_band_chans(b1, c1, b2, c2, chans_per_band=512):
    """
    Returns an index mapping of length nchans1 from one set of bands and
    channels to another. Note that unmapped indices are returned as -1, so this
    must be handled before (or after) indexing or else you'll get weird
    results.

    Args
    ----
        b1 (np.ndarray):
            Array of length nchans1 containing the smurf band of each channel
        c1 (np.ndarray):
            Array of length nchans1 containing the smurf channel of each channel
        b2 (np.ndarray):
            Array of length nchans2 containing the smurf band of each channel
        c2 (np.ndarray):
            Array of length nchans2 containing the smurf channel of each channel
        chans_per_band (int):
            Lets just hope this never changes.
    """
    acs1 = b1 * chans_per_band + c1
    acs2 = b2 * chans_per_band + c2

    mapping = np.full_like(acs1, -1, dtype=int)
    for i, ac in enumerate(acs1):
        x = np.where(acs2 == ac)[0]
        if len(x > 0):
            mapping[i] = x[0]

    return mapping


def get_metadata(S, cfg):
    """
    Gets collection of metadata from smurf and detconfig instance that's
    useful for pretty much everything. This should be included in all output
    data files created by sodetlib.
    """
    return {
        'tunefile': getattr(S, 'tune_file', None),
        'high_low_current_ratio': S.high_low_current_ratio,
        'R_sh': S.R_sh,
        'pA_per_phi0': S.pA_per_phi0,
        'rtm_bit_to_volt': S._rtm_slow_dac_bit_to_volt,
        'bias_line_resistance': S.bias_line_resistance,
        'chans_per_band': S.get_number_channels(),
        'high_current_mode': get_current_mode_array(S),
        'timestamp': time.time(),
        'stream_id': cfg.stream_id,
        'action': S.pub._action,
        'action_timestamp': S.pub._action_ts,
        'bgmap_file': cfg.dev.exp.get('bgmap_file'),
        'iv_file': cfg.dev.exp.get('iv_file'),
        'v_bias': S.get_tes_bias_bipolar_array(),
        'pysmurf_client_version': pysmurf.__version__,
        'rogue_version': S._caget(f'{S.epics_root}:AMCc:RogueVersion'),
        'smurf_core_version': S._caget(f'{S.epics_root}:AMCc:SmurfApplication:SmurfVersion'),
        'sodetlib_version': sodetlib.__version__,
        'fpga_git_hash': S.get_fpga_git_hash_short(),
        'cryocard_fw_version': S.C.get_fw_version(),
        'crate_id': cfg.sys['crate_id'],
        'slot': cfg.slot,
    }


def validate_and_save(fname, data, S=None, cfg=None, register=True,
                      make_path=True):
    """
    Does some basic data validation for sodetlib data files to make sure
    they are normalized based on the following  rules:
        1. The 'meta' key exists storing metadata from smurf and cfg instances
           such as the stream-id, pysmurf constants like high-low-current-ratio
           etc., action and timestamp and more. See the get_metadata function
           definition for more.
        2. 'bands', and 'channels' arrays are defined, specifying the band/
           channel combinations for data taking / analysis products.
        3. 'sid' field exists to contain one or more session-ids for the
           analysis.

    Args
    ----
        fname (str):
            Filename or full-path of datafile. If ``make_path`` is set to True,
            this will be turned into a file-path using the ``make_filename``
            function. If not, this will be treated as the save-file-path
        data (dict):
            Data to be written to disk. This should contain at least the keys
            ``bands``, ``channels``, ``sid``, along with any other data
            products. If the key ``meta`` does not already exist, metadata
            will be populated using the smurf and det-config instances.
        S (SmurfControl):
            Pysmurf instance. This must be set if registering data file,
            ``meta`` is not yet set, or ``make_path`` is True.
        cfg (DetConfig):
            det-config instance. This must be set if registering data file,
            ``meta`` is not yet set, or ``make_path`` is True.
        register (bool):
            If True, will register the file with the pysmurf publisher
        make_path (bool):
            If true, will create a path by passing fname to the function
            make_filename (putting it in the pysmurf output directory)
    """
    if register or make_path or 'meta' not in data:
        if S is None or cfg is None:
            raise ValueError("Pysmurf and cfg instances must be set")

    _data = data.copy()
    if 'meta' not in data:
        meta = get_metadata(S, cfg)
        _data['meta'] = meta

    for k in ['channels', 'bands', 'sid']:
        if k not in data:
            raise ValueError(f"Key '{k}' is required in data")

    if make_path and S is not None:
        path = make_filename(S, fname)
    else:
        path = fname

    np.save(path, _data, allow_pickle=True)
    if S is not None and register:
        S.pub.register_file(path, 'sodetlib_data', format='npy')
    return path


def get_tracking_kwargs(S, cfg, band, kwargs=None):
    band_cfg = cfg.dev.bands[band]
    tk = {
        'reset_rate_khz': cfg.dev.exp['flux_ramp_rate_khz'],
        'lms_freq_hz': band_cfg['lms_freq_hz'],
        'lms_gain': band_cfg['lms_gain'],
        'fraction_full_scale': band_cfg['frac_pp'],
        'make_plot': True,
        'show_plot': True,
        'channel': [],
        'nsamp': 2**18,
        'return_data': True,
        'feedback_start_frac': cfg.dev.exp['feedback_start_frac'],
        'feedback_end_frac': cfg.dev.exp['feedback_end_frac'],
        'feedback_gain': band_cfg['feedback_gain'],
    }
    if kwargs is not None:
        tk.update(kwargs)
    return tk

def get_psd(S, times, phases, detrend='constant', nperseg=2**12, fs=None):
    """
    Returns PSD for all channels.
    Args:
        S:
            pysmurf.SmurfControl object
        times: np.ndarray
            timestamps (in ns)
        phases: np.ndarray
            Array of phases
        detrend: str
            Detrend argument to pass to signal.welch
        nperseg: int
            nperseg arg for signal.welch
        fs: float
            sample frequency for signal.welch. If None will calculate using the
            timestamp array.
    Returns:
        f: np.ndarray
            Frequencies
        Pxx: np.ndarray
            PSD in pA/sqrt(Hz)
    """
    if fs is None:
        fs = 1/np.diff(times/1e9).mean()
    current = phases * S.pA_per_phi0 / (2 * np.pi)
    f, Pxx = signal.welch(current, detrend=detrend, nperseg=nperseg, fs=fs)
    Pxx = np.sqrt(Pxx)
    return f, Pxx

def get_asd(am, pA_per_phi0=9e6, **psd_kwargs):
    """
    Returns ASD (sqrt(PSD)) for all channels.

    Args
    ----
    am: AxisManager
        timestamps (in ns)
    pA_per_phi0: float
        Conversion from phi_0 to pA, defaults to 9e6 set by the mux chip
        mutual inductance between TES to SQUID.
    psd_kwargs: dictionary
        keyword arguments taken by scipy.welch function.

    Returns:
        f: np.ndarray
            Frequencies
        Axx: np.ndarray
            ASD in pA/sqrt(Hz)
    """
    f, Pxx = calc_psd(am, **psd_kwargs)
    Axx = np.sqrt(Pxx)*pA_per_phi0/(2 * np.pi)
    return f, Axx


class SectionTimer:
    def __init__(self):
        self.sections = []
        self.start_time = None
        self.stop_time = None

    def start_section(self, name):
        if self.start_time is None:
            self.start_time = time.time()
        self.sections.append((time.time(), name))

    def stop(self):
        self.stop_time = time.time()
        self.sections.append((time.time(), 'STOP'))

    def reset(self):
        self.sections = []
        self.start_time = None
        self.stop_time = None

    def summary(self):
        out = "="*80 + '\nTiming Summary\n' + '-'*80 + '\n'
        out += f"Total time: {self.stop_time - self.start_time} sec\n"
        out += 'name\tdur\tstart\n' + '='*80 + '\n'

        name_len = max([len(name) for t, name in self.sections])

        for i in range(len(self.sections) - 1):
            t, name = self.sections[i]
            dur = self.sections[i+1][0] - t
            out += f'{name:{name_len}s}\t{dur:.2f}\t{t:.0f}\n'

        return out


def dev_cfg_from_pysmurf(S, save_file=None, clobber=True):
    """
    Creates a populated device cfg object from a fully tuned pysmurf instance.
    If a save-file is specifed, the device config file will be written there.
    By default this will not save the device config to a file!! If you want
    overwrite the currently used device cfg, you can run::

        dev_cfg_from_pysmurf(S, save_file=cfg.dev_file, clobber=True)

    Args
    ----
    S : SmurfControl object
        The pysmurf instance should be in a state where a tunefile is loaded,
        attenuations, biases, and any other parameter are already set
        correctly.
    save_file : path
        Path to save-file location. Remember that if you are running in a
        docker container, you have to give the path as it is inside the
        container. For example, the OCS_CONFIG_DIR is mapped to /config inside
        the docker.
    clobber : bool
        If true, will overwrite the save_file if one already exists at that
        location.
    """
    dev = det_config.DeviceConfig()

    # Experiment setup
    amp_biases = S.get_amplifier_biases()
    if hasattr(S, 'tune_file'):
        tunefile = S.tune_file
    else:
        cprint("No tunefile is loaded! Loading tunefile=None", False)
        tunefile = None
    dev.exp.update({
        'amp_50k_Id': amp_biases['50K_Id'],
        'amp_50k_Vg': amp_biases['50K_Vg'],
        'amp_hemt_Id': amp_biases['hemt_Id'],
        'amp_hemt_Vg': amp_biases['hemt_Vg'],
        'tunefile': tunefile,
        'bias_line_resistance': S._bias_line_resistance,
        'high_low_current_ratio': S._high_low_current_ratio,
        'pA_per_phi0': S._pA_per_phi0,
    })

    # Right now not getting any bias group info
    for band in S._bands:
        tone_powers = S.get_amplitude_scale_array(band)[S.which_on(band)]
        if len(tone_powers) == 0:
            drive = S._amplitude_scale[band]
            cprint(f"No channels are on in band {band}. Setting drive to "
                   f"pysmurf-cfg value: {drive}", style=TermColors.WARNING)
        else:
            drives, counts = np.unique(tone_powers, return_counts=True)
            drive = drives[np.nanargmax(counts)]
            if len(drives) > 1:
                print(f"Multiple drive powers exist for band {band} ({drives})!")
                print(f"Using most common power: {drive}")

        feedback_start_frac = S._feedback_to_feedback_frac(band, S.get_feedback_start(band))
        feedback_end_frac = S._feedback_to_feedback_frac(band, S.get_feedback_end(band))

        flux_ramp_rate_khz = S.get_flux_ramp_freq()
        lms_freq_hz = S.get_lms_freq_hz(band)
        nphi0 = np.round(lms_freq_hz / flux_ramp_rate_khz / 1e3)

        dev.bands[band].update({
            'uc_att': S.get_att_uc(band),
            'dc_att': S.get_att_dc(band),
            'drive': drive,
            'feedback_start_frac': feedback_start_frac,
            'feedback_end_frac': feedback_end_frac,
            'lms_gain': S.get_lms_gain(band),
            'frac_pp': S.get_fraction_full_scale(),
            'flux_ramp_rate_khz': flux_ramp_rate_khz,
            'lms_freq_hz': lms_freq_hz,
            'nphi0': nphi0
        })

    if save_file is not None:
        if clobber and os.path.exists(save_file):
            print(f"Rewriting existing file: {save_file}")
        dev.dump(save_file, clobber=clobber)
    return dev


def get_wls_from_am(am, nperseg=2**16, fmin=10., fmax=20., pA_per_phi0=9e6):
    """
    Gets white-noise levels for each channel from the axis manager returned
    by smurf_ops.load_session.

    Args
    ----
    am : AxisManager
        Smurf data returned by so.load_session or the G3tSmurf class
    nperseg : int
        nperseg to be passed to welch
    fmin : float
        Min frequency to use for white noise mask
    fmax : float
        Max freq to use for white noise mask
    pA_per_phi0 : float
        S.pA_per_phi0 unit conversion. This will eventually make its way
        into the axis manager, but for now I'm just hardcoding this here
        as a keyword argument until we get there.

    Returns
    --------
    wls : array of floats
        Array of the white-noise level for each channel, indexed by readout-
        channel number
    band_medians : array of floats
        Array of the median white noise level for each band.
    """
    fsamp = 1./np.median(np.diff(am.timestamps))
    fs, pxx = signal.welch(am.signal * pA_per_phi0 / (2*np.pi),
                           fs=fsamp, nperseg=nperseg)
    pxx = np.sqrt(pxx)
    fmask = (fmin < fs) & (fs < fmax)
    wls = np.median(pxx[:, fmask], axis=1)
    band_medians = np.zeros(8)
    for i in range(8):
        m = am.ch_info.band == i
        band_medians[i] = np.median(wls[m])
    return wls, band_medians


# useful functions for analysis etc.
def invert_mask(mask):
    """
    Converts a readout mask from (band, chan)->rchan form to rchan->abs_chan
    form.
    """
    bands, chans = np.where(mask != -1)
    maskinv = np.zeros_like(bands, dtype=np.int16)
    for b, c in zip(bands, chans):
        maskinv[mask[b, c]] = b * CHANS_PER_BAND + c
    return maskinv


def get_r2(sig, sig_hat):
    """ Gets r-squared value for a signal"""
    sst = np.sum((sig - sig.mean())**2)
    sse = np.sum((sig - sig_hat)**2)
    r2 = 1 - sse / sst
    if r2 < 0:
        return 0
    return r2


class _Register:
    def __init__(self, S, addr):
        self.S = S
        self.addr = S.epics_root + ":" + addr

    def get(self, **kw):
        return self.S._caget(self.addr, **kw)

    def set(self, val, **kw):
        self.S._caput(self.addr, val, **kw)

class Registers:
    """
    Utility class for storing, getting, and setting SO-rogue registers even if
    they are not in the standard rogue tree, or settable by existing pysmurf
    get/set functions
    """
    _root = 'AMCc:'
    _processor = _root + "SmurfProcessor:"
    _sostream = _processor + "SOStream:"
    _sofilewriter = _sostream + 'SOFileWriter:'
    _source_root = _root + 'StreamDataSource:'

    _registers = {
        'pysmurf_action': _sostream + 'pysmurf_action',
        'pysmurf_action_timestamp': _sostream + "pysmurf_action_timestamp",
        'stream_tag': _sostream + "stream_tag",
        'open_g3stream': _sostream + "open_g3stream",
        'g3_session_id': _sofilewriter + "session_id",
        'g3_filepath': _sofilewriter + "filepath",
        'debug_data': _sostream + "DebugData",
        'debug_meta': _sostream + "DebugMeta",
        'debug_builder': _sostream + "DebugBuilder",
        'flac_level': _sostream + "FlacLevel",
        'source_enable': _source_root + 'SourceEnable',
        'enable_compression': _sostream + 'EnableCompression',
        'agg_time': _sostream + 'AggTime',
    }

    def __init__(self, S):
        self.S = S
        for name, reg in self._registers.items():
            setattr(self, name, _Register(S, reg))


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


def set_current_mode(S, bgs, mode, const_current=True):
    """
    Sets one or more bias lines to high current mode. If const_current is True,
    will also update the DC bias voltages to try and preserve current based on
    the set high_low_current_ratio. We need this function to replace the
    existing pysmurf function so we can set both PV's in a single epics call,
    minimizing heating on the cryostat.

    This function will attempt to check the existing rogue relay state to
    determine if the voltages need to be updated or not. This may not work all
    of the time since there is no relay readback for the high-current-mode
    relays.  That means this will set the voltages incorrectly if there is
    somehow an inconsistency between the rogue relay state and the real relay
    state.

    Args
    ----
        S : SmurfControl
            Pysmurf control instance
        bgs : (int, list)
            Bias groups to switch to high-current-mode
        mode : int
            1 for high-current, 0 for low-current
        const_current : bool
            If true, will adjust voltage values simultaneously based on
            S.high_low_current_ratio
    """

    bgs = np.atleast_1d(bgs).astype(int)

    # DO this twice bc pysmurf does for some reason
    old_relay = S.get_cryo_card_relays()
    old_relay = S.get_cryo_card_relays()
    new_relay = np.copy(old_relay)

    old_dac_volt_arr = S.get_rtm_slow_dac_volt_array()
    new_dac_volt_arr = np.copy(old_dac_volt_arr)

    for bg in bgs:
        # Gets relay bit index for bg
        idx = np.where(S._pic_to_bias_group[:, 1] == bg)[0][0]
        # Bit in relay mask corresponding to this bg's high-current
        r = S._pic_to_bias_group[idx, 0]

        # Index of bg's DAC pair
        pair_idx = np.where(S._bias_group_to_pair[:, 0] == bg)[0][0]
        pos_idx = S._bias_group_to_pair[pair_idx, 1] - 1
        neg_idx = S._bias_group_to_pair[pair_idx, 2] - 1
        if mode:
            # sets relay bit to 1
            new_relay = new_relay | (1 << r)

            # if const_current and the old_relay bit was zero, divide bias
            # voltage by high_low_ratio
            if const_current and not (old_relay >> r) & 1:
                new_dac_volt_arr[pos_idx] /= S.high_low_current_ratio
                new_dac_volt_arr[neg_idx] /= S.high_low_current_ratio
        else:
            # Sets relay bit to 0
            new_relay = new_relay & ~(1 << r)

            # if const_current and the old_relay bit was one, mult bias
            # voltage by high_low_ratio
            if const_current and (old_relay >> r) & 1:
                new_dac_volt_arr[pos_idx] *= S.high_low_current_ratio
                new_dac_volt_arr[neg_idx] *= S.high_low_current_ratio

    relay_data = cmd_make(0, S.C.relay_address, new_relay)

    # Sets DAC data values, clipping the data array to make sure they contain
    # the correct no. of bits
    dac_data = (new_dac_volt_arr / S._rtm_slow_dac_bit_to_volt).astype(int)
    nbits = S._rtm_slow_dac_nbits
    dac_data = np.clip(dac_data, -2**(nbits-1), 2**(nbits-1)-1)

    dac_data_reg = S.rtm_spi_max_root + S._rtm_slow_dac_data_array_reg


    if isinstance(S.C.writepv, str):
        cryocard_writepv = S.C.writepv
    else:
        cryocard_writepv = S.C.writepv.pvname

    # It takes longer for DC voltages to settle than it does to toggle the
    # high-current relay, so we can set them at the same time when switchign
    # to hcm, but when switching to lcm we need a sleep statement to prevent
    # dets from latching.
    if mode:
        epics.caput_many([cryocard_writepv, dac_data_reg], [relay_data, dac_data],
                         wait=True)
    else:
        S._caput(dac_data_reg, dac_data)
        time.sleep(0.04)
        S._caput(cryocard_writepv, relay_data)

    time.sleep(0.1)  # Just to be safe


def save_fig(S, fig, name, tag=''):
    """
    Saves figure to pysmurf plots directory and publishes result.

    Args
    -----
    S : SmurfControl
        Pysmurf instantc
    fig : Figure
        Matplotlib figure to save
    name : Figure name
        This will be passed to ``make_filename`` so it is timestamped
        and placed in the pysmurf plots directory.
    """
    fname = make_filename(S, name, plot=True)
    fig.savefig(fname)
    S.pub.register_file(fname, tag, plot=True)
    return fname


################################################################################
# AxisManager Utility Functions
################################################################################

class RestrictionException(Exception):
    """Exception for when cannot restrict AxisManger properly"""

def restrict_to_times(am, t0, t1, in_place=False):
    """
    Restricts axis manager to a time range (t0, t1)
    """
    m = (t0 < am.timestamps) & (am.timestamps < t1)
    if not m.any():
        raise RestrictionException
    i0, i1 = np.where(m)[0][[0, -1]] + am.samps.offset
    return am.restrict('samps', (i0, i1), in_place=in_place)

def dict_to_am(d, skip_bad_types=False):
    """
    Attempts to convert a dictionary into an axis manager. This can be used on
    dicts containing basic types such as (str, int, float) along with numpy
    arrays. The AxisManager will not have any structure such as "axes", but
    this is useful if you want to nest semi-arbitrary data such as the "meta"
    dict into an axis manager.

    Args
    -----
    d : dict
        Dict to convert ot axismanager
    skip_bad_types : bool
        If True, will skip any value that is not a str, int, float, or
        np.ndarray. If False, this will raise an error for invalid types.
    """
    allowed_types = (str, int, float, np.ndarray)
    am = AxisManager()
    for k, v in d.items():
        if isinstance(v, dict):
            am.wrap(k, dict_to_am(v))
        if isinstance(v, allowed_types):
            am.wrap(k, v)
        elif not skip_bad_types:
            raise ValueError(
                f"Key {k} is of type {type(v)} which cannot be wrapped by an "
                 "axismanager")
    return am

def remap_dets(src, dst, load_axes=False, idxmap=None):
    """
    This function takes in two axis-managers, and returns a copy of the 2nd
    which is identical except all fields aligned with 'dets' are remapped so they
    match up with the 'dets' field of the first array. This is very useful if you
    want to compare data across tunes, or across cryostats. An additional field
    "unmapped" of shape ``(dets, )`` will be added containing mask for which
    channels were unmapped by the idxmap.

    Args
    ----
    src : AxisManager
        "source" data. A copy of this data will be returned, with the 'dets'
        axis remapped to match those of the dest AxisManager
    dst : AxisManager
        "destination" data. The 'dets' axis will be taken from this AxisManager.
    load_axes : bool
        If True, will add ``src`` axes into the copy, and copy the axis
        assignments. Note that if you are trying to nest this in another
        axismanager, this may cause trouble if 'src', and 'dst' have 
        axes with the same name but different values. If this is false,
        only the 'dets' assignment will be used, allowing for safe nesting.
    idxmap : optional, np.ndarry
        If an idxmap is passed, that will be used to remap the dets axis.
        If none is passed, an idxmap will be created to match up bands and
        channels of the src and dst. This might be useful if you want to
        map across cooldowns based on a detmap.
    """
    if idxmap is None:
        idxmap = map_band_chans(
            dst.bands, dst.channels,
            src.bands, src.channels,
        )

    axes = [dst.dets]
    if load_axes:
        axes.extend([v for k, v in src._axes.items() if k != 'dets'])

    am_new = AxisManager(*axes)
    for k, axes in src._assignments.items():
        f = src._fields[k]
        if isinstance(f, AxisManager):
            am_new.wrap(k, remap_dets(f, dst, load_axes=load_axes, idxmap=idxmap))
        elif isinstance(f, np.ndarray):
            assignments = []
            if load_axes:
                assignments.extend([
                    (i, n) for i, n in enumerate(axes) 
                    if n is not None and n!='dets'
                ])
            if 'dets' in axes:
                i = axes.index('dets')
                f = np.rollaxis(np.rollaxis(f, i)[idxmap], -i)
            am_new.wrap(k, f, assignments)
        else:  # Literals
            am_new.wrap(k, f)

    am_new.wrap('unmapped', idxmap == -1, [(0, 'dets')])
    return am_new


def overbias_dets(S, cfg, bias_groups=None, biases=None, cool_wait=None,
                  high_current_mode=False):
    """
    Overbiases detectors based on parameters in the device cfg.

    Args
    ------
    S : SmurfControl
        Pysmurf control instance
    cfg : DetConfig
        DetConfig instance
    bias_groups : list
        List of bias groups to overbias
    biases: list
        Array of final bias values to set after overbiasing. This must
        be an array of length 12 where biases[bg] is the bias of bias group bg.
        If this is not set, the ``cool_voltage`` set in the device cfg file
        will be used for each bias group.
    cool_wait : float
        If set, time to wait after overbiasing dets. Defaults to no wait time.
    high_current_mode : bool, List[bool]
        Whether the bias groups should be left in high-current-mode. If a list is
        passed, it should be an array of len 12 where high_current_mode[bg] is
        the desired mode for bias line ``bg``.
    """
    if bias_groups is None:
        bias_groups = cfg.dev.exp['active_bgs']

    if isinstance(high_current_mode, (bool, int)):
        high_current_mode = [high_current_mode for _ in range(12)]
    high_current_mode = np.atleast_1d(high_current_mode)

    S.log("Overbiasing Detectors")
    set_current_mode(S, bias_groups, 1, const_current=False)
    for bg in bias_groups:
        S.set_tes_bias_bipolar(bg, cfg.dev.bias_groups[bg]['overbias_voltage'])

    wait_time = cfg.dev.exp['overbias_wait']
    S.log(f"Waiting at ob volt for {wait_time} sec")
    time.sleep(wait_time)

    for bg in bias_groups:
        if biases is not None:
            bias = biases[bg]
        else:
            bias = cfg.dev.bias_groups[bg]['cool_voltage']
        S.set_tes_bias_bipolar(bg, bias)

    lcm_bgs = np.where(~high_current_mode)[0]
    set_current_mode(S, lcm_bgs, 0, const_current=False)

    if cool_wait is not None:
        time.sleep(cool_wait)

    return S.get_tes_bias_bipolar_array()
