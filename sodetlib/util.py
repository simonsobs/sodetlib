"""
Module for miscellaneous functions and classes that are useful in many sodetlib
scripts.
"""
import numpy as np
from scipy import signal
import time
import os
from collections import namedtuple
from sodetlib import det_config

try:
    import epics
    from pysmurf.client.command.cryo_card import cmd_make
except Exception:
    pass


# This is a pysmurf constant
# Max bias voltage is 20 V, and there are 20 bits
# so this number is 20/2**20 = 1.907e-5
rtm_bit_to_volt = 1.9073486328125e-05
CHANS_PER_BAND = 512

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


def get_tracking_kwargs(S, cfg, band, kwargs=None):
    band_cfg = cfg.dev.bands[band]
    tk = {
        'reset_rate_khz': band_cfg['flux_ramp_rate_khz'],
        'lms_freq_hz': band_cfg['lms_freq_hz'],
        'lms_gain': band_cfg.get('lms_gain', 0),
        'fraction_full_scale': band_cfg['frac_pp'],
        'make_plot': True, 'show_plot': True, 'channel': [],
        'nsamp': 2**18, 'return_data': True,
        'feedback_start_frac': 0.02,
        'feedback_end_frac': 0.94,
        'return_data': True}
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
            drive = drives[np.argmax(counts)]
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

    def get(self):
        return self.S._caget(self.addr)

    def set(self, val):
        self.S._caput(self.addr, val)

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
        'debug_data': _sostream + "DebugData",
        'debug_meta': _sostream + "DebugMeta",
        'debug_builder': _sostream + "DebugBuilder",
        'flac_level': _sostream + "FlacLevel",
        'source_enable': _source_root + 'SourceEnable',
        'enable_compression': _sostream + 'EnableCompression',
    }

    def __init__(self, S):
        self.S = S
        for name, reg in self._registers.items():
            setattr(self, name, _Register(S, reg))


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


    # It takes longer for DC voltages to settle than it does to toggle the
    # high-current relay, so we can set them at the same time when switchign
    # to hcm, but when switching to lcm we need a sleep statement to prevent
    # dets from latching.
    if mode:
        epics.caput_many([S.C.writepv, dac_data_reg], [relay_data, dac_data],
                         wait=True)
    else:
        S._caput(dac_data_reg, dac_data)
        time.sleep(0.04)
        S._caput(S.C.writepv, relay_data)

    time.sleep(0.1)  # Just to be safe

