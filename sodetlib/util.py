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
        'lms_gain': band_cfg['lms_gain'],
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
