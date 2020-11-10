"""
Module for miscellaneous functions and classes that are useful in many sodetlib
scripts.
"""
import numpy as np
from scipy import signal


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
        fs = 1/np.diff(times).mean()
    current = phases * S.pA_per_phi0 / (2 * np.pi)
    f, Pxx = signal.welch(current, detrend=detrend, nperseg=nperseg, fs=fs)
    Pxx = np.sqrt(Pxx)
    return f, Pxx
