"""
Module for miscellaneous functions and classes that are useful in many sodetlib
scripts.
"""
import numpy as np
from scipy import signal
import time


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
