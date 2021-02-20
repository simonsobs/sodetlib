import numpy as np
from scipy.optimize import curve_fit


def sine(ts, amp, phi, freq):
    return amp * np.sin(2*np.pi*freq*ts + phi)


def fit_sine(times, sig, freq, nperiods=6):
    """
    Finds the amplitude and phase of a sine wave of a given frequency.

    Args
    ----
    times: np.ndarray
        timestamp array [seconds]. Note that this fit will *not* work if
        timestamps are ctimes since they are too large to be accurately fit.
        For an accurate fit, make sure to shift so that times[0] is 0.
    sig: np.ndararay
        Signal array
    freq: float
        Frequency of sine wave [Hz]
    nperiods: float
        Number of periods to fit to. Limitting the number of periods usually
        results in a better fit due to glitches in longer timestreams.

    Returns
    -------
    amp: float
        Amplitude of sine wave (in whatever units the signal is in)
    phase: float
        Phase offset in radians
    """
    amp_guess = (np.max(sig) - np.min(sig)) / 2
    offset_guess = (np.max(sig) + np.min(sig)) / 2
    sig -= offset_guess
    (amp, phase), pcov = curve_fit(
        lambda *args: sine(*args, freq), times, sig,
        p0=(amp_guess, np.pi/4)
    )
    if amp < 0:
        amp = np.abs(amp)
        phase += np.pi
    phase %= 2*np.pi

    return amp, phase


def get_r2(sig, sig_hat):
    sst = np.sum((sig - sig.mean())**2)
    sse = np.sum((sig - sig_hat)**2)
    return 1 - sse / sst
