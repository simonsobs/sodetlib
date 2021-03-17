import numpy as np
import os
from scipy.optimize import curve_fit
from collections import namedtuple
from sodetlib.util import cprint, make_filename
import matplotlib.pyplot as plt


CHANS_PER_BAND = 512


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


def analyze_biasgroup_data(times, sig, mask, freq):
    """
    Returns the fit amplitude, phase, and r-squared value for each channel in
    a biasgroup file.
    """
    mask = invert_mask(mask)
    amps = np.zeros_like(mask, dtype=np.float32)
    phases = np.zeros_like(mask, dtype=np.float32)
    r2s = np.zeros_like(mask, dtype=np.float32)

    times = (times - times[0]) / 1e9   # Converts to sec and removes offset
    for i, abschan in enumerate(mask):
        amps[i], phases[i] = fit_sine(times, sig[i], freq)
        sighat = sine(times, amps[i], phases[i], freq)
        r2s[i] = get_r2(sig[i], sighat)

    return mask, amps, phases, r2s


def predict_bg_index(amps, r2s, thresh=0.9):
    """
    Predicts bias group index based fit amplitudes and r-squared values.
    Predicted biasgroup will be the biasgroup with the largest response such
    that the r-squared is above a certain threshold.

    Will return -1 if no biasgroups are above the threshold.
    """
    possible_idxs = np.where(r2s > thresh)[0]
    if len(possible_idxs) == 0:
        return -1
    bg_idx = possible_idxs[np.argmax(amps[possible_idxs])]
    return bg_idx


def plot_tickle_summary(S, summary, save_dir=None):
    """
    Makes summary plots from tickle analysis results
    """
    bgs = summary['bg_assignments']
    res = summary['resistances']
    r2s = summary['rsquared']
    classes = summary['classifications']
    class_cmap = {
        "sc": "C0", "transition": "C1", "normal": "red", "no_tes": "black"
    }
    cs = np.array([class_cmap[c] for c in classes])

    for bg in np.unique(bgs):
        if bg == -1:
            continue
        fig, ax = plt.subplots()
        m = (bgs == bg)
        num_chans = np.sum(m)
        num_sc = np.sum((classes[m] == "sc"))
        num_trans = np.sum(classes[m] == "transition")
        num_normal = np.sum(classes[m] == "normal")
        ax.scatter(np.arange(num_chans), res[m]*1000, c=cs[m])
        txt = "\n".join([
            f"Total channels: {num_chans}",
            f"SC channels: {num_sc}",
            f"Transition channels: {num_trans}",
            f"Normal channels: {num_normal}"
        ])
        ax.text(0, 0.1, txt, alpha=0.9,
                bbox={'facecolor': 'white'})
        ax.set(title=f"Bias Group {bg}", ylabel="Resistance (mOhm)")
        if save_dir is not None:
            fname = os.path.join(
                save_dir, f"{S.get_timestamp()}_tickle_summary_bg{bg}.png"
            )
            fig.savefig(fname)
            S.pub.register_file(fname, "tickle_summary", plot=True)
        plt.close(fig)

    # Bias plots altogether
    fig, axes = plt.subplots(3, 4, figsize=(30, 15))
    for i, ax in enumerate(axes.flatten()):
        m = bgs == i
        scatter = ax.scatter(np.arange(np.sum(m)), res[m]*1000, c=r2s[m],
                             cmap='Reds', vmin=0, vmax=1)
        ax.set(title=f"Biasgroup {i}")

    axes[1, 0].set_ylabel("Resistance (mOhm)", fontsize=22)
    cbar = fig.colorbar(scatter, ax=axes)
    cbar.set_label("$R^2$ values")

    if save_dir is not None:
        fname = os.path.join(
            save_dir, f"{S.get_timestamp()}_tickle_summary_all_bg.png"
        )
        fig.savefig(fname)
        S.pub.register_file(fname, "tickle_summary", plot=True)

    fig.show()

def analyze_tickle_data(S, tickle_file, data=None, assignment_thresh=0.9,
                        normal_thresh=4e-3, sc_thresh=1e-5):
    tickle_info = np.load(tickle_file, allow_pickle=True).item()
    biasgroups = tickle_info['bias_groups']
    dat_files = tickle_info['dat_files']
    tickle_freq = tickle_info['tickle_freq']
    tickle_voltage = tickle_info['tone_voltage']
    R_sh = S.R_sh
    pA_per_phi0 = S.pA_per_phi0
    bias_line_resistance = S.bias_line_resistance

    if (R_sh is None) or (pA_per_phi0 is None):
        raise ValueError("Pysmurf not loaded with config properties!")

    DataSegment = namedtuple("DataSegment", "times sig mask")
    segs = []
    cprint(f"Reading in segmented data. Biasgroups used: {biasgroups}")
    if data is None:
        for f in dat_files:
            segs.append(DataSegment(*S.read_stream_data(f)))
    else:
        segs = [DataSegment(d) for d in data]

    # Fit amplitude and calculate rsquared for each bg segment
    cprint("Fitting sine wave for each biasgroup segment")
    mask = invert_mask(segs[0].mask)
    amps_full = np.zeros((len(mask), len(biasgroups)))
    r2s_full = np.zeros((len(mask), len(biasgroups)))
    for i, bg in enumerate(biasgroups):
        _chans, _amps, _phases, _r2s = analyze_biasgroup_data(*segs[i],
                                                              tickle_freq)
        amps_full[:, i] = _amps
        r2s_full[:, i] = _r2s

    # Assigns bias groups, calculates resistance, and classifies channel
    bg_assignments = np.full(len(mask), -1)
    resistances = np.zeros_like(mask, dtype=np.float32)
    r2s = np.zeros_like(mask, dtype=np.float32)
    classifications = np.full(len(mask), "no_tes", dtype="U15")
    for i in range(len(mask)):
        bgidx = predict_bg_index(amps_full[i], r2s_full[i])
        if bgidx == -1:
            bg_assignments[i] = -1
            resistances[i] = 0
            classifications[i] = "no_tes"
            continue
        bg_assignments[i] = biasgroups[bgidx]
        r2s[i] = r2s_full[i, bgidx]
        current_cmd = tickle_voltage / bias_line_resistance
        current_meas = amps_full[i, bgidx] * pA_per_phi0 * 1e-12
        voltage_meas = (current_cmd - current_meas) * S.R_sh
        res = max(voltage_meas / current_meas, 0.)
        if res < sc_thresh:
            classifications[i] = "sc"
        elif res < normal_thresh:
            classifications[i] = "transition"
        else:
            classifications[i] = "normal"
        resistances[i] = res

    num_chans = len(mask)
    assigned_chans = np.sum(bg_assignments >= 0)
    cprint(f"{assigned_chans} / {num_chans} had a tickle response and were "
           "assigned to bias groups")

    summary = {
        "abs_channels": mask,
        "bg_assignments": bg_assignments,
        "rsquared": r2s,
        "resistances": resistances,
        "classifications": classifications
    }

    if S.output_dir is not None:
        fname = make_filename(S, "tickle_analysis.npy")
        np.save(fname, summary, allow_pickle=True)
        plot_dir = S.plot_dir
    else:
        plot_dir = None

    cprint("Making summary plots")
    plot_tickle_summary(S, summary, save_dir=plot_dir)
    return summary
