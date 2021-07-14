import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from collections import namedtuple
from sodetlib.util import cprint, make_filename, invert_mask, get_r2
import sodetlib.smurf_funcs.smurf_ops as so

from pysmurf.client.util.pub import set_action



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


def analyze_biasgroup_data(times, sig, mask, freq):
    """
    Analyzes a bias-tickle segment of data. For each streamed channel, fits
    amplitude, phase and finds the r-squared value.

    Args
    ----
    times:
        Array of timestamps (in ns because that's what read_stream_data
        returns)
    sig:
        [nchan x nsamp] array of detector signals (in phi0)
    mask:
        Channel mask as returned from read_stream_data. Array mapping
        (band, chan) --> rchan.
    freq:
        Tickle frequency
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

    Args
    ----
    amps:
        Array of fit amplitudes for each segment
    r2s:
        Array of r-squared values for each segment
    thresh:
        Minimum r-squared needed to be considered a fit.

    Returns
    -------
    bg_idx:
        Index corresponding to bg prediction
    """
    possible_idxs = np.where(r2s > thresh)[0]
    if len(possible_idxs) == 0:
        return -1
    bg_idx = possible_idxs[np.argmax(amps[possible_idxs])]
    return bg_idx


def plot_tickle_summary(S, summary, save_dir=None, timestamp=None):
    """
    Makes summary plots from tickle analysis results

    Args
    ----
    S:
        SmurfControl object
    summary:
        Dictionary containing results from tickle analysis
    save_dir:
        Save directory. If None, will not try to save figures.
    """
    if timestamp is None:
        timestamp = S.get_timestamp()
    bgs = summary['bg_assignments']
    res = summary['resistances']
    r2s = summary['rsquared']
    classes = summary['classifications']
    class_cmap = {
        "sc": "C0", "transition": "C1", "normal": "red", "no_tes": "grey"
    }
    cs = np.array([class_cmap[c] for c in classes])

    ### Individual bias group plots
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
        ax.text(0, 0.5, txt, alpha=0.9, transform=ax.transAxes,
                bbox={'facecolor': 'white'})
        ax.set(title=f"Bias Group {bg}", ylabel="Resistance (mOhm)")
        if save_dir is not None:
            fname = os.path.join(
                save_dir, f"{timestamp}_tickle_summary_bg{bg}.png"
            )
            fig.savefig(fname)
            S.pub.register_file(fname, "tickle_summary", plot=True)
        plt.close(fig)

    # Bias plots altogether
    fig, axes = plt.subplots(3, 4, figsize=(30, 15))
    fig.patch.set_facecolor('white')
    for i, ax in enumerate(axes.flatten()):
        m = bgs == i
        ax.scatter(np.arange(np.sum(m)), res[m]*1000, c=cs[m])
        ax.set(title=f"Biasgroup {i}")

    axes[1, 0].set_ylabel("Resistance (mOhm)", fontsize=22)

    if save_dir is not None:
        fname = os.path.join(
            save_dir, f"{timestamp}_tickle_summary_all_bg.png"
        )
        fig.savefig(fname)
        S.pub.register_file(fname, "tickle_summary", plot=True)

    fig.show()

    #  Channel division bar plot
    fig, ax = plt.subplots(figsize=(16, 8))
    xs = np.arange(-1, 12)
    class_order = ['no_tes', 'sc', 'transition', 'normal']
    totals = []
    width = 0.2
    for i, cl in enumerate(class_order):
        m = classes == cl
        totals.append(np.sum(m))
        ys = [np.sum(bgs[m] == bg) for bg in xs]
        offset = 0
        if cl != 'no_tes':
            offset = 2*width
        ax.bar(xs + i*width - offset, ys, width=width, color=class_cmap[cl],
               label=cl)
    ax.set_xticks(xs)
    ax.set_xticklabels(['None'] + list(range(0, 12)))
    ax.set(yscale='log', xlabel='Bias Group', ylabel='Num Channels')
    labels = [f"{class_order[i]} -- {totals[i]} total"
              for i in range(len(class_order))]
    ax.legend(labels)

    if save_dir is not None:
        fname = os.path.join(
            save_dir, f"{timestamp}_channels_assignment_summary.png"
        )
        fig.savefig(fname)
        S.pub.register_file(fname, "tickle_summary", plot=True)
    fig.show()


@set_action()
def analyze_tickle_data(S, tickle_file, assignment_thresh=0.9,
                        normal_thresh=4e-3, sc_thresh=1e-5,
                        make_channel_plots=False, return_full=False,
                        return_segs=False):
    """
    Analyzes bias tickle data.

    Args
    ----
    S:
        SmurfControl object
    tickle_file:
        Summary file created during the take_tickle function.
    assinment_thresh: float
        Minimum r-squared value needed to make an assignment
    normal_thresh: float
        Resistance (Ohm) above which detector will be classified as "normal"
    sc_thresh: float
        Resistance (Ohms) below which detector will be classified as "sc"
    """
    tickle_info = np.load(tickle_file, allow_pickle=True).item()
    biasgroups = tickle_info['bias_groups']
    dat_files = tickle_info['dat_files']
    tickle_freq = tickle_info['tickle_freq']
    tickle_voltage = tickle_info['tone_voltage']
    R_sh = S.R_sh
    pA_per_phi0 = S.pA_per_phi0
    bias_line_resistance = S.bias_line_resistance

    current_cmd = tickle_voltage / bias_line_resistance
    if tickle_info['high_current']:
        current_cmd *= S.high_low_current_ratio

    if R_sh is None:
        raise ValueError("Pysmurf not loaded with config properties!")

    DataSegment = namedtuple("DataSegment", "times sig mask")
    segs = []
    cprint(f"Reading in segmented data. Biasgroups used: {biasgroups}")
    for f in dat_files:
        segs.append(DataSegment(*S.read_stream_data(f)))

    # Fit amplitude and calculate rsquared for each bg segment
    cprint("Fitting sine wave for each biasgroup segment")
    mask = invert_mask(segs[0].mask)
    amps_full = np.zeros((len(mask), len(biasgroups)))
    phases_full = np.zeros((len(mask), len(biasgroups)))
    r2s_full = np.zeros((len(mask), len(biasgroups)))
    for i, bg in enumerate(biasgroups):
        _chans, _amps, _phases, _r2s = analyze_biasgroup_data(
            *segs[i], tickle_freq)
        amps_full[:, i] = _amps
        phases_full[:, i] = _phases
        r2s_full[:, i] = _r2s

    if make_channel_plots:
        cprint("Making channel plots")
        for i, abschan in enumerate(mask):
            fig, ax = plt.subplots()
            txt_array = []
            for bgidx, bg in enumerate(biasgroups):
                s = segs[bgidx]
                ts = (s.times - s.times[0])/1e9
                ys = s.sig[i] * pA_per_phi0 / (2*np.pi)
                ys -= ys.mean()

                if r2s_full[i, bgidx] > assignment_thresh:
                    ax.plot(ts, ys)
                    # Plots fitted signal
                    amp = amps_full[i, bgidx] * S.pA_per_phi0 / (2*np.pi)
                    phase = phases_full[i, bgidx]
                    sighat = sine(ts, amp, phase, tickle_freq)
                    ax.plot(ts, sighat)
                    txt_array.append(f"bias group {bg}, amp={amp:.2f} pA")
                else:
                    ax.plot(ts, ys, color='black', alpha=0.4)

            ax.set(ylabel="Current (pA)", title=f"Channel {abschan}")
            ax.text(0, 0.5, "\n".join(txt_array), transform=ax.transAxes,
                    bbox={'facecolor': 'white'})
            fname = make_filename(S, f"tickle_channel_{abschan}.png",
                                  plot=True)
            fig.savefig(fname)
            plt.close(fig)

    #  Assigns bias groups, calculates resistance, and classifies channel
    bg_assignments = np.full(len(mask), -1)
    resistances = np.zeros_like(mask, dtype=np.float64)
    r2s = np.zeros_like(mask, dtype=np.float64)
    current_meas = np.zeros_like(r2s)
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
        curr_meas = amps_full[i, bgidx] * pA_per_phi0 * 1e-12 / (2*np.pi)
        current_meas[i] = curr_meas
        voltage_meas = (current_cmd - curr_meas) * S.R_sh
        res = voltage_meas / curr_meas
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
        "R_sh": R_sh,
        "run_summary": tickle_info,
        "abs_channels": mask,
        "bg_assignments": bg_assignments,
        "rsquared": r2s,
        "current_meas": current_meas,
        "current_cmd": current_cmd,
        "resistances": resistances,
        "classifications": classifications
    }
    if return_full:
        summary.update({
            'r2s_full': r2s_full,
            'amps_full': amps_full,
        })

    if S.output_dir is not None:
        fname = make_filename(S, "tickle_analysis.npy")
        np.save(fname, summary, allow_pickle=True)
        S.pub.register_file(fname, 'tickle_analysis', format='npy')
        plot_dir = S.plot_dir
    else:
        plot_dir = None

    cprint("Making summary plots")
    plot_tickle_summary(S, summary, save_dir=plot_dir)
    if return_segs:
        return summary, segs
    else:
        return summary


def load_from_dat(S, datfile):
    """
    Loads data from .dat files and returns the timestamps,
    phase, and mask. Just a wrapper for S.read_stream_data().

    Args
    ----
    S:
        SmurfControl object
    datfile: str
        Filepath to .dat file.

    Returns
    -------
    timestamp: numpy.ndarray
        timestamp data in seconds
    phase: numpy.ndarray
        resonator data in units of phi0
    mask: numpy.ndarray
        maskfile. This looks like the ch_info loaded
        from G3t_Smurf. The output array is a
        2x<num_resonator> array, where the first row
        is the band number 0-7 and the second row is the channel
        number 0-511
    tes_biases: numpy.ndarray
        array containing tes biases in units of volts

    """

    (timestamp,
        phase,
        mask,
        tes_biases) = S.read_stream_data(datfile, return_tes_bias=True)

    # converts timestamps from nanoseconds to seconds
    timestamp = timestamp*1e-9
    bands, chans = np.where(mask != -1)
    mask = np.array([bands, chans])
    tes_biases = tes_biases * 2 * S._rtm_slow_dac_bit_to_volt
    return timestamp, phase, mask, tes_biases


# update this so db_path isn't required
def load_from_g3(archive_path, meta_path, db_path, start, stop):
    """
    Loads data from .g3 files using G3t_Smurf and returns the
    timestamps, phase, mask, and TES biases. Requires installs of
    sotodlib, so3g, and spt3g.

    Args
    ----
    archive_path: str
        path to smurf timestreams
    meta_path: str
        path to non-timestream smurf data
    db_path: str
        path to indexed database for loading g3 files.
        If None, defaults to database associated with your
        archive_path.
    start: timestamp or DateTime
        start time for loading the data
    stop: timestamp or DateTime
        stop time for loading the data

    Returns
    -------
    timestamps: numpy.ndarray
        timestamp data in seconds
    phase: numpy.ndarray
        resonator data in units of phi0
    mask: numpy.ndarray
        maskfile. The output array is a
        2x<num_resonator> array, where the first row
        is the band number 0-7 and the second row is the channel
        number 0-511
    tes_biases: numpy.ndarray
        array containing tes biases in units of volts

    """

    # putting these imports here just so sotodlib is not required
    # to run the other analysis functions, if you don't have an
    # active copy
    from sotodlib.io.load_smurf import G3tSmurf

    if db_path:
        SMURF = G3tSmurf(archive_path=archive_path,
                         meta_path=meta_path,
                         db_path=db_path)
    else:
        SMURF = G3tSmurf(archive_path=archive_path,
                         meta_path=meta_path)

    aman = SMURF.load_data(start, stop)
    timestamps = aman.timestamps
    phase = aman.signal
    mask = np.array([aman.ch_info.band, aman.ch_info.channel])
    # this is hard-coded, and is a pysmurf constant.
    # should be not hard-coded...
    rtm_bit_to_volt = 1.9073486328125e-05

    # there are 2 RTMs per bias line
    tes_biases = 2 * aman.biases * rtm_bit_to_volt

    return timestamps, phase, mask, tes_biases


def load_from_sid(cfg, iv_info_fp):
    """
    Loads data for IV analysis from the g3 session ID
    stored in the iv_info file. 

    Args
    ----
    cfg : DetConfig object
        DetConfig object
    iv_info_fp : str
        full filepath to iv_info npy file

    Returns
    -------
    timestamps: numpy.ndarray
        timestamp data in seconds
    phase: numpy.ndarray
        resonator data in units of phi0
    mask: numpy.ndarray
        maskfile. The output array is a
        2x<num_resonator> array, where the first row
        is the band number 0-7 and the second row is the channel
        number 0-511
    tes_biases: numpy.ndarray
        array containing tes biases in units of volts

    """

    iv_info = np.load(iv_info_fp, allow_pickle=True).item()
    aman = so.load_session(cfg, iv_info['session id'])

    timestamps = aman.timestamps
    phase = aman.signal
    mask = np.array([aman.ch_info.band, aman.ch_info.channel])
    # this is hard-coded, and is a pysmurf constant.
    # should be not hard-coded...
    rtm_bit_to_volt = 1.9073486328125e-05

    # there are 2 RTMs per bias line
    tes_biases = 2 * aman.biases * rtm_bit_to_volt

    return timestamps, phase, mask, tes_biases


def analyze_iv_info(iv_info_fp, phase, v_bias, mask,
                    phase_excursion_min=3.0, psat_level=0.9):
    """
    Analyzes an IV curve that was taken using sodetlib's take_iv function,
    detailed in the iv_info dictionary. Based on pysmurf IV analysis.
    Does not require pysmurf, just the data and corresponding metadata.

    Args
    ----
    iv_info_fp: str
        Filepath to the iv_info.npy file generated when the IV
        was taken.
    phase: numpy.ndarray
        Array containing the detector response in units of radians.
    v_bias: numpy.ndarray
        Array containing the bias line information in volts.
    mask: numpy.ndarray
        Array containing the band and channel assignments for any
        active resonator channels.
    phase_excusrion_min: float
        Default 3.0. In radians, the minimum response a channel must
        have to be considered a detector coupled to the bias line.
    psat_level: float
        Default 0.9. Value of Rfrac to calculate Psat.

    Returns
    -------
    iv_full_dict: dict
        Dictionary containing analyzed IV data, keyed by band
        and channel. Also contains the iv_info dict that was
        used for the analysis.
    """

    iv_info = np.load(iv_info_fp, allow_pickle=True).item()

    R_sh = iv_info['R_sh']
    pA_per_phi0 = iv_info['pA_per_phi0']
    bias_line_resistance = iv_info['bias_line_resistance']
    high_current_mode = iv_info['high_current_mode']
    high_low_current_ratio = iv_info['high_low_ratio']
    bias_group = np.atleast_1d(iv_info['bias group'])

    iv_full_dict = {'metadata': {}, 'data': {}}

    iv_full_dict['metadata']['iv_info'] = iv_info_fp
    if iv_info['wafer_id'] is not None:
        iv_full_dict['metadata']['wafer_id'] = iv_info['wafer_id']
    else:
        iv_full_dict['metadata']['wafer_id'] = None
    iv_full_dict['metadata']['version'] = 'v1'

    bands = mask[0]
    chans = mask[1]

    for c in range(len(chans)):

        phase_ch = phase[c]
        phase_exc = np.ptp(phase_ch)

        if phase_exc < phase_excursion_min:
            print(f'Phase excursion too small.'
                  f'Skipping band {bands[c]}, channel {chans[c]}')
            continue

        # assumes biases are the same on all bias groups
        v_bias_bg = v_bias[bias_group[0]]
        v_bias_bg = np.abs(v_bias_bg)

        resp = phase_ch * pA_per_phi0/(2.*np.pi*1e6)  # convert phase to uA

        step_loc = np.where(np.diff(v_bias_bg))[0]

        if step_loc[0] != 0:
            step_loc = np.append([0], step_loc)  # starts from zero
        n_step = len(step_loc) - 1

        # arrays for holding response, I, and V
        resp_bin = np.zeros(n_step)
        v_bias_bin = np.zeros(n_step)
        i_bias_bin = np.zeros(n_step)

        r_inline = bias_line_resistance

        if high_current_mode:

            r_inline /= high_low_current_ratio

        i_bias = 1.0E6 * v_bias_bg / r_inline

        # Find steps and then calculate the TES values in bins
        for i in np.arange(n_step):
            s = step_loc[i]
            e = step_loc[i+1]

            st = e - s
            sb = int(s + np.floor(st/2))
            eb = int(e - np.floor(st/10))

            resp_bin[i] = np.mean(resp[sb:eb])
            v_bias_bin[i] = v_bias_bg[sb]
            i_bias_bin[i] = i_bias[sb]

        d_resp = np.diff(resp_bin)
        d_resp = d_resp[::-1]
        dd_resp = np.diff(d_resp)
        v_bias_bin = v_bias_bin[::-1]
        i_bias_bin = i_bias_bin[::-1]
        resp_bin = resp_bin[::-1]

        # PROBLEMS FROM THIS FITTING SEEM TO COME FROM HOW IT FINDS
        # SC IDX AND NB IDX

        # index of the end of the superconducting branch
        dd_resp_abs = np.abs(dd_resp)
        sc_idx = np.ravel(np.where(dd_resp_abs == np.max(dd_resp_abs)))[0] + 1
        if sc_idx == 0:
            sc_idx = 1

        # index of the start of the normal branch
        # default to partway from beginning of IV curve
        nb_idx_default = int(0.8*n_step)
        nb_idx = nb_idx_default
        for i in np.arange(nb_idx_default, sc_idx, -1):
            # look for minimum of IV curve outside of superconducting region
            # but get the sign right by looking at the sc branch
            if d_resp[i]*np.mean(d_resp[:sc_idx]) < 0.:
                nb_idx = i+1
                break

        nb_fit_idx = int(np.mean((n_step, nb_idx)))
        norm_fit = np.polyfit(i_bias_bin[nb_fit_idx:],
                              resp_bin[nb_fit_idx:], 1)
        if norm_fit[0] < 0:  # Check for flipped polarity
            resp_bin = -1 * resp_bin
            norm_fit = np.polyfit(i_bias_bin[nb_fit_idx:],
                                  resp_bin[nb_fit_idx:], 1)

        resp_bin -= norm_fit[1]  # now in real current units

        sc_fit = np.polyfit(i_bias_bin[:sc_idx], resp_bin[:sc_idx], 1)

        # subtract off unphysical y-offset in superconducting branch; this is
        # probably due to an undetected phase wrap at the kink between the
        # superconducting branch and the transition, so it is *probably*
        # legitimate to remove it by hand. We don't use the offset of the
        # superconducting branch for anything meaningful anyway. This will just
        # make our plots look nicer.
        resp_bin[:sc_idx] -= sc_fit[1]
        sc_fit[1] = 0  # now change s.c. fit offset to 0 for plotting

        R = R_sh * (i_bias_bin/(resp_bin) - 1)
        R_n = np.mean(R[nb_fit_idx:])
        R_L = np.mean(R[1:sc_idx])

        if R_n < 0:
            print(f'Fitted normal resistance is negative. '
                  f'Skipping band {bands[c]}, channel {chans[c]}')
            continue

        v_tes = i_bias_bin*R_sh*R/(R+R_sh)  # voltage over TES
        i_tes = v_tes/R  # current through TES
        p_tes = (v_tes**2)/R  # electrical power on TES

        # calculates P_sat as P_TES at 90% R_n
        # if the TES is at 90% R_n more than once, set to nan
        level = psat_level
        cross_idx = np.where(np.logical_and(R/R_n - level >= 0,
                             np.roll(R/R_n - level, 1) < 0))[0]

        if len(cross_idx) == 1:
            cross_idx = cross_idx[0]
            if cross_idx == 0:
                print(f'Error when finding 90% Rfrac for channel '
                      f'{(bands[c], chans[c])}. Check channel manually.')
                cross_idx = -1
                p_sat = np.nan
            else:
                p_sat = interp1d(R[cross_idx-1:cross_idx+1]/R_n,
                                p_tes[cross_idx-1:cross_idx+1])
                p_sat = p_sat(level)
        else:
            cross_idx = -1
            p_sat = np.nan

        smooth_dist = 5
        w_len = 2*smooth_dist + 1

        # Running average
        w = (1./float(w_len))*np.ones(w_len)  # window
        i_tes_smooth = np.convolve(i_tes, w, mode='same')
        v_tes_smooth = np.convolve(v_tes, w, mode='same')
        r_tes_smooth = v_tes_smooth/i_tes_smooth

        # Take derivatives
        di_tes = np.diff(i_tes_smooth)
        dv_tes = np.diff(v_tes_smooth)
        R_L_smooth = np.ones(len(r_tes_smooth))*R_L
        R_L_smooth[:sc_idx] = dv_tes[:sc_idx]/di_tes[:sc_idx]
        r_tes_smooth_noStray = r_tes_smooth - R_L_smooth
        i0 = i_tes_smooth[:-1]
        r0 = r_tes_smooth_noStray[:-1]
        rL = R_L_smooth[:-1]
        beta = 0.

        # artificially setting rL to 0 for now,
        # to avoid issues in the SC branch
        # don't expect a large change, given the
        # relative size of rL to the other terms

        rL = 0

        # Responsivity estimate
        # add where eq comes from (irwin hilton)
        si = -(1./i0)*(dv_tes/di_tes - (r0+rL+beta*r0)) / \
            ((2.*r0-rL+beta*r0)*dv_tes/di_tes - 3.*rL*r0 - rL**2)

        iv_dict = {}
        iv_dict['R'] = R
        iv_dict['R_n'] = R_n
        iv_dict['idxs'] = np.array([sc_idx, nb_idx, cross_idx])
        iv_dict['p_tes'] = p_tes
        iv_dict['p_sat'] = p_sat
        iv_dict['v_bias'] = v_bias_bin
        iv_dict['v_tes'] = v_tes
        iv_dict['i_tes'] = i_tes
        iv_dict['si'] = si

        iv_full_dict['data'].setdefault(bands[c], {})
        iv_full_dict['data'][bands[c]][chans[c]] = iv_dict

    return iv_full_dict


@set_action()
def analyze_iv_and_save(S, cfg, iv_info_fp, phase, v_bias, mask,
                        phase_excursion_min=3.0, psat_level=0.9, outfile=None):
    """
    Runs analyze_iv_info and saves the output properly, archiving the
    resulting iv_analyze dictionary. Requires pysmurf.

    Args
    ----
    S:
        Smurf Control object. Required for publishing output dict.
    cfg:
        DetConfig object
    iv_info_fp: str
        Filepath to the iv_info.npy file generated when the IV
        was taken.
    phase: numpy.ndarray
        Array containing the detector response in units of radians.
    v_bias: numpy.ndarray
        Array containing the bias line information in volts.
    mask: numpy.ndarray
        Array containing the band and channel assignments for any
        active resonator channels.
    phase_excusrion_min: float
        Default 3.0. In radians, the minimum response a channel must
        have to be considered a detector coupled to the bias line.
    psat_level: float
        Default 0.9. Value of Rfrac to calculate Psat.
    outfile: str
        Filepath to save the output dictionary. Default is the
        output directory in the iv_info dictionary plus the
        basename ctime associated with the IV curve.

    Returns
    -------
    outfile: str
        Filepath where output dictionary is saved.
    """

    iv_info = np.load(iv_info_fp, allow_pickle=True).item()

    iv_analyze = analyze_iv_info(iv_info_fp=iv_info_fp, phase=phase,
                                 v_bias=v_bias, mask=mask,
                                 phase_excursion_min=phase_excursion_min, 
                                 psat_level=psat_level)

    if outfile is None:
        outfile = os.path.join(iv_info['output_dir'],
                               iv_info['basename']+'_iv_analyze.npy')

    np.save(outfile, iv_analyze)
    S.pub.register_file(outfile, 'iv_analyze', format='npy')

    return outfile


def iv_channel_plots(iv_info, iv_analyze, bands=None, chans=None,
                     plot_dir=None, show_plot=False, save_plot=True):
    """
    Generates individual channel plots from an analyzed IV dictionary.
    Will make an IV plot, Rfrac plot, S_I vs. Rfrac plot, and RP plot.

    Args
    ----
    iv_info: dict
        Dictionary loaded from iv_info.npy that contains information from
        when IV curve was taken.
    iv_analyze: dict
        Dictionary generated that contains all fitted and calculated
        parameters from IV analysis.
    bands: list or numpy.ndarray
        Default None. List of bands you want to generate plots for.
        If None, will use all the bands in the analyzed dictionary.
    chans: list or numpy.ndarray
        Default None. List of channels you want to generate plots for.
        If None, will use all the bands in the analyzed dictionary.
    plot_dir: str
        Default None. Directory where you want to save the plots.
        If None, will use the plot_dir specified in the iv_info dict.
    show_plot: bool
        Default False. Whether or not to show the plots.
    save_plot: bool
        Default True. Whether or not to save the plots.
    Returns
    -------
    plot_dir: str
        Filepath where plots are saved, if save_plot is True.
    """
    if plot_dir is None:
        plot_dir = iv_info['plot_dir']

    if bands is None:
        bands = iv_analyze['data'].keys()

    for b in bands:
        print(f'Making plots for band {b}')
        if chans is None:
            chans_iter = iv_analyze['data'][b].keys()
        else:
            chans_iter = chans

        for c in tqdm(chans_iter):

            if np.isnan(iv_analyze['data'][b][c]['p_sat']):
                print(f'Non-physical P_sat. Skipping band {b}, channel {c}.')
                continue

            R_n = iv_analyze['data'][b][c]['R_n']
            R_sh = iv_info['R_sh']
            R = iv_analyze['data'][b][c]['R']

            v_bias = iv_analyze['data'][b][c]['v_bias']
            i_tes = iv_analyze['data'][b][c]['i_tes']
            p_tes = iv_analyze['data'][b][c]['p_tes']
            p_sat = iv_analyze['data'][b][c]['p_sat']

            si = iv_analyze['data'][b][c]['si']

            r_inline = iv_info['bias_line_resistance']
            if iv_info['high_current_mode']:
                r_inline /= iv_info['high_low_ratio']

            plt.figure()
            plt.plot(v_bias, i_tes, color='black')
            plt.plot(v_bias, v_bias/r_inline * (R_sh/(R_n + R_sh))*1e6,
                     ls='--', color='red',
                     label=fr'Normal fit: R$_n$ = {R_n*1e3:.2f} m$\Omega$')
            plt.xlabel(r'V$_{bias}$ (V)')
            plt.ylabel(r'I$_{TES}$ ($\mu$A)')
            plt.title(fr'Band {b}, Ch {c} IV Curve')
            plt.legend()
            if save_plot:
                plt.savefig(os.path.join(plot_dir,
                            iv_info['basename']+f'_b{b}c{c}_iv.png'))
            if show_plot:
                plt.show()
            else:
                plt.close()

            plt.figure()
            plt.plot(v_bias, R/R_n, color='black')
            plt.xlabel(r'V$_{bias}$ (V)')
            plt.ylabel(r'R/R$_n$')
            plt.title(fr'Band {b}, Ch {c} R$_{{frac}}$ vs. Bias')
            if save_plot:
                plt.savefig(os.path.join(plot_dir,
                            iv_info['basename']+f'_b{b}c{c}_rfrac.png'))
            if show_plot:
                plt.show()
            else:
                plt.close()

            sc_idx = iv_analyze['data'][b][c]['idxs'][0]

            plt.figure()
            plt.plot(R[sc_idx:-1]/R_n, si[sc_idx:], color='black')
            plt.xlabel(r'R/R$_n$')
            plt.ylabel(r'S$_I$')
            plt.title(fr'Band {b}, Ch {c} S$_I$ vs. Rfrac')
            if save_plot:
                plt.savefig(os.path.join(plot_dir,
                            iv_info['basename']+f'_b{b}c{c}_si.png'))
            if show_plot:
                plt.show()
            else:
                plt.close()

            plt.figure()
            plt.plot(p_tes, R/R_n, color='black')
            plt.axvline(p_sat, color='red',
                        label=fr'P$_{{TES}}$ at 90% R$_n$ = {p_sat:.2f} pW')
            plt.legend()
            plt.xlabel(r'P$_{TES} (pW)$')
            plt.ylabel(r'R/R$_n$')
            plt.title(f'Band {b}, Ch {c} R-P curve')
            if save_plot:
                plt.savefig(os.path.join(plot_dir,
                            iv_info['basename']+f'_b{b}c{c}_rp.png'))
            if show_plot:
                plt.show()
            else:
                plt.close()

    if save_plot:
        print(f'Plots saved to {plot_dir}.')
        return plot_dir


def iv_summary_plots(iv_info, iv_analyze, Rn_bins=None, Psat_bins=None,
                     plot_dir=None, show_plot=False, save_plot=True):
    """
    Generates summary plots from an analyzed IV dictionary.
    Will make histograms of R_n and electrical power at 90% Rn (Psat).
    Will ignore any channels with unreasonable/unexpected normal
    resistance, or negative saturation powers.

    Args
    ----
    iv_info: dict
        Dictionary loaded from iv_info.npy that contains information from
        when IV curve was taken.
    iv_analyze: dict
        Dictionary generated that contains all fitted and calculated
        parameters from IV analysis.
    plot_dir: str
        Default None. Directory where you want to save the plots.
        If None, will use the plot_dir specified in the iv_info dict.
    show_plot: bool
        Default False. Whether or not to show the plots.
    save_plot: bool
        Default True. Whether or not to save the plots.
    Returns
    -------
    plot_dir: str
        Filepath where plots are saved, if save_plot is True.
    """
    if plot_dir is None:
        plot_dir = iv_info['plot_dir']

    Rns = []
    Psats = []

    Rn_upper = 0.02
    Rn_lower = 0.0

    print('Not including any channels with atypical normal resistances.')

    for b in iv_analyze['data'].keys():
        for c in iv_analyze['data'][b].keys():
            if (iv_analyze['data'][b][c]['R_n'] < Rn_upper
                    and iv_analyze['data'][b][c]['R_n'] > Rn_lower):
                Rns.append(iv_analyze['data'][b][c]['R_n']*1e3)

            if not np.isnan(iv_analyze['data'][b][c]['p_sat']):
                Psats.append(iv_analyze['data'][b][c]['p_sat'])

    Rns = np.array(Rns)
    Psats = np.array(Psats)

    Rn_median = np.median(Rns)

    if Rn_bins is None:
        Rn_bins = np.arange(np.min(Rns)-1, np.max(Rns)+1, 0.1)
    if Psat_bins is None:
        Psat_bins = np.arange(np.min(Psats)-1, np.max(Psats)+1, 0.5)

    plt.figure()
    plt.hist(Rns, ec='k', bins=Rn_bins, color='grey')
    plt.axvline(Rn_median, color='purple', lw=2.0,
                label=fr'Median R$_n$ = {Rn_median:.2f} m$\Omega$')
    plt.legend()
    plt.xlabel(r'R$_n$ (m$\Omega$)')
    plt.ylabel('Counts')
    plt.title('Normal Resistance Distribution')
    if save_plot:
        plt.savefig(os.path.join(plot_dir, iv_info['basename']+'_rn_hist.png'))
    if show_plot:
        plt.show()
    else:
        plt.close()

    Psat_median = np.nanmedian(Psats)

    plt.figure()
    plt.hist(Psats, ec='k', bins=Psat_bins, color='grey')
    plt.axvline(Psat_median, color='purple', lw=2.0,
                label=fr'Median P$_{{sat}}$ = {Psat_median:.2f} pW')
    plt.legend()
    plt.xlabel(r'P$_{{sat}} (pW)$')
    plt.ylabel('Counts')
    plt.title(r'Electrical Power at 90% R$_n$')
    if save_plot:
        plt.savefig(os.path.join(plot_dir,
                    iv_info['basename']+'_psat_hist.png'))
    if show_plot:
        plt.show()
    else:
        plt.close()

    if save_plot:
        print(f'Plots saved to {plot_dir}.')
        return plot_dir


@set_action()
def make_bias_group_map(S, tsum_fp):
    """
    Uses the analyzed tickle data to generate a map describing which
    band/channel pairs are connected to which bias groups.

    Args
    ----
    S:
        SmurfControl object
    tsum_fp: str
        path to a .npy file containing the analyzed dict generated by 
        analyze_tickle_data

    Returns
    -------
    bg_map_fp: str
        Filepath to the bias group map, which is saved on the smurf server
        in /data/smurf_data/bias_group_maps, with the ctime corresponding to 
        the ctime of the analyzed tickle dict. The .npy file contains a dict that is 
        keyed by band and channel, and contains the bias group that the keys for 
        band and channel correspond to.

    """

    tsum = np.load(tsum_fp, allow_pickle=True).item()
    band = tsum['abs_channels']//512
    chan = tsum['abs_channels'] % 512

    bg_map = {}
    for i, c in enumerate(chan):
        bg_map.setdefault(band[i], {})
        bg_map[band[i]][c] = tsum['bg_assignments'][i]

    # Save bg map to the folder on the smurf server
    basename = os.path.split(tsum_fp)[1][:10]
    bg_map_fp = os.path.join('/data/smurf_data/bias_group_maps',
                             basename + '_bg_map.npy')

    np.save(bg_map_fp, bg_map)
    S.log(f'Writing bias group mapping to {bg_map_fp}.')
    S.pub.register_file(bg_map_fp, 'bias_group_map', format='npy')

    return bg_map_fp


@set_action()
def bias_points_from_rfrac(S, iv_analyze_fp, bias_group_map_fp,
                           rfrac=0.5, bias_groups=None):
    """
    Finds ideal bias points for each bias group requested.

    Args
    ----
    S:
        SmurfControl object
    iv_analyze_fp: str
        path to a .npy file containing the iv_analyze dict generated
        in the sodetlib IV analysis functions.
    bias_group_map_fp: str
        filepath to the bias group map dictionary, which map the band/channel
        pair to the connected bias group.
    rfrac: float, default = 0.5
        Number between 0 and 1. Specifies the %Rn that you would like to bias
        the detectors to.
    bias_groups: list or numpy.ndarray, default None
        Which bias groups you want to find bias points for. If None, will use whatever
        bias groups are specified in the iv_info file corresponding to the analyzed IV.

    Returns
    -------
    biases_fp: str
        Filepath to the dictionary with the bias points for the specified bias groups.
        This will contain the bias voltage for each bias group to place as many channels
        connected to the bias group as close as possible to the specified bias_point.

    """

    iv_analyze = np.load(iv_analyze_fp, allow_pickle=True).item()
    iv_info_fp = iv_analyze['metadata']['iv_info']
    iv_info = np.load(iv_info_fp, allow_pickle=True).item()

    if bias_groups is None:
        bias_groups = iv_info['bias group']
    elif bias_groups not in iv_info['bias group']:
        print('Error: Specified bias groups not included in '
              'analyzed IV curves.')
        return  # should this return something specific?

    bias_group_map = np.load(bias_group_map_fp, allow_pickle=True).item()

    bg_ch_bias_targets = {}
    print('Ignoring channels with atypical normal resistances '
          ' and non-physical Psats.')

    for b in iv_analyze['data'].keys():
        for c in iv_analyze['data'][b].keys():
            try:
                bg = bias_group_map[b][c]
            except KeyError:
                print(f'Band {b}, channel {c} does not have '
                      'an assigned bias group in this map. '
                      'Skipping pair.')
                continue
            if bg not in bias_groups:
                print(f'Band {b}, channel {c} connected to '
                      f'bias group {bg}, which is not included '
                      'in this call. Skipping this pair.')
                continue

            bg_ch_bias_targets.setdefault(bg, [])

            Rn_upper = 0.02
            Rn_lower = 0.0

            if (iv_analyze['data'][b][c]['R_n'] < Rn_upper
                    and iv_analyze['data'][b][c]['R_n'] > Rn_lower):
                if not np.isnan(iv_analyze['data'][b][c]['p_sat']):
                    R_frac = iv_analyze['data'][b][c]['R']/iv_analyze['data'][b][c]['R_n']
                    bias_idx = np.argmin(np.abs(R_frac - rfrac))
                    ch_v_bias_target = iv_analyze['data'][b][c]['v_bias'][bias_idx]
                    bg_ch_bias_targets[bg].append(ch_v_bias_target)

    bg_biases = {'metadata': {}, 'biases': {}}
    bg_biases['metadata']['iv_analyze'] = iv_analyze_fp
    if iv_info['wafer_id'] is not None:
        bg_biases['metadata']['wafer_id'] = iv_info['wafer_id']
    else:
        bg_biases['metadata']['wafer_id'] = None

    for bg in bias_groups:

        bg_biases['biases'][bg] = np.mean(bg_ch_bias_targets[bg])

    # need to save the bg_biases object to some fp and then return the fp

    biases_fp = os.path.join(iv_info['output_dir'],
                             iv_info['basename'] + '_bias_points.npy')

    np.save(biases_fp, bg_biases)
    S.log(f'Writing chosen bias points to {biases_fp}.')
    S.pub.register_file(biases_fp, 'bias_points', format='npy')

    return biases_fp
