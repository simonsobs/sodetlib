# psatvstemp_analysis.py
# Functions for analyzing and plotting data taken
# as part of a cold load ramp or bath ramp.

import os
import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt


def collect_psatvstemp_data(
    metadata_fp,
    cut_increasing_psat=True,
    temp_offset=0,
    temps_to_cut=None,
    plot=True,
    plot_title="",
    figsize=None,
    return_data=False,
):
    """
    Collect Psat vs. temperature data from a metadata csv.
    Works for both bath temp sweeps and cold load sweeps.

    Parameters
    ----------
    metadata_fp : str
        Path to a metadata csv file with columns ordered like
        temp, bias_v, bias_line, smurf_band, filepath, note.
    cut_increasing_psat : bool
        Cut detector data points that have a higher Psat than
        the preceding point. Default is True.
    temp_offset : float
        Apply a fixed offset to all temperature values. Default is 0.
    temps_to_cut : array_like, optional
        A list of temperatures to exclude from the collected data.
    plot : bool
        Plot the data. Default True.
    plot_title : str
    figsize : (float, float), optional
    return_data : bool
        Return the collected data as a dictionary. Default is False.

    Returns
    -------
    data_dict : dict
        Dictionary organized by bias line and smurf band+channel label,
        and containing ordered arrays of temperature and psat values.
        Only returned if *return_data* is True.
    """
    metadata = np.genfromtxt(metadata_fp, delimiter=",", dtype="str")
    temps_to_cut = np.atleast_1d(temps_to_cut)
    data_dict = {}
    for line in metadata:
        temp, bias, bl, sb, fp, note = line
        # Put temperatures into Kelvin.
        temp = float(temp)
        if temp > 40:
            temp *= 1e-3
        if temp in temps_to_cut or temp * 1e3 in temps_to_cut:
            continue
        if bl == "all":  # or note.lower() != "iv":
            continue
        bl = int(bl)
        if "iv_raw_data" in fp:
            iv_analyzed_fp = fp.replace("iv_raw_data", "iv")
        elif "iv_info" in fp:
            iv_analyzed_fp = fp.replace("iv_info", "iv_analyze")
        else:
            iv_analyzed_fp = fp
        # First look for data on long term storage /data2
        iv_analyzed_fp = iv_analyzed_fp.replace("/data/smurf_data", "/data2/smurf_data")
        # If not there, look for copy on smurf-srv15
        if not os.path.exists(iv_analyzed_fp):
            _, _, _, _, date, slot, sess, _, fp = iv_analyzed_fp.split("/")
            new_fp = os.path.join("/data/smurf/", fp[0:5], slot, "*run_iv/outputs", fp)
            try:
                iv_analyzed_fp = glob(new_fp)[0]
            except IndexError:
                raise FileNotFoundError(
                    f"Could not find {iv_analyzed_fp} or "
                    f"any file matching {new_fp} on daq."
                )
        iv_analyzed = np.load(iv_analyzed_fp, allow_pickle=True).item()
        if "data" in iv_analyzed.keys():
            iv_analyzed = iv_analyzed["data"]

        if bl not in data_dict.keys():
            data_dict[bl] = {}
        for sb in iv_analyzed.keys():
            if sb == "high_current_mode":
                continue
            for ch, d in iv_analyzed[sb].items():
                abs_ch = sb * 512 + ch

                # same cuts as for iv plots
                ind = np.where(d["p_tes"] > 15)[0]
                if len(ind) == 0:
                    continue
                if np.abs(np.std(d["R"][-100:]) / np.mean(d["R"][-100:])) > 5e-3:
                    continue
                if d["R"][-1] < 2e-3:
                    continue
                try:
                    psat_idx = np.where(d["R"] < 0.9 * d["R_n"])[0][-1]
                except:
                    continue
                psat = d["p_tes"][psat_idx]

                if cut_increasing_psat:
                    try:
                        prev_psat = data_dict[bl][abs_ch]["psat"][-1]
                        if psat > prev_psat:
                            continue
                    except:
                        pass
                # cuts passed

                if abs_ch not in data_dict[bl].keys():
                    data_dict[bl][abs_ch] = {"temp": [], "psat": []}
                data_dict[bl][abs_ch]["temp"].append(temp + temp_offset)
                data_dict[bl][abs_ch]["psat"].append(psat)

    if plot:
        tot = 0
        ncols = np.min((len(data_dict.keys()), 3))
        nrows = int(np.ceil(len(data_dict.keys()) / 3))
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
        axes = np.atleast_2d(axes)
        for idx, bl in enumerate(sorted(list(data_dict.keys()))):
            inds = np.unravel_index(idx, (nrows, ncols))
            ax = axes[inds]
            tes_yield = len(data_dict[bl].keys())
            tot += tes_yield
            for abs_ch, d in data_dict[bl].items():
                ax.plot(d["temp"], d["psat"], marker=".", alpha=0.4, linewidth=1)
            ax.set_title(f"BL {bl}, yield {tes_yield}", fontsize=10)
            ax.grid(linestyle="--")
            if inds[1] == 0:
                ax.set_ylabel("Psat [pW]")
            if inds[0] == 3:
                ax.set_xlabel("Temperature [K]")
        plt.suptitle(plot_title, fontsize=12)
        plt.tight_layout()
        print(f"Total TES yield: {tot}")

    if return_data:
        return data_dict


def fit_bathramp_data(
    data_dict,
    assem_type="ufm",
    array_freq="mf",
    optical_bl=[0, 1, 2, 3],
    restrict_n=False,
    fit_g=False,
    correct_temp=False,
    return_data=False,
    plot=True,
    plot_title="",
):
    """
    Fit thermal parameters from bath ramp data.

    Parameters
    ----------
    data_dict : dict
        Dictionary returned by collect_psatvstemp_data
    assem_type : {'ufm', 'spb'}
    array_freq : {'mf', 'uhf'}
    optical_bl : array-like
        Data from these bias lines will not be included
        in plotted histograms.
    restrict_n : bool
        Restrict fit values of n within 2--4
    fit_g : bool
        If True, fits G, n, Tc, Psat.
        Otherwise fits k, n, Tc, Psat, solves for G.
    correct_temp : bool
        If True, apply the necessary correction for
        thermometer X-129
    return_data : bool
        If True, returns dictionary of fit parameters
    plot : bool
        Plot histograms of the fit parameters.
    plot_title : str
        Title for output plots

    Returns
    -------
    param_dict : dict
        Fit thermal parameters, keyed by bias line
        and absolute readout channel.

    """
    if not fit_g:
        p0 = [370, 0.190, 3]
        if restrict_n:

            def PsatofT(Tb, k, Tc, n):
                if n < 2 or n > 4:
                    return 9e9
                power = k * (Tc ** n - Tb ** n)
                return power

        else:

            def PsatofT(Tb, k, Tc, n):
                power = k * (Tc ** n - Tb ** n)
                return power

    else:
        p0 = [100, 0.190, 3]
        if restrict_n:

            def PsatofT(Tb, G, Tc, n):
                if n < 2 or n > 4:
                    return 9e9
                return (G / n) * (Tc - Tb ** n / Tc ** (n - 1))

        else:

            def PsatofT(Tb, G, Tc, n):
                return (G / n) * (Tc - Tb ** n / Tc ** (n - 1))

    if correct_temp:
        slope, offset = 0.87482423, 0.01402322438
    else:
        slope, offset = 1, 0

    if array_freq.lower() == "uhf":
        freq1, freq2 = "220", "280"
    else:
        freq1, freq2 = "90", "150"

    param_dict = {}

    for bl in data_dict.keys():
        for abs_ch, d in data_dict[bl].items():
            if len(d["psat"]) < 4:
                continue
            try:
                parameters = scipy.optimize.curve_fit(
                    PsatofT,
                    (np.array(d["temp"]) * slope + offset),
                    np.asarray(d["psat"]),
                    p0=p0,
                )
            except RuntimeError:
                continue
            if bl not in param_dict.keys():
                param_dict[bl] = dict()
            param_dict[bl][abs_ch] = dict()

            kg, tc, n = parameters[0]
            if not fit_g:
                param_dict[bl][abs_ch]["g"] = n * kg * (tc ** (n - 1))
                param_dict[bl][abs_ch]["k"] = kg
            else:
                param_dict[bl][abs_ch]["g"] = kg
                param_dict[bl][abs_ch]["k"] = kg / (n * tc ** (n - 1))
            param_dict[bl][abs_ch]["tc"] = tc
            param_dict[bl][abs_ch]["n"] = n

            if 0.100 in d["temp"]:
                psat = d["psat"][(d["temp"].index(0.100))]
            else:
                psat = param_dict[bl][abs_ch]["k"] * (tc ** n - 0.100 ** n)
            param_dict[bl][abs_ch]["psat100mk"] = psat

    if plot:
        # Build a dictionary that's useful for histograms,
        # and that works for MF/UHF UFMs and SPBs.
        if assem_type.lower() == "ufm":
            bl_freq_map = {bl: freq1 for bl in [0, 1, 4, 5, 8, 9]}
            bl_freq_map.update({bl: freq2 for bl in [2, 3, 6, 7, 10, 11]})
            freq_colors = [(freq1, "C0"), (freq2, "C2")]
        else:
            bl_freq_map = {bl: freq1 + "/" + freq2 for bl in np.arange(12)}
            freq_colors = [(freq1 + "/" + freq2, "C0")]
            optical_bl = []
        plotting_dict = dict()
        for bl in param_dict.keys():
            if bl in optical_bl:
                continue
            freq = bl_freq_map[bl]
            if freq not in plotting_dict.keys():
                plotting_dict[freq] = dict()
            for key in ["g", "k", "tc", "n", "psat100mk"]:
                if key not in plotting_dict[freq].keys():
                    plotting_dict[freq][key] = []
                now_param = [
                    param_dict[bl][abs_ch][key] for abs_ch in param_dict[bl].keys()
                ]
                plotting_dict[freq][key] += now_param

        fig, ax = plt.subplots(nrows=4, figsize=(9, 9))
        for freq, c in freq_colors:
            h = ax[0].hist(
                plotting_dict[freq]["psat100mk"],
                fc=c,
                alpha=0.4,
                bins=30,
                range=(0, 20),
            )
            med = np.nanmedian(plotting_dict[freq]["psat100mk"])
            if not np.isnan(med):
                ax[0].axvline(
                    med,
                    linestyle="--",
                    color=c,
                    label="%sGHz: %.1fpW" % (freq, med),
                )
            ax[0].set_xlabel("Psat at 100 mK [pW]")
            ax[0].set_ylabel("# of TESs")
            ax[0].set_title(" ")

            h = ax[1].hist(
                plotting_dict[freq]["tc"],
                fc=c,
                alpha=0.4,
                bins=40,
                range=(0.150, 0.230),
            )
            med = np.nanmedian(plotting_dict[freq]["tc"])
            if not np.isnan(med):
                ax[1].axvline(
                    med,
                    linestyle="--",
                    color=c,
                    label="%sGHz: %.3fK" % (freq, med),
                )
            ax[1].set_xlabel("Tc [K]")
            ax[1].set_ylabel("# of TESs")
            ax[1].set_title(" ")

            h = ax[2].hist(
                plotting_dict[freq]["g"],
                fc=c,
                alpha=0.4,
                bins=30,
                range=(100, 600),
            )
            med = np.nanmedian(plotting_dict[freq]["g"])
            if not np.isnan(med):
                ax[2].axvline(
                    med,
                    linestyle="--",
                    color=c,
                    label="%sGHz: %0dpW/K" % (freq, med),
                )
            ax[2].set_xlabel("G [pW/K]")
            ax[2].set_ylabel("# of TESs")
            ax[2].set_title(" ")

            h = ax[3].hist(
                plotting_dict[freq]["n"],
                fc=c,
                alpha=0.4,
                bins=30,
                range=(1, 5),
            )
            med = np.nanmedian(plotting_dict[freq]["n"])
            if not np.isnan(med):
                ax[3].axvline(
                    med,
                    linestyle="--",
                    color=c,
                    label="%sGHz: %.1f" % (freq, med),
                )
            if restrict_n:
                ax[3].set_xlabel("n (restricted to 2--4)")
            else:
                ax[3].set_xlabel("n (free)")
            ax[3].set_ylabel("# of TESs")
            ax[3].set_title(" ")

        for ind in [0, 1, 2, 3]:
            ax[ind].legend(fontsize="small", loc=2)

        plt.suptitle(plot_title, fontsize=16)
        plt.tight_layout()

    if return_data:
        return param_dict


def analyze_bathramp(
    metadata_fp,
    array_freq="mf",
    assem_type="ufm",
    optical_bl=[0, 1, 2, 3],
    restrict_n=False,
    fit_g=False,
    correct_temp=False,
    temp_offset=0,
    cut_increasing_psat=True,
    temps_to_cut=None,
    plot=True,
    plot_title="",
    figsize=None,
    return_data=False,
):
    data_dict = collect_psatvstemp_data(
        metadata_fp,
        temp_offset=temp_offset,
        cut_increasing_psat=cut_increasing_psat,
        temps_to_cut=temps_to_cut,
        plot=plot,
        plot_title=plot_title,
        figsize=figsize,
        return_data=True,
    )

    param_dict = fit_bathramp_data(
        data_dict,
        array_freq=array_freq,
        assem_type=assem_type,
        optical_bl=optical_bl,
        correct_temp=correct_temp,
        restrict_n=restrict_n,
        fit_g=fit_g,
        return_data=return_data,
        plot=plot,
        plot_title=plot_title,
    )

    if return_data:
        return data_dict, param_dict
