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
    optical_bl=None,
    temp_list=None,
    temps_to_cut=None,
    temp_offset=0,
    temp_scaling=1,
    min_rn=0,
    max_rn=np.inf,
    bl_plot=True,
    freq_plot=False,
    array_freq="mf",
    assem_type="ufm",
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
    optical_bl : array-like
        Data from these bias lines will be plotted separately.
    temp_list : array-like

    temps_to_cut : array_like, optional
        A list of temperatures to exclude from the collected data.
        Evaluated after any corrections have been applied.
    temp_offset : float
        Apply a fixed offset (in Kelvin) to all temperature values.
        Default is 0. For recalibrating Princeton thermometry, use
        0.0140 for X-129 and 0.0117 for X-066.
    temp_scaling : float
        Scale temperature values. Default is 1. For recalibrating
        Princeton thermometry, use
        0.875 for X-129 and 0.870 for X-066.
    min_rn: float
        In ohms.
    max_rn: float
        In ohms.
    bl_plot : bool
        Plot the data by bias line. Default True.
    freq_plot : bool
        Plot the data by observing frequency. Default False.
    assem_type : {'ufm', 'spb'}
    array_freq : {'mf', 'uhf'}
    plot_title : str
    figsize : (float, float), optional
    return_data : bool
        Return the collected data as a dictionary. Default is False.

    Returns
    -------
    data_dict : dict
        Dictionary organized by bias line, then, smurf band, then channel,
        and containing ordered arrays of temperature and psat values.
        Only returned if *return_data* is True.
    """
    metadata = np.genfromtxt(metadata_fp, delimiter=",", dtype="str")
    temps_to_cut = np.atleast_1d(temps_to_cut)
    data_dict = {}
    if temp_list is not None:
        temps = []
        for line in metadata:
            if line[0] not in temps:
                temps.append(line[0])
        if len(temps) != len(temp_list):
            raise ValueError(
                "Length of supplied temperature list does not"
                + " match the number of recorded temperatures."
            )
    for line in metadata:
        temp, bias, bl, sb, fp, note = line
        if temp_list is not None:
            # Replace temperature with its corresponding
            # entry in temps_list
            temp = temp_list[temps.index(temp)]
        # Put temperatures into Kelvin.
        temp = float(temp)
        if temp > 40:
            temp *= 1e-3
        if temp in temps_to_cut or temp * 1e3 in temps_to_cut:
            continue
        temp_corr = temp * temp_scaling + temp_offset
        # Ignore files that aren't single-bias-line IV curves
        if bl == "all":
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
                if psat < 0:
                    continue

                if cut_increasing_psat:
                    try:
                        prev_psat = data_dict[bl][sb][ch]["psat"][-1]
                        if psat > prev_psat:
                            continue
                    except:
                        pass

                if not min_rn < d["R_n"] < max_rn:
                    continue

                # key creation
                if sb not in data_dict[bl].keys():
                    data_dict[bl][sb] = dict()
                if ch not in data_dict[bl][sb].keys():
                    data_dict[bl][sb][ch] = {
                        "temp": [],
                        "psat": [],
                        "R_n": [],
                    }
                data_dict[bl][sb][ch]["temp"].append(temp_corr)
                data_dict[bl][sb][ch]["psat"].append(psat)
                data_dict[bl][sb][ch]["R_n"].append(d["R_n"])

    if bl_plot:
        tot = 0
        ncols = np.min((len(data_dict.keys()), 4))
        nrows = int(np.ceil(len(data_dict.keys()) / 4))
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
        axes = np.atleast_2d(axes)
        for idx, bl in enumerate(sorted(list(data_dict.keys()))):
            inds = np.unravel_index(idx, (nrows, ncols))
            ax = axes[inds]
            tes_yield = np.sum(
                [len(data_dict[bl][sb].keys()) for sb in data_dict[bl].keys()]
            )
            tot += tes_yield
            for sb in data_dict[bl].keys():
                for ch, d in data_dict[bl][sb].items():
                    if min(d["temp"]) > 1:
                        run = "Coldload"
                    else:
                        run = "Bath"
                    ax.plot(d["temp"], d["psat"], marker=".", alpha=0.4, linewidth=1)
            ax.set_title(f"BL {bl}, yield {tes_yield}", fontsize=10)
            ax.grid(linestyle="--")
            if inds[1] == 0:
                ax.set_ylabel("Psat [pW]")
            if inds[0] == nrows - 1:
                ax.set_xlabel(f"{run} Temperature [K]")
        plt.suptitle(plot_title, fontsize=12)
        plt.tight_layout()
        print(f"Total TES yield: {tot}")

    if freq_plot:
        assert assem_type == "ufm"
        if array_freq.lower() == "uhf":
            freq1, freq2 = "220", "280"
        else:
            freq1, freq2 = "90", "150"
        bl_freq_map = {bl: freq1 for bl in [0, 1, 4, 5, 8, 9]}
        bl_freq_map.update({bl: freq2 for bl in [2, 3, 6, 7, 10, 11]})
        freq_colors = {
            freq1: "C0",
            freq2: "C2",
            "Dark_" + freq1: "C3",
            "Dark_" + freq2: "C1",
        }
        labeled_dict = dict()
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=figsize)
        for idx, bl in enumerate(sorted(list(data_dict.keys()))):
            freq = bl_freq_map[bl]
            if freq == freq1:
                ax = axes[0]
            else:
                ax = axes[1]
            if bl not in optical_bl:
                freq = "Dark_" + freq
            c = freq_colors[freq]
            for sb in data_dict[bl].keys():
                for ind, (ch, d) in enumerate(data_dict[bl][sb].items()):
                    label = None
                    if ind == 0:
                        labeled = labeled_dict.get(freq, False)
                        if not labeled:
                            label = freq + "GHz"
                            labeled_dict[freq] = True
                    if min(d["temp"]) > 1:
                        run = "Coldload"
                    else:
                        run = "Bath"
                    ax.plot(
                        d["temp"],
                        d["psat"],
                        alpha=0.2,
                        color=c,
                        label=label,
                        marker=".",
                    )
        axes[0].set_title(freq1 + "GHz")
        axes[1].set_title(freq2 + "GHz")
        for ax in axes:
            ax.set_xlabel(f"{run} Temperature [K]")
            ax.set_ylabel("Psat (pW)")
            ax.legend(loc="best")
        plt.suptitle(plot_title, fontsize=12)
        plt.tight_layout()

    if return_data:
        return data_dict


def fit_bathramp_data(
    data_dict,
    assem_type="ufm",
    array_freq="mf",
    optical_bl=None,
    restrict_n=False,
    fit_g=False,
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
        Psats from these bias lines will be plotted separately.
    restrict_n : bool
        Restrict fit values of n within 2--4
    fit_g : bool
        If True, fits G, n, Tc, Psat.
        Otherwise fits k, n, Tc, Psat, solves for G.
    return_data : bool
        If True, returns dictionary of fit parameters
    plot : bool
        Plot histograms of the fit parameters.
    plot_title : str
        Title for output plots

    Returns
    -------
    results_dict : dict
        Fit thermal parameters, keyed by bias line
        and absolute readout channel.

    """
    if not fit_g:
        p0 = [370, 0.190, 3]
        if restrict_n:

            def PsatofT(Tb, k, Tc, n):
                if n < 2 or n > 4:
                    return 9e9
                power = k * (Tc**n - Tb**n)
                return power

        else:

            def PsatofT(Tb, k, Tc, n):
                power = k * (Tc**n - Tb**n)
                return power

    else:
        p0 = [100, 0.190, 3]
        if restrict_n:

            def PsatofT(Tb, G, Tc, n):
                if n < 2 or n > 4:
                    return 9e9
                return (G / n) * (Tc - Tb**n / Tc ** (n - 1))

        else:

            def PsatofT(Tb, G, Tc, n):
                return (G / n) * (Tc - Tb**n / Tc ** (n - 1))

    param_dict = {}

    for bl in data_dict.keys():
        for sb in data_dict[bl].keys():
            for ch, d in data_dict[bl][sb].items():
                if len(d["psat"]) < 4:
                    continue
                try:
                    popt, pcov = scipy.optimize.curve_fit(
                        PsatofT,
                        d["temp"],
                        np.asarray(d["psat"]),
                        p0=p0,
                    )
                except RuntimeError:
                    continue
                if bl not in param_dict.keys():
                    param_dict[bl] = dict()
                if sb not in param_dict[bl].keys():
                    param_dict[bl][sb] = dict()

                param_dict[bl][sb][ch] = {"R_n": data_dict[bl][sb][ch]["R_n"][-1]}

                kg, tc, n = popt
                sigma_kg, sigma_tc, sigma_n = np.sqrt(np.diag(pcov))
                if not fit_g:
                    param_dict[bl][sb][ch]["g"] = n * kg * (tc ** (n - 1))
                    param_dict[bl][sb][ch]["k"] = kg
                    param_dict[bl][sb][ch]["sigma_k"] = sigma_kg
                else:
                    param_dict[bl][sb][ch]["g"] = kg
                    param_dict[bl][sb][ch]["sigma_g"] = sigma_kg
                    param_dict[bl][sb][ch]["k"] = kg / (n * tc ** (n - 1))
                param_dict[bl][sb][ch]["tc"] = tc
                param_dict[bl][sb][ch]["sigma_tc"] = sigma_tc
                param_dict[bl][sb][ch]["n"] = n
                param_dict[bl][sb][ch]["sigma_n"] = sigma_n

                if 0.100 in d["temp"]:
                    psat = d["psat"][(d["temp"].index(0.100))]
                else:
                    psat = param_dict[bl][sb][ch]["k"] * (tc**n - 0.100**n)
                param_dict[bl][sb][ch]["psat100mk"] = psat

    if plot:
        if array_freq.lower() == "uhf":
            freq1, freq2 = "220", "280"
        else:
            freq1, freq2 = "90", "150"

        if assem_type.lower() == "ufm":
            bl_freq_map = {bl: freq1 for bl in [0, 1, 4, 5, 8, 9]}
            bl_freq_map.update({bl: freq2 for bl in [2, 3, 6, 7, 10, 11]})
            freq_colors = [
                (freq1, "C0"),
                (freq2, "C2"),
                ("Optical_" + freq1, "C0"),
                ("Optical_" + freq2, "C2"),
            ]
        else:
            bl_freq_map = {bl: freq1 + "/" + freq2 for bl in np.arange(12)}
            freq_colors = [(freq1 + "/" + freq2, "C0")]
            optical_bl = []

        # Build a dictionary that's useful for histograms,
        # and that works for MF/UHF UFMs and SPBs.
        plotting_dict = dict()
        for bl in param_dict.keys():
            for sb in param_dict[bl].keys():
                freq = bl_freq_map[bl]
                if bl in optical_bl:
                    freq = "Optical_" + freq
                if freq not in plotting_dict.keys():
                    plotting_dict[freq] = dict()
                for key in ["g", "k", "tc", "n", "psat100mk", "R_n"]:
                    if key not in plotting_dict[freq].keys():
                        plotting_dict[freq][key] = []
                    now_param = [
                        param_dict[bl][sb][ch][key] for ch in param_dict[bl][sb].keys()
                    ]
                    plotting_dict[freq][key] += now_param

        fig, ax = plt.subplots(nrows=4, figsize=(9, 9))
        for freq, c in freq_colors:
            if "Optical" in freq:
                dark = False
                histtype = "step"
                ec = c
                lalpha = 0.4
            else:
                dark = True
                histtype = "bar"
                ec = None
                lalpha = 1

            # Psat
            h = ax[0].hist(
                plotting_dict[freq]["psat100mk"],
                fc=c,
                alpha=0.4,
                bins=30,
                range=(0, 20),
                histtype=histtype,
                ec=ec,
                linewidth=1.5,
            )
            med = np.nanmedian(plotting_dict[freq]["psat100mk"])
            std = np.nanstd(plotting_dict[freq]["psat100mk"])
            if not np.isnan(med):
                ax[0].axvline(
                    med,
                    linestyle="--",
                    color=c,
                    alpha=lalpha,
                    label="%sGHz: %.1f $\pm$ %.1f pW" % (freq, med, std),
                )
            ax[0].set_xlabel("Psat at 100 mK [pW]")
            ax[0].set_ylabel("# of TESs")
            ax[0].set_title(" ")

            if dark:
                # Tc
                h = ax[1].hist(
                    plotting_dict[freq]["tc"],
                    fc=c,
                    alpha=0.4,
                    bins=40,
                    range=(0.150, 0.230),
                    histtype=histtype,
                    ec=ec,
                )
                med = np.nanmedian(plotting_dict[freq]["tc"])
                std = np.nanstd(plotting_dict[freq]["tc"])
                if not np.isnan(med):
                    ax[1].axvline(
                        med,
                        linestyle="--",
                        color=c,
                        alpha=lalpha,
                        label="%sGHz: %.3f $\pm$ %.3f K" % (freq, med, std),
                    )
                ax[1].set_xlabel("Tc [K]")
                ax[1].set_ylabel("# of TESs")
                ax[1].set_title(" ")

                # G
                h = ax[2].hist(
                    plotting_dict[freq]["g"],
                    fc=c,
                    alpha=0.4,
                    bins=30,
                    range=(20, 600),
                    histtype=histtype,
                    ec=ec,
                )
                med = np.nanmedian(plotting_dict[freq]["g"])
                std = np.nanstd(plotting_dict[freq]["g"])
                if not np.isnan(med):
                    ax[2].axvline(
                        med,
                        linestyle="--",
                        color=c,
                        alpha=lalpha,
                        label="%sGHz: %0d $\pm$ %0d pW/K" % (freq, med, std),
                    )
                ax[2].set_xlabel("G [pW/K]")
                ax[2].set_ylabel("# of TESs")
                ax[2].set_title(" ")

                # n
                h = ax[3].hist(
                    plotting_dict[freq]["n"],
                    fc=c,
                    alpha=0.4,
                    bins=30,
                    range=(1, 5),
                    histtype=histtype,
                    ec=ec,
                )
                med = np.nanmedian(plotting_dict[freq]["n"])
                std = np.nanstd(plotting_dict[freq]["n"])
                if not np.isnan(med):
                    ax[3].axvline(
                        med,
                        linestyle="--",
                        color=c,
                        alpha=lalpha,
                        label="%sGHz: %.1f $\pm$ %.1f" % (freq, med, std),
                    )
                if restrict_n:
                    ax[3].set_xlabel("n (restricted to 2--4)")
                else:
                    ax[3].set_xlabel("n (free)")
                ax[3].set_ylabel("# of TESs")
                ax[3].set_title(" ")

        for ind in [0, 1, 2, 3]:
            ax[ind].legend(fontsize="small", loc="best")

        plt.suptitle(plot_title, fontsize=16)
        plt.tight_layout()

        # Rn plot
        plt.figure()
        all_rn = []
        for freq, d in plotting_dict.items():
            all_rn += d["R_n"]
        med = np.nanmedian(all_rn)
        plt.hist(all_rn, ec="k", histtype="step", bins=20, range=(5e-3, 10e-3))
        plt.axvline(
            med,
            linestyle="--",
            color="k",
            label="%.1f $\pm$ %.1f mohms" % (med * 1e3, np.nanstd(all_rn) * 1e3),
        )
        plt.xlabel("R_n (ohms)")
        plt.ylabel("Count")
        plt.title(plot_title)
        plt.legend()

    results_dict = {}
    results_dict["data"] = param_dict
    results_dict["metadata"] = {
        "units": {
            "psat100mk": "pW",
            "tc": "K",
            "g": "pW/K",
            "n": "",
            "k": "",
            "R_n": "ohms",
        }
    }
    if return_data:
        return results_dict


def analyze_bathramp(
    metadata_fp,
    array_freq="mf",
    assem_type="ufm",
    optical_bl=None,
    restrict_n=False,
    fit_g=False,
    temp_list=None,
    temps_to_cut=None,
    temp_offset=0,
    temp_scaling=1,
    min_rn=0,
    max_rn=np.inf,
    cut_increasing_psat=True,
    plot=True,
    plot_title="",
    figsize=None,
    return_data=False,
):
    data_dict = collect_psatvstemp_data(
        metadata_fp,
        array_freq=array_freq,
        assem_type=assem_type,
        temp_list=temp_list,
        temp_offset=temp_offset,
        temp_scaling=temp_scaling,
        cut_increasing_psat=cut_increasing_psat,
        min_rn=min_rn,
        max_rn=max_rn,
        temps_to_cut=temps_to_cut,
        optical_bl=optical_bl,
        bl_plot=plot,
        plot_title=plot_title,
        figsize=figsize,
        return_data=True,
    )

    results_dict = fit_bathramp_data(
        data_dict,
        array_freq=array_freq,
        assem_type=assem_type,
        optical_bl=optical_bl,
        restrict_n=restrict_n,
        fit_g=fit_g,
        return_data=return_data,
        plot=plot,
        plot_title=plot_title,
    )

    if return_data:
        return data_dict, results_dict
