import numpy as np
import time
from scipy.interpolate import interp1d
import sodetlib as sdl
import matplotlib.pyplot as plt
import os
from copy import deepcopy
from tqdm.auto import trange
from typing import Optional, Dict, Any
from dataclasses import dataclass, asdict

import warnings
warnings.filterwarnings('ignore')
np.seterr(all="ignore")

class IVAnalysis:
    """
    IVAnalysis is the object used to hold take_iv information and analysis
    products. When instantiating from scratch, all keyword arguments must be
    specified. (This is not true if you are loading from an existing file
    using the load function).

    Args
    ----
    S : SmurfControl
        Pysmurf instance
    cfg : DetConfig
        DetConfig instance
    run_kwargs : dict
        Dictionary of keyword arguments passed to the take_iv function.
    sid : int
        Session id from the IV stream session
    start_times : np.ndarray
        Array of start_times of each bias point
    stop_times : np.ndarray
        Array of stop_times of each bias point

    Attributes
    -----------
    meta : dict
        Dictionary of pysmurf and sodetlib metadata during the IV.
    biases_cmd : np.ndarray
        Array of commanded biases voltages (in low-current-mode units)
    nbiases : int
        Number of bias points commanded throughout the IV
    bias_groups : np.ndarray
        Array containing the bias-groups included in this IV
    am : AxisManager
        AxisManager containing TOD from the IV (this is not saved to disk,
        but can be loaded with the _load_am function)
    nchans : int
        Number of channels included in the IV
    bands : np.ndarray
        Array of shape (nchans) containing the smurf-band of each channel
    channels : np.ndarray
        Array of shape (nchans) containing the smurf-channel of each channel
    v_bias : np.ndarray
        Array of shape (nbiases) containing the bias voltage at each bias point
        (in low-current-mode Volts)
    i_bias : np.ndarray
        Array of shape (nbiases) containing the commanded bias-current
        at each step (Amps)
    resp : np.ndarray
        Array of shape (nchans, nbiases) containing the squid response (Amps)
        at each bias point
    R : np.ndarray
        Array of shape (nchans, nbiases) containing the TES Resistance of each
        channel at each bias point
    p_tes : np.ndarray
        Array of shape (nchans, nbiases) containing the electrical power on the
        TES (W) of each channel at each bias point
    v_tes : np.ndarray
        Array of shape (nchans, nbiases) containing the voltage across the TES
        for each channel at each bias point (V)
    i_tes : np.ndarray
        Array of shape (nchans, nbiases) containing the current across the TES
        for each channel at each bias point (Amps)
    R_n : np.ndarray
        Array of shape (nchans) containing the normal resistance (Ohms) of the
        TES
    R_L : np.ndarray
        Array of shape (nchans) containing the non-TES resistance (Ohms).
        Should be shunt resistance + parasitic resistance
    p_sat : np.ndarray
        Array of shape (nchans) containing the electrical power (W) at which
        Rfrac crosses the 90% threshold (or whatever arg is passed to the
        analysis fn). Note that this is only truly the saturation power in
        the absence of optical power.
    si : np.ndarray
        Array of shape (nchans, nbiases) containing the responsivity (1/V)
        of the TES for each bias-step
    idxs : np.ndarray
        Array of shape (nchans, 3) containing:
            1. Last index of the SC branch
            2. First index off the normal branch
            3. Index at which p_tes crosses the 90% thresh
    bgmap : np.ndarray
        Array of shape (nchans) containing bias-group assignment of each
        channel in the IV. This is not calculated with the IV but pulled in
        from the device cfg, so it's important you run the take_bgmap function
        to generate the bias-group map before running the IV.
    polarity : np.ndarray
        Array of shape (nchans) containing the polarity of each channel (also
        from the bias-map file in the device cfg). This tells you if the TES
        current changes in the same direction or opposite direction of the bias
        current.
    """
    def __init__(self, S=None, cfg=None, run_kwargs=None, sid=None,
                 start_times=None, stop_times=None) -> None:

        self._S = S
        self._cfg = cfg
        self.run_kwargs = run_kwargs
        self.sid = sid
        self.start_times = start_times
        self.stop_times = stop_times
        self.am = None

        if self._S is not None:
            # Initialization stuff on initial creation with a smurf instance
            self.meta = sdl.get_metadata(S, cfg)
            self.biases_cmd = np.sort(self.run_kwargs['biases'])
            self.nbiases = len(self.biases_cmd)
            self.bias_groups = self.run_kwargs['bias_groups']

            am = self._load_am()
            self.nchans= len(am.signal)

            self.bands = am.ch_info.band
            self.channels = am.ch_info.channel
            self.v_bias = np.full((self.nbiases, ), np.nan)
            self.i_bias = np.full((self.nbiases, ), np.nan)
            self.v_thevenin = np.full((self.nbiases, ), np.nan)
            self.resp = np.full((self.nchans, self.nbiases), np.nan)
            self.R = np.full((self.nchans, self.nbiases), np.nan)
            self.p_tes = np.full((self.nchans, self.nbiases), np.nan)
            self.v_tes = np.full((self.nchans, self.nbiases), np.nan)
            self.i_tes = np.full((self.nchans, self.nbiases), np.nan)
            self.R_n = np.full((self.nchans, ), np.nan)
            self.R_L = np.full((self.nchans, ), np.nan)
            self.p_sat = np.full((self.nchans, ), np.nan)
            self.si = np.full((self.nchans, self.nbiases), np.nan)
            self.idxs = np.full((self.nchans, 3), -1, dtype=int)

            self.bgmap, self.polarity = sdl.load_bgmap(
                self.bands, self.channels, self.meta['bgmap_file']
            )

    def to_dict(self):
        saved_fields = [
            'meta', 'bands', 'channels', 'sid', 'start_times', 'stop_times',
            'run_kwargs', 'biases_cmd', 'bias_groups', 'nbiases', 'nchans',
            'bgmap', 'polarity', 'resp', 'v_bias', 'i_bias', 'R', 'R_n', 'R_L',
            'idxs', 'p_tes', 'v_tes', 'i_tes', 'p_sat', 'si', 'v_thevenin'
        ]
        return {
            field: getattr(self, field, None) for field in saved_fields
        }

    def save(self, path=None, update_cfg=False):
        data = self.to_dict()
        if path is not None:
            np.save(path, data, allow_pickle=True)
        else:
            self.filepath = sdl.validate_and_save(
                'iv_analysis.npy', data, S=self._S, cfg=self._cfg, register=True,
                make_path=True)
            if update_cfg:
                self._cfg.dev.update_experiment({'iv_file': self.filepath},
                                                update_file=True)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'IVAnalysis':
        iva = cls()
        for key, val in data.items():
            setattr(iva, key, val)

        if len(iva.start_times) == 1:
            # Data was taken before we started saving timestamps for each
            # BG separately. Convert this to the 2d array that's now expected
            # by analysis.
            iva.start_times = np.vstack([iva.start_times for _ in range(15)])
            iva.stop_times = np.vstack([iva.stop_times for _ in range(15)])

        return iva

    @classmethod
    def load(cls, path) -> 'IVAnalysis':
        return cls.from_dict(np.load(path, allow_pickle=True).item())


    def _load_am(self, arc=None):
        if self.am is None:
            if arc:
                self.am = arc.load_data(self.start_times[0],
                                        self.stop_times[-1])
            else:
                self.am = sdl.load_session(self.meta['stream_id'], self.sid)
        return self.am


def compute_psats(iva, psat_level=0.9):
    """
    Computes Psat for an IVAnalysis object. Will save results to iva.p_sat.
    This assumes i_tes, v_tes, and r_tes have already been calculated.

    Args
    ----
    iva : IVAnalysis
        IV Analysis object
    psat_level : float
        R_frac level for which Psat is defined. If 0.9, Psat will be the
        power on the TES when R_frac = 0.9.

    Returns
    -------
    p_sat : np.ndarray
        Array of length (nchan) with the p-sat computed for each channel (W)
    """
    # calculates P_sat as P_TES when Rfrac = psat_level
    # if the TES is at 90% R_n more than once, just take the last crossing
    for i in range(iva.nchans):
        if np.isnan(iva.R_n[i]):
            continue

        level = psat_level
        R = iva.R[i]
        R_n = iva.R_n[i]
        p_tes = iva.p_tes[i]
        cross_idx = np.where(R/R_n < level)[0]

        if len(cross_idx) == 0:
            iva.p_sat[i] = np.nan
            continue

        # Takes cross-index to be the last time Rfrac crosses psat_level
        cross_idx = cross_idx[-1]
        if cross_idx == 0:
            iva.p_sat[i] = np.nan
            continue

        iva.idxs[i, 2] = cross_idx
        try:
            iva.p_sat[i] = interp1d(
                R[cross_idx:cross_idx+2]/R_n,
                p_tes[cross_idx:cross_idx+2]
            )(level)
        except ValueError:
            iva.p_sat[i] = np.nan

    return iva.p_sat


def compute_si(iva):
    """
    Computes responsivity S_i for an IV analysis object. Will save results
    to iva.si. This assumes i_tes, v_tes, and r_tes have already been
    calculated.

    Args
    ----
    iva : IVAnalysis
        IV Analysis object

    Returns
    -------
    si : np.ndarray
        Array of length (nchan, nbiases) with  the responsivity as a fn of bias
        voltage for each channel (V^-1).
    """
    smooth_dist = 5
    w_len = 2 * smooth_dist + 1
    w = (1./float(w_len))*np.ones(w_len)  # window

    for i in range(iva.nchans):
        if np.isnan(iva.R_n[i]):
            continue

        # Running average
        i_tes_smooth = np.convolve(iva.i_tes[i], w, mode='same')
        v_tes_smooth = np.convolve(iva.v_tes[i], w, mode='same')
        r_tes_smooth = v_tes_smooth/i_tes_smooth

        sc_idx = iva.idxs[i, 0]
        if sc_idx == -1:
            continue
        R_L = iva.R_L[i]

        # Take derivatives
        di_tes = np.diff(i_tes_smooth)
        dv_tes = np.diff(v_tes_smooth)
        R_L_smooth = np.ones(len(r_tes_smooth-1)) * R_L
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

        # Responsivity estimate, derivation done here by MSF
        # https://www.overleaf.com/project/613978cb38d9d22e8550d45c
        si = -(1./(i0*r0*(2+beta)))*(1-((r0*(1+beta)+rL)/(dv_tes/di_tes)))
        si[:sc_idx] = np.nan
        iva.si[i, :-1] = si


def analyze_iv(iva, psat_level=0.9, save=False, update_cfg=False, show_pb=False):
    """
    Runs main analysis for an IVAnalysis object. This calculates the attributes
    defined in the IVAnalysis class.

    Args
    ----
    iva : IVAnalysis
        IV analysis object
    psat_level : float
        R_frac for which P_sat is defined. Defaults to 0.9
    save : bool
        If true, will save IVAnalysis object after completing
    update_cfg : bool
        If true, will update the device config with the new IV analysis
        filepath
    show_pb : bool
        If true, will display a progress bar for the IV analysis
    """
    am = iva._load_am()
    R_sh = iva.meta['R_sh']

    ts = am.primary.UnixTime / 1e9
    # Calculate phase response and bias_voltages / currents
    for i in trange(iva.nbiases, disable=(not show_pb)):
        # Start from back because analysis is easier low->high voltages
        for j, bg in enumerate(iva.bias_groups):
            t0, t1 = iva.start_times[bg, -(i+1)], iva.stop_times[bg, -(i+1)]
            chan_mask = iva.bgmap == bg
            m = (t0 < ts) & (ts < t1)
            iva.resp[chan_mask, i] = np.nanmean(am.signal[:, m][chan_mask], axis=1)
            if j == 0:
                bias_bits = np.median(am.biases[bg, m])
                iva.v_bias[i] = bias_bits * 2 * iva.meta['rtm_bit_to_volt']

    if iva.run_kwargs['high_current_mode']:
        iva.v_bias *= iva.meta['high_low_current_ratio']

    iva.i_bias = iva.v_bias / iva.meta['bias_line_resistance']

    # Convert phase to Amps
    A_per_rad = iva.meta['pA_per_phi0'] / (2*np.pi) * 1e-12
    iva.resp = (iva.resp.T * iva.polarity).T * A_per_rad

    for i in trange(iva.nchans, disable=(not show_pb)):
        d_resp = np.diff(iva.resp[i])
        dd_resp = np.diff(d_resp)
        dd_resp_abs = np.abs(dd_resp)

        # Find index of superconducting branch
        try:
            sc_idx = np.nanargmax(dd_resp_abs) + 1
        except ValueError:
            continue
        if sc_idx == 1:
            continue
        iva.idxs[i, 0] = sc_idx

        # Find index of normal branch by finding the min index after
        # sc branch. (Skips a few indices after sc branch to avoid possible
        # phase skipping)
        try:
            nb_idx = sc_idx + 1 + np.nanargmin(iva.resp[i, sc_idx+1:])
        except ValueError:
            continue
        iva.idxs[i, 1] = nb_idx
        nb_fit_idx = (iva.nbiases + nb_idx) // 2
        fin_idx = (np.isfinite(iva.i_bias[nb_fit_idx:]) &
                   np.isfinite(iva.resp[i, nb_fit_idx:]))
        try:
            norm_fit = np.polyfit(iva.i_bias[nb_fit_idx:][fin_idx],
                                  iva.resp[i, nb_fit_idx:][fin_idx], 1)
        except:
            continue
        iva.resp[i] -= norm_fit[1]  # Put resp in real current units

        fin_idx = (np.isfinite(iva.i_bias[:sc_idx]) &
                   np.isfinite(iva.resp[i, :sc_idx]))
        try:
            sc_fit = np.polyfit(iva.i_bias[:sc_idx][fin_idx],
                                iva.resp[i, :sc_idx][fin_idx], 1)
        except:
            continue

        # subtract off unphysical y-offset in superconducting branch; this
        # is probably due to an undetected phase wrap at the kink between
        # the superconducting branch and the transition, so it is
        # *probably* legitimate to remove it by hand. 
        iva.resp[i, :sc_idx] -= sc_fit[1]
        sc_fit[1] = 0  # now change s.c. fit offset to 0 for plotting

        R = R_sh * (iva.i_bias/(iva.resp[i]) - 1)
        R_n = np.nanmean(R[nb_fit_idx:])
        R_L = np.nanmean(R[1:sc_idx])

        iva.v_tes[i] = iva.i_bias * R_sh * R / (R + R_sh)
        iva.i_tes[i] = iva.v_tes[i] / R
        iva.p_tes[i] = (iva.v_tes[i]**2) / R
        iva.R[i] = R
        iva.R_n[i] = R_n
        iva.R_L[i] = R_L

    compute_psats(iva, psat_level)
    compute_si(iva)

    if save:
        iva.save(update_cfg=update_cfg)

def plot_Rfracs(iva, Rn_range=(5e-3, 12e-3), bgs=None):
    """
    Plots Stacked Rfrac curves of each channel.
    """
    fig, ax = plt.subplots()
    Rfrac = (iva.R.T / iva.R_n).T
    if bgs is None:
        bgs = np.arange(12)
    bgs = np.atleast_1d(bgs)
    for i, rf in enumerate(Rfrac):
        bg = iva.bgmap[i]
        if bgs is not None:
            if bg not in bgs:
                continue

        if not Rn_range[0] < iva.R_n[i] < Rn_range[1]:
            continue

        if bg == -1:
            continue

        ax.plot(iva.v_bias, rf, alpha=0.1, color=f'C{bg}')
    ax.set(ylim=(0, 1.1))
    ax.set_xlabel("Bias Voltage (V)", fontsize=14)
    ax.set_ylabel("$R_\mathrm{frac}$", fontsize=14)
    return fig, ax

def plot_Rn_hist(iva, range=(0, 10), text_loc=(0.05, 0.05), bbox_props=None):
    """
    Plots summary of channel normal resistances.

    Args
    -----
    iva : IVAnalysis
        IVAnalysis object
    range : tuple
        Range of histogram
    text_loc : tuple
        Location of text box in coordinate frame of axis transform
    bbox_props : dict
        Additional bbox properties to add to the textbox
    """
    fig, ax = plt.subplots()
    hist = ax.hist(iva.R_n*1000, range=range, bins=40)
    ax.axvline(np.nanmedian(iva.R_n)*1000, color='red', ls='--')

    chans_pictured = int(np.sum(hist[0]))
    txt = f"{chans_pictured} / {iva.nchans} channels pictured"
    txt += '\n' + get_plot_text(iva)
    txt += f'\nMedian: {np.nanmedian(iva.R_n)*1000:0.2f} mOhms'   

    bbox = dict(facecolor='wheat', alpha=0.8)
    if bbox_props is not None:
        bbox.update(bbox_props)

    ax.text(*text_loc, txt, bbox=bbox, transform=ax.transAxes)
    ax.set_xlabel("$R_n$ (m$\Omega$)", fontsize=14)
    return fig, ax


def plot_channel_iv(iva, rc):
    """

    Plots anlayzed IV results for a given channel.

    Args
    ----
    iva : IVAnalysis
        Analyzed IVAnalysis instance
    rc : int
        Readout channel to plot
    """
    fig, axes = plt.subplots(4, 1, figsize=(10, 10))
    for ax in axes.ravel():
        v_sc = iva.v_bias[iva.idxs[rc, 0]]
        v_norm = iva.v_bias[iva.idxs[rc, 1]]
        ax.axvspan(0, v_sc, alpha=0.1, color='C0')
        ax.axvspan(v_sc, v_norm, alpha=0.1, color='C2')
        ax.axvspan(v_norm, iva.v_bias[-1], alpha=0.1, color='C1')

    axes[0].plot(iva.v_bias, iva.i_tes[rc], color='black')
    axes[0].set_ylabel("Current (Amps)")
    axes[1].plot(iva.v_bias, iva.R[rc]*1000, color='black')
    axes[1].set_ylabel("R (mOhms)")
    axes[2].plot(iva.v_bias, iva.p_tes[rc]*1e12, color='black')
    axes[2].set_ylabel("Pbias (pW)")
    axes[3].plot(iva.v_bias, iva.si[rc]*1e-6, color='black')
    axes[3].set_ylabel("Si (1/uV)")

    b, c, bg = iva.bands[rc], iva.channels[rc], iva.bgmap[rc]
    axes[0].set_title(f"Band: {b}, Chan: {c}, BG: {bg}", fontsize=18)

    axes[-1].set_xlabel("Voltage (V)")
    return fig, axes

@dataclass
class IVConfig:
    bias_groups: Optional[list] = None
    overbias_voltage: float = 18.0
    overbias_wait: float = 5.0
    high_current_mode: bool = True
    cool_wait: float = 30.
    cool_voltage: Optional[float] = None
    biases: Optional[np.ndarray] = None
    bias_high: float = 18.0
    bias_low: float = 0.0
    bias_step: float = 0.025
    show_plots: bool = True
    wait_time: float = 0.1
    run_analysis: bool = True
    run_serially: bool = False
    serial_wait_time: float = 10.
    g3_tag: Optional[str] = None
    analysis_kwargs: Optional[dict] = None

    def __post_init__(self):
        if self.biases is None:
            self.biases = np.arange(self.bias_high, self.bias_low - self.bias_step, -self.bias_step)
        self.biases = np.sort(self.biases)[::-1]

        if self.analysis_kwargs is None:
            self.analysis_kwargs = {}


@sdl.set_action()
def take_iv(S, cfg, **kwargs):
    """
    Takes an IV.

    This function requires an accurate bias-group-map, so be sure to run
    ``take_bgmap`` before this.

    Args
    ----
    S : SmurfControl
        pysmurf instance
    cfg : DetConfig
        detconfig instance
    bias_groups : list, int
        List of bias groups to run on. Defaults to all 12.
    overbias_voltage : float
        Voltage to use to overbias detectors
    overbias_wait : float
        Time (sec) to wait at overbiased voltage
    high_current_mode : bool
        If True, will run IV in high-current-mode. This is highly recommended
        to avoid the bias-line filter.
    cool_wait : float
        Amout of time to wait at the first bias step after overbiasing.
    cool_voltage : float
        TES bias voltage to sit at during the cool_wait time while the cryostat
        cools
    biases : np.ndarray, optional
        array of biases to use for IV.
        This should be in units of Low-Current-Mode volts. If you are running
        in high-current-mode this will automatically be adjusted for you!!
    bias_high : float
        Upper limit for biases if not manually set. (to be used in np.arange)
        This should be in units of Low-Current-Mode volts. If you are running
        in high-current-mode this will automatically be adjusted for you!!
    bias_low : float
        Lower limit for biases if not manually set. (to be used in np.arange)
        This should be in units of Low-Current-Mode volts. If you are running
        in high-current-mode this will automatically be adjusted for you!!
    bias_step : float
        Step size for biases if not manually set. (to be used in np.arange)
        This should be in units of Low-Current-Mode volts. If you are running
        in high-current-mode this will automatically be adjusted for you!!
    wait_time : float
        Amount of time to wait at each bias point.
    run_analysis : bool
        If True, will automatically run analysis, save it, and update device
        cfg. (unless otherwise specified in analysis_kwargs)
    show_plots : bool
        If true, will show summary plots
    run_serially : bool
        If True, will run IV sweeps in serial, running each of the specified
        bias groups independently instead of all together.
    serial_wait_time : float
        Time to sleep between serial IV sweeps (sec)
    g3_tag : string (optional)
        If not None, overrides default tag "oper,iv" sent to g3 file.
    analysis_kwargs : dict
        Keyword arguments to pass to analysis

    Returns
    -------
    iva : IVAnalysis
        IVAnalysis object. Will be already analyzed if run_analysis=True.
    """
    if not hasattr(S, 'tune_file'):
        raise AttributeError('No tunefile loaded in current '
                             'pysmurf session. Load active tunefile.')
    
    kw = deepcopy(cfg.dev.exp['iv_defaults'])
    kw.update(kwargs)
    ivcfg = IVConfig(**kw)

    if ivcfg.bias_groups is None:
        ivcfg.bias_groups = cfg.dev.exp['active_bgs']
    ivcfg.bias_groups = np.atleast_1d(ivcfg.bias_groups).astype(int)

    run_kwargs = asdict(ivcfg)

    start_times = np.zeros((S._n_bias_groups, len(ivcfg.biases)))
    stop_times = np.zeros((S._n_bias_groups, len(ivcfg.biases)))
    sdl.stop_point(S)
    def overbias_and_sweep(bgs, cool_voltage=None):
        """
        Helper function to run IV sweep for a single bg or group of bgs.
        """
        bgs = np.atleast_1d(bgs).astype(int)
        _bias_arr = S.get_tes_bias_bipolar_array()
        _bias_arr[ivcfg.bias_groups] = 0
        S.set_tes_bias_bipolar_array(_bias_arr)
        if ivcfg.overbias_voltage > 0:
            if cool_voltage is None:
                cool_voltage = np.max(ivcfg.biases)
            S.overbias_tes_all(
                bias_groups=bgs, overbias_wait=ivcfg.overbias_wait,
                tes_bias=cool_voltage, cool_wait=ivcfg.cool_wait,
                high_current_mode=ivcfg.high_current_mode,
                overbias_voltage=ivcfg.overbias_voltage
            )

        S.log(f"Starting TES Bias Ramp on bg {bgs}")
        for i, bias in enumerate(ivcfg.biases):
            sdl.stop_point(S)
            S.log(f"Setting bias to {bias:4.3f}", S.LOG_INFO)
            _bias_arr[bgs] = bias
            S.set_tes_bias_bipolar_array(_bias_arr)
            start_times[bgs, i] = time.time()
            time.sleep(ivcfg.wait_time)
            stop_times[bgs, i] = time.time()

    if ivcfg.high_current_mode:
        ivcfg.biases /= S.high_low_current_ratio
    try:
        sid = sdl.stream_g3_on(S, tag=ivcfg.g3_tag, subtype='iv')
        if ivcfg.run_serially:
            for bg in ivcfg.bias_groups:
                overbias_and_sweep(bg, cool_voltage=ivcfg.cool_voltage)
                time.sleep(ivcfg.serial_wait_time)
        else:
            overbias_and_sweep(ivcfg.bias_groups)
    finally:
        sdl.stream_g3_off(S)

        # Turn off biases and streaming on error
        for bg in ivcfg.bias_groups:
            S.set_tes_bias_bipolar(bg, 0)


    iva = IVAnalysis(S, cfg, run_kwargs, sid, start_times, stop_times)

    if ivcfg.run_analysis:
        sdl.stop_point(S)
        _analysis_kwargs = {'save': True, 'update_cfg': True}
        _analysis_kwargs.update(ivcfg.analysis_kwargs)
        analyze_iv(iva, **_analysis_kwargs)

        # Save and publish plots
        is_interactive = plt.isinteractive()
        try:
            if not ivcfg.show_plots:
                plt.ioff()
            fig, ax = plot_Rfracs(iva)
            fname = sdl.make_filename(S, 'iv_rfracs.png', plot=True)
            fig.savefig(fname)
            S.pub.register_file(fname, 'iv', format='png', plot=True)
            if not ivcfg.show_plots:
                plt.close(fig)

            fig, ax = plot_Rn_hist(iva)
            fname = sdl.make_filename(S, 'iv_rns.png', plot=True)
            fig.savefig(fname)
            S.pub.register_file(fname, 'iv', format='png', plot=True)
            if not ivcfg.show_plots:
                plt.close(fig)
        finally:
            if is_interactive:
                plt.ion()

    return iva


def get_plot_text(iva):
    """
    Gets text to add to text-box for IV plots.
    """
    return '\n'.join([
        f"stream_id: {iva.meta['stream_id']}",
        f"sid: {iva.sid}",
        f"path: {os.path.basename(iva.filepath)}",
    ])

