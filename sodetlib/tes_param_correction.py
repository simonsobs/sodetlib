"""
This is a module for a more accurate estimation of TES parameters than what
currently exists in the IV and Bias Step Modules.

It contains the follwing procedures:
  - recompute_ivpars: corrects errors in IV analysis
  - run_correction: Takes IV and bias step data to compute corrected TES params
    based on the RP fitting method described here:
        https://simonsobs.atlassian.net/wiki/spaces/~5570586d07625a6be74c8780e4b96f6156f5e6/blog/2024/02/02/286228683/Nonlinear+TES+model+using+RP+curve

Authors: Remy Gerras, Satoru Takakura, Jack Lashner
"""

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from tqdm.auto import trange
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed, Future
from typing import Optional, List, Tuple, TypeVar
from dataclasses import dataclass
import traceback
from copy import deepcopy
from numpy.typing import NDArray

from sodetlib.operations.iv import IVAnalysis
from sodetlib.operations.bias_steps import BiasStepAnalysis

D = TypeVar("D", float, np.ndarray)


def model_logalpha(r: D, logp0=0.0, p1=7.0, p2=1.0, logp3=0.0) -> D:
    """
    Fits Alpha' parameter, or alpha / GT. P0 has units of [uW]^-1.
    """
    return logp0 + p1 * r - p2 * r ** np.exp(logp3) * np.log(1 - r)


@dataclass
class AnalysisCfg:
    """
    Class for configuring behavior of the parameter estimation functions
    """

    default_nprocs: int = 10
    "Default number of processes to use when running correction in parallel."

    raise_exceptions: bool = False
    "If true, will raise exceptions thrown in the run_correction function."

    psat_level: float = 0.9
    "rfrac for which psat is defined."

    rpfit_min_rfrac: float = 0.05
    "Min rfrac used for RP fits and estimation."

    rpfit_max_rfrac: float = 0.95
    "Max rfrac used for RP fits and estiamtion."

    rpfit_intg_nsamps: int = 500
    "Number of samples to use in the RP fit and integral."

    sc_rfrac_thresh: float = 0.02
    "Bias step rfrac threshold below which TES should be considered "
    "superconducting, and the correction skipped."

    show_pb: bool = True
    "If true, will show a pb when running on all channels together"

    def __post_init__(self):
        if self.rpfit_min_rfrac < 0.0001:
            raise ValueError("rp_fit_min_rfrac must be larger than 0.0001")
        if self.rpfit_max_rfrac > 0.9999:
            raise ValueError("rp_fit_max_rfrac must be smaller than 0.9999")


@dataclass
class RpFitChanData:
    """
    Channel data needed to perform RP correction, pulled from bias step and IV
    calibration functions.

    Args
    -------
    bg: int
        Bias Line
    Rsh: float
        Shunt Resistance [Ohms]
    band: int
        Readout band
    channel :int
        Readout channel
    iv_idx_sc: int
        Index in IV data for SC transition
    iv_resp: np.ndarray
        TES current response relative from IV.
    iv_R: np.ndarray
        TES resistance [Ohms] through IV.
    iv_p_tes: np.ndarray
        TES Bias power [pW] through IV.
    iv_Rn: float
        TES normal resistance, measured from IV.
    iv_ibias: np.ndarray
        TES bias current [uA] throughout IV.
    iv_si: np.ndarray
        TES responsivity [1/uV] throughout IV.
    bs_meas_dIrat: float
        TES measured dIdrat from bias steps.
    bs_Rtes: float
        TES resistance [Ohm] measured from bias steps.
    bs_Ibias: float
        TES bias current [uA] measured from bias steps.
    bs_Si: float
        TES responsivity [1/uV] measured from bias steps.
    """

    bg: int
    Rsh: float

    band: int
    channel: int

    iv_idx_sc: int
    iv_resp: np.ndarray
    iv_R: np.ndarray  # Ohm
    iv_p_tes: np.ndarray  # pW
    iv_Rn: float
    iv_ibias: np.ndarray  # uA
    iv_si: np.ndarray  # 1/uV

    bs_meas_dIrat: float
    bs_Rtes: float = np.nan  # Ohm
    bs_Ibias: float = np.nan  # uA
    bs_Si: float = np.nan  # 1/uV

    @classmethod
    def from_data(
        cls, iva: IVAnalysis, bsa: BiasStepAnalysis, band: int, channel: int
    ) -> "RpFitChanData":
        """Create class from IV and BSA dicts"""
        iv_idx = np.where((iva.channels == channel) & (iva.bands == band))[0]
        if len(iv_idx) == 0:
            raise ValueError(
                f"Could not find band={band} channel={channel} in IVAnalysis"
            )
        iv_idx = iv_idx[0]

        bs_idx = np.where((bsa.channels == channel) & (bsa.bands == band))[0]
        if len(bs_idx) == 0:
            raise ValueError(
                f"Could not find band={band} channel={channel} in Bias Steps"
            )
        bs_idx = bs_idx[0]

        bg = iva.bgmap[iv_idx]

        return cls(
            bg=bg,
            band=band,
            channel=channel,
            Rsh=iva.meta["R_sh"],
            iv_idx_sc=iva.idxs[iv_idx, 0],
            iv_resp=iva.resp[iv_idx],
            iv_R=iva.R[iv_idx],
            iv_Rn=iva.R_n[iv_idx],
            iv_p_tes=iva.p_tes[iv_idx] * 1e12,
            iv_si=iva.si[iv_idx] * 1e-6,
            iv_ibias=iva.i_bias * 1e6,
            bs_meas_dIrat=bsa.dItes[bs_idx] / bsa.dIbias[bg],
            bs_Rtes=bsa.R0[bs_idx],
            bs_Ibias=bsa.Ibias[bg] * 1e6,
            bs_Si=bsa.Si[bs_idx] * 1e-6,
        )


@dataclass
class CorrectedParams:
    "Log(alpha') fit popts"

    logalpha_popts: np.ndarray
    "Log alpha fit parameters"
    pbias_model_offset: float
    "delta Popt between IV and bias steps"
    delta_Popt: float
    "TES Current  [uA]"
    corrected_I0: float
    "TES resistance [Ohm]"
    corrected_R0: float
    "Bias power [aW]"
    corrected_Pj: float
    "Responsivity [1/uV]"
    corrected_Si: float
    "Loopgain at bias current"
    loopgain: float


@dataclass
class CorrectionResults:
    """
    Results from the parameter correction procedure
    """

    chdata: RpFitChanData
    "Channel data used for correction"

    success: bool = False
    "If correction finished successfully"

    traceback: Optional[str] = None
    "Traceback on failure"

    corrected_params: Optional[CorrectedParams] = None
    "Model popts and corrected TES parameters"


def find_logalpha_popts(
    chdata: RpFitChanData, cfg: AnalysisCfg
) -> Tuple[np.ndarray, float]:
    """
    Fit Log(alpha') to IV data.

    Returns
    ---------
    popts: np.ndarray
        Optimal parameters for log(alpha') fit.
    pbias_model_offset: float
        Offset between pbias_model from alpha' integration and data [aW]
    """
    smooth_dist = 5
    w_len = 2 * smooth_dist + 1
    w = (1.0 / float(w_len)) * np.ones(w_len)  # window

    rfrac = chdata.iv_R / chdata.iv_Rn
    mask = np.ones_like(rfrac, dtype=bool)
    mask[: chdata.iv_idx_sc + w_len + 2] = False  # Try to cut out SC branch + smoothing
    mask &= (rfrac > cfg.rpfit_min_rfrac) * (rfrac < cfg.rpfit_max_rfrac)
    rfrac = np.convolve(rfrac, w, mode="same")
    pbias = np.convolve(chdata.iv_p_tes, w, mode="same")

    rfrac = rfrac[mask]
    pbias = pbias[mask]
    drfrac = np.diff(rfrac, prepend=np.nan)
    dpbias = np.diff(pbias, prepend=np.nan)
    logalpha = np.log(1 / rfrac * drfrac / dpbias)
    logalpha_mask = np.isfinite(logalpha)

    def logalpha_fit_func(logalpha_pars):
        model = model_logalpha(rfrac, *logalpha_pars)
        chi2 = np.sum((logalpha[logalpha_mask] - model[logalpha_mask]) ** 2)
        return chi2

    fitres = minimize(logalpha_fit_func, [4, 10, 1, 1])
    logalpha_popts = fitres.x

    logalpha_model = model_logalpha(rfrac, *fitres.x)

    pbias_model = np.nancumsum(drfrac / (rfrac * np.exp(logalpha_model)))

    def pbias_offset_fit_func(pbias_offset):
        return np.sum((pbias_model + pbias_offset - pbias) ** 2)

    pbias_model_offset = minimize(pbias_offset_fit_func, pbias[0]).x[0]

    return logalpha_popts, pbias_model_offset


def run_correction(chdata: RpFitChanData, cfg: AnalysisCfg) -> CorrectionResults:
    """
    Runs TES param correction procedure.

    Args
    ----
    chdata: RpFitChanData
        IV and Bias Step calibration data
    cfg: AnalysisCfg
        Correction analysis configuration object.
    """
    res = CorrectionResults(chdata)
    try:
        if np.isnan(chdata.iv_Rn):
            raise ValueError("IV Rn is NaN")

        # Find logalpha popts
        logalpha_popts, pbias_model_offset = find_logalpha_popts(chdata, cfg)

        if np.abs(chdata.bs_Rtes / chdata.iv_Rn) < cfg.sc_rfrac_thresh:
            # Cannot perform correction, since detectors are SC
            res.corrected_params = CorrectedParams(
                logalpha_popts=logalpha_popts,
                pbias_model_offset=pbias_model_offset,
                delta_Popt=np.nan,
                corrected_I0=0.0,
                corrected_R0=0.0,
                corrected_Pj=0.0,
                corrected_Si=np.nan,
                loopgain=np.nan,
            )
            res.success = True
            return res

        # compute dPopt by minimizing (dIrat_IV - dIrat_BS) at chosen bias voltage with respect to delta_popt
        dIrat_meas = chdata.bs_meas_dIrat
        Rsh = chdata.Rsh
        Ibias_setpoint = chdata.bs_Ibias
        rfrac = np.linspace(
            cfg.rpfit_min_rfrac, cfg.rpfit_max_rfrac, cfg.rpfit_intg_nsamps
        )
        dr = rfrac[1] - rfrac[0]
        logalpha_model = model_logalpha(rfrac, *logalpha_popts)
        alpha = np.exp(logalpha_model)
        R = rfrac * chdata.iv_Rn
        pbias = np.cumsum(dr / (rfrac * np.exp(logalpha_model))) + pbias_model_offset

        def dIrat_IV(dPopt):
            P_ = pbias - dPopt
            L0_ = alpha * P_
            i_tes_ = np.sqrt(P_ / R)
            i_bias_ = i_tes_ * (R + Rsh) / Rsh
            irat = (1.0 - L0_) / ((1.0 - L0_) + (1.0 + L0_) * R / Rsh)
            return np.interp(Ibias_setpoint, i_bias_, irat)

        def fit_func(dPopt):
            diff = (dIrat_IV(dPopt) - dIrat_meas) ** 2
            return diff if not np.isnan(diff) else np.inf

        dPopt_fitres = minimize(fit_func, [0])
        if not dPopt_fitres.success:
            raise RuntimeError("dPopt fit failed")

        dPopt = dPopt_fitres.x[0]

        # Adjust IV parameter curves with delta_Popt, and find params at Ibias setpoint
        pbias -= dPopt
        ok = (pbias > 0.0) * (R > Rsh)
        pbias = pbias[ok]
        R = R[ok]
        L0 = alpha[ok] * pbias
        L = (R - Rsh) / (R + Rsh) * L0
        Ites = np.sqrt(pbias / R)
        Ibias = Ites * (R + Rsh) / Rsh
        Si = -1 / (Ites * (R - Rsh)) * (L / (L + 1))

        res.corrected_params = CorrectedParams(
            logalpha_popts=logalpha_popts,
            pbias_model_offset=pbias_model_offset,
            delta_Popt=dPopt,
            corrected_I0=np.interp(Ibias_setpoint, Ibias, Ites).item(),
            corrected_R0=np.interp(Ibias_setpoint, Ibias, R).item(),
            corrected_Pj=np.interp(Ibias_setpoint, Ibias, pbias).item(),
            corrected_Si=np.interp(Ibias_setpoint, Ibias, Si).item(),
            loopgain=np.interp(Ibias_setpoint, Ibias, L0).item(),
        )
        res.success = True
    except Exception:
        if cfg.raise_exceptions:
            raise
        else:
            res.traceback = traceback.format_exc()
            res.success = False

    return res


def run_corrections_parallel(
        iva: IVAnalysis, bsa: BiasStepAnalysis, cfg: AnalysisCfg, nprocs=None,
        executor=None, as_completed_callable=None) -> List[CorrectionResults]:
    """
    Runs correction procedure in parallel for all channels in IV and BSA object
    """

    nchans = iva.nchans
    pb = trange(nchans, disable=(not cfg.show_pb))
    results = []

    if nprocs is None:
        nprocs = cfg.default_nprocs

    # Error handler
    def errback(e):
        raise e

    # Create executor (optionally externally provided)
    if executor is None:
        executor = ProcessPoolExecutor(max_workers=nprocs)
        as_completed_callable = as_completed
        close_executor = True
    else:
        close_executor = False

    try:
        futures = []
        for idx in range(nchans):
            chdata = RpFitChanData.from_data(
                iva, bsa, iva.bands[idx], iva.channels[idx]
            )
            future = executor.submit(run_correction, chdata, cfg)
            futures.append(future)

        for future in as_completed_callable(futures):
            try:
                res = future.result()
                results.append(res)
            except Exception as e:
                errback(e)

            futures.remove(future)
            pb.update(1)

        pb.close()

    finally:
        if close_executor:
            executor.shutdown(wait=True, cancel_futures=True)

    return results

def compute_psats(iva: IVAnalysis, cfg: AnalysisCfg) -> Tuple[np.ndarray, np.ndarray]:
    """
    Re-computes Psat for an IVAnalysis object. Will save results to iva.p_sat.
    This assumes i_tes, v_tes, and r_tes have already been calculated. This will
    not modify the original IVAnalysis object.

    Args
    ----
    iva2 : Dictionary
        Dictionary built from original IV Analysis .npy file
    psat_level : float
        R_frac level for which Psat is defined. If 0.9, Psat will be the
        power on the TES when R_frac = 0.9.

    Returns
    -------
    p_sat : np.ndarray
        Array of length (nchan) with the p-sat computed for each channel (W)
    psat_cross_idx : np.ndarray
        Array of indices at which the psat level is crossed for each channel
    """
    # calculates P_sat as P_TES when Rfrac = psat_level
    # if the TES is at 90% R_n more than once, just take the first crossing
    psats = np.full(iva.nchans, np.nan)
    psat_cross_idx = np.full(iva.nchans, -1)

    for i in range(iva.nchans):
        R = iva.R[i]
        R_n = iva.R_n[i]
        p_tes = iva.p_tes[i]

        if np.isnan(R_n):
            continue

        cross_idx = np.where(R / R_n > cfg.psat_level)[0]
        if len(cross_idx) == 0:
            continue

        # Takes cross-index to be the first time Rfrac crosses psat_level
        cross_idx = cross_idx[0]
        if cross_idx == 0:
            continue

        psat_cross_idx[i] = cross_idx
        try:
            psat = interp1d(
                R[cross_idx - 1 : cross_idx + 1] / R_n,
                p_tes[cross_idx - 1 : cross_idx + 1],
            )(cfg.psat_level)
        except ValueError:
            continue
        psats[i] = psat

    return psats, psat_cross_idx


def compute_si(iva: IVAnalysis) -> np.ndarray:
    """
    Recalculates responsivity using the thevenin equivalent voltage. This will
    not modify the original IVAnalysis object.

    Args
    ----
    iva : IVAnalysis
        IVAnalysis object for which you want to compute Si. This should already
        have items like R, R_n, R_L, i_tes, v_tes, computed.

    Returns
    -------
    si : np.ndarray
        Array of length (nchan, nbiases) with  the responsivity as a fn of
        thevenin equivalent voltage for each channel (V^-1).
    """
    si_all = np.full(iva.si.shape, np.nan)

    smooth_dist = 5
    w_len = 2 * smooth_dist + 1
    w = (1.0 / float(w_len)) * np.ones(w_len)  # window

    v_thev_smooth = np.convolve(iva.v_thevenin, w, mode="same")
    dv_thev = np.diff(v_thev_smooth)

    for i in range(iva.nchans):
        sc_idx = iva.idxs[i, 0]

        if np.isnan(iva.R_n[i]) or sc_idx == -1:
            continue

        # Running average
        i_tes_smooth = np.convolve(iva.i_tes[i], w, mode="same")
        v_tes_smooth = np.convolve(iva.v_tes[i], w, mode="same")
        r_tes_smooth = v_tes_smooth / i_tes_smooth

        R_L = iva.R_L[i]

        # Take derivatives
        di_tes = np.diff(i_tes_smooth)
        dv_tes = np.diff(v_tes_smooth)
        R_L_smooth = np.ones(len(r_tes_smooth - 1)) * R_L
        R_L_smooth[:sc_idx] = dv_tes[:sc_idx] / di_tes[:sc_idx]
        r_tes_smooth_noStray = r_tes_smooth - R_L_smooth
        i0 = i_tes_smooth[:-1]
        r0 = r_tes_smooth_noStray[:-1]
        rL = R_L_smooth[:-1]
        beta = 0.0

        # artificially setting rL to 0 for now,
        # to avoid issues in the SC branch
        # don't expect a large change, given the
        # relative size of rL to the other terms
        # rL = 0

        # Responsivity estimate, derivation done here by MSF
        # https://www.overleaf.com/project/613978cb38d9d22e8550d45c
        si = -(1.0 / (i0 * r0 * (2 + beta))) * (
            1 - ((r0 * (1 + beta) + rL) / (dv_thev / di_tes))
        )
        si[:sc_idx] = np.nan
        si_all[i, :-1] = si

    return si_all


def recompute_ivpars(iva: IVAnalysis, cfg: AnalysisCfg) -> IVAnalysis:
    """
    Takes in an IV Analysis object and analysis cfg params, and recomputes
    TES voltage, current, bias power, resistance, responsivity, and saturation
    powers using the corrected TES formulas for R_L and V_thevenin.
    """
    iva_new = IVAnalysis.from_dict(deepcopy(iva.to_dict()))
    R_sh = iva.meta["R_sh"]
    R_bl = iva.meta["bias_line_resistance"]
    iva_new.i_bias = iva_new.v_bias / R_bl
    iva_new.v_thevenin = iva_new.i_bias * R_sh
    iva_new.v_tes = np.full(iva.v_tes.shape, np.nan)
    iva_new.i_tes = np.full(iva.i_tes.shape, np.nan)
    iva_new.p_tes = np.full(iva.p_tes.shape, np.nan)
    iva_new.R = np.full(iva.R.shape, np.nan)
    iva_new.R_n = np.full(iva.R_n.shape, np.nan)
    iva_new.R_L = np.full(iva.R_L.shape, np.nan)

    iva_new.resp
    for i in range(iva.nchans):
        sc_idx = iva.idxs[i, 0]
        nb_idx = iva.idxs[i, 1]

        R: NDArray[np.float] = R_sh * (iva.i_bias / iva.resp[i] - 1)
        R_par: float = np.nanmean(R[1:sc_idx])
        R_n: float = np.nanmean(R[nb_idx:]) - R_par
        R_L: float = R_sh + R_par
        R_tes: NDArray[np.float] = R - R_par

        iva_new.v_tes[i] = iva_new.v_thevenin * (R_tes / (R_tes + R_L))
        iva_new.i_tes[i] = iva_new.v_tes[i] / R_tes
        iva_new.p_tes[i] = iva_new.v_tes[i] ** 2 / R_tes
        iva_new.R[i] = R_tes
        iva_new.R_n[i] = R_n
        iva_new.R_L[i] = R_L

    iva_new.p_sat, iva_new.idxs[:, 2] = compute_psats(iva_new, cfg)
    iva_new.si = compute_si(iva_new)

    return iva_new
