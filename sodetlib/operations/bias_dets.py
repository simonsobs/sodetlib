import numpy as np
import sodetlib as sdl
import time
import matplotlib.pyplot as plt
from sodetlib.operations.iv import IVAnalysis

np.seterr(all="ignore")


sdl.set_action()
def bias_to_rfrac_range(
        S, cfg, rfrac_range=(0.3, 0.6), bias_groups=None, iva=None,
        overbias=True, Rn_range=(5e-3, 12e-3),
        math_only=False):
    """
    Biases detectors to transition given an rfrac range. This function will choose
    TES bias voltages for each bias-group that maximize the number of channels
    that will be placed in the allowable rfrac-range.

    Args
    ----
    S : SmurfControl
        Pysmurf instance
    cfg : DetConfig
        Detconfig instance
    rfrac_range : tuple
        Range of allowable rfracs
    bias_groups : list, optional
        Bias groups to bias. Defaults to all of them.
    iva : IVAnalysis, optional
        IVAnalysis object. If this is passed, will use it to determine
        bias voltages. Defaults to using value in device cfg.
    overbias_voltage : float
        Voltage to use to overbias detectors
    overbias_wait : float
        Time (sec) to wait at overbiased voltage
    Rn_range : tuple
        A "reasonable" range for the TES normal resistance. This will
        be cut on when determining which IV's should be used to determine
        the optimal bias-voltage.
    math_only : bool
        If this is set, will not actually over-bias voltages, and will
        just return the target biases.

    Returns
    ----------
    biases : np.ndarray
        Array of Smurf bias voltages. Note that this includes all smurf
        bias lines (all 15 of them), but only voltages for the requested
        bias-groups will have been modified.
    """
    if bias_groups is None:
        bias_groups = cfg.dev.exp['active_bgs']
    bias_groups = np.atleast_1d(bias_groups)

    if iva is None:
        iva = IVAnalysis.load(cfg.dev.exp['iv_file'])

    biases = S.get_tes_bias_bipolar_array()

    Rfrac = (iva.R.T / iva.R_n).T
    in_range = (rfrac_range[0] < Rfrac) & (Rfrac < rfrac_range[1])

    for bg in bias_groups:
        m = (iva.bgmap == bg)
        m = m & (Rn_range[0] < iva.R_n) & (iva.R_n < Rn_range[1])

        if not m.any():
            continue

        nchans_in_range = np.sum(in_range[m, :], axis=0)
        target_idx = np.nanargmax(nchans_in_range)
        biases[bg] = iva.v_bias[target_idx]

    if math_only:
        return biases

    S.log(f"Target biases: ")
    for bg in bias_groups:
        S.log(f"BG {bg}: {biases[bg]:.2f}")

    if overbias:
        sdl.overbias_dets(S, cfg, bias_groups=bias_groups)
    S.set_tes_bias_bipolar_array(biases)

    return biases

def bias_to_volt_arr(S, cfg, biases, bias_groups=None, overbias=True):
    """
    Biases detectors using an array of voltages.

    Args
    -----
    S : SmurfControl
        Pysmurf instance
    cfg : DetConfig
        det config instance
    biases : np.ndarray
        This must be an array of length 12, with the ith element being the
        voltage bias of bias group i. Note that the bias of any bias group
        that is not active, or not specified in the ``bias_groups`` parameter
        will be ignored.
    bias_groups : list
        List of bias groups to bias. If None, this will default to the active
        bgs specified in the device cfg.
    overbias : bool
        If true, will overbias specified bias lines. If false, will set bias
        voltages without overbiasing detectors.
    """
    if bias_groups is None:
        bias_groups = cfg.dev.exp['active_bgs']
    bias_groups = np.atleast_1d(bias_groups)

    _biases = S.get_tes_bias_bipolar_array()
    for bg in bias_groups:
        _biases[bg] = biases[bg]

    S.log(f"Target biases: ")
    for bg in bias_groups:
        S.log(f"BG {bg}: {biases[bg]:.2f}")

    if overbias:
        sdl.overbias_dets(S, cfg, bias_groups=bias_groups)

    S.set_tes_bias_bipolar_array(biases)
    return


sdl.set_action()
def bias_to_rfrac(S, cfg, rfrac=0.5, bias_groups=None, iva=None,
                  overbias=True, Rn_range=(5e-3, 12e-3), math_only=False):
    """
    Biases detectors to a specified Rfrac value

    Args
    ----
    S : SmurfControl
        Pysmurf instance
    cfg : DetConfig
        Detconfig instance
    rfrac : float
        Target rfrac. Defaults to 0.5
    bias_groups : list, optional
        Bias groups to bias. Defaults to all of them.
    iva : IVAnalysis, optional
        IVAnalysis object. If this is passed, will use it to determine
        bias voltages. Defaults to using value in device cfg.
    overbias_voltage : float
        Voltage to use to overbias detectors
    overbias_wait : float
        Time (sec) to wait at overbiased voltage
    Rn_range : tuple
        A "reasonable" range for the TES normal resistance. This will
        be cut on when determining which IV's should be used to determine
        the optimal bias-voltage.
    math_only : bool
        If this is set, will not actually over-bias voltages, and will
        just return the target biases.

    Returns
    ----------
    biases : np.ndarray
        Array of Smurf bias voltages. Note that this includes all smurf
        bias lines (all 15 of them), but only voltages for the requested
        bias-groups will have been modified.
    """
    if bias_groups is None:
        bias_groups = cfg.dev.exp['active_bgs']
    bias_groups = np.atleast_1d(bias_groups)

    if iva is None:
        iva = IVAnalysis.load(cfg.dev.exp['iv_file'])

    biases = S.get_tes_bias_bipolar_array()

    Rfrac = (iva.R.T / iva.R_n).T
    for bg in bias_groups:
        m = (iva.bgmap == bg)
        m = m & (Rn_range[0] < iva.R_n) & (iva.R_n < Rn_range[1])

        if not m.any():
            continue

        target_biases = []
        for rc in np.where(m)[0]:
            target_idx = np.nanargmin(np.abs(Rfrac[rc] - rfrac))
            target_biases.append(iva.v_bias[target_idx])
        biases[bg] = np.median(target_biases)

    if math_only:
        return biases

    S.log(f"Target biases: ")
    for bg in bias_groups:
        S.log(f"BG {bg}: {biases[bg]:.2f}")

    if overbias:
        sdl.overbias_dets(S, cfg, bias_groups=bias_groups)

    S.set_tes_bias_bipolar_array(biases)

    return biases

