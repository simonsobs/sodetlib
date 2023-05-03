import numpy as np
import sodetlib as sdl
import time
import matplotlib.pyplot as plt
from sodetlib.operations.iv import IVAnalysis
from sodetlib.operations import uxm_setup, uxm_relock, tracking, bias_steps, iv, bias_dets
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

def biasstep_rebias(
        S, cfg, target_percentage_rn = 0.5, bias_groups=None,
        math_only=False):

    """
    Scripts work well when detectors are already in transistion. For extreme cases
    such as detectors are all in SC, it takes a few run to put them perfectly in
    transition. Next step is making this script can handle extreme cases faster


    """


    ## take the initial biasstep
    S.log("taking the first biasstep")
    bsa_0 = bias_steps.take_bias_steps(S, cfg)
    if not math_only:
        bias_steps.plot_Rfrac(bsa_0)

    repeat_biasstep = False
    ## check for script failue
    for bl in bias_groups:
        mask_bg = np.where(bsa_0.bgmap==bl)[0]
        per_bl_R0 = bsa_0.R0[mask_bg]
        if np.isnan(np.nanmedian(per_bl_R0)):
            ## the yield is very low, could it be the biasstep analysis script failed? repeat again
            repeat_biasstep = True
            S.log("the biasstep script might have failed, going to repeat the measurement")

    if repeat_biasstep:
        S.log("retaking biasstep due to failure")
        bsa_0 = bias_steps.take_bias_steps(S, cfg)
        repeat_biasstep = False
    
    percentage_rn_0 = bsa_0.R0/bsa_0.R_n_IV
    ## add in checks if detectors are SC  
    bg_overbias_needed = []
    overbias_needed = False
    for bl in bias_groups:
        mask_bg = np.where(bsa_0.bgmap==bl)[0]
        per_bl_percentage_rn_0 = percentage_rn_0[mask_bg]
        ## mask for SC
        mask_sc = np.where((per_bl_percentage_rn_0 < 0.1) & (per_bl_percentage_rn_0 > -0.1))
        ## if more than 10 percent of the detectors are SC
        if len(mask_sc[0]) > 0.1*len(mask_bg):
            bg_overbias_needed.append(bl)
            overbias_needed = True

    ## add in check if detectors are normal, hence we should increase the drop step
    bg_detectors_normal = []
    drop_from_normal = False
    for bl in bias_groups:
        mask_bg = np.where(bsa_0.bgmap==bl)[0]
        per_bl_percentage_rn_0 = percentage_rn_0[mask_bg]
        ## mask for normal
        mask_normal = np.where((per_bl_percentage_rn_0 > 0.9) | (per_bl_percentage_rn_0 < -0.9))
        ## if more than half of the detectors are normal
        if len(mask_normal[0]) > 0.5*len(mask_bg):
            bg_detectors_normal.append(bl)   
            drop_from_normal = True


    if overbias_needed:
        S.log("some BL are stuck in SC, going to over bias them, this takes about 4mins")
        previous_dc_biases = S.get_tes_bias_bipolar_array()
        
        sdl.overbias_dets(S, cfg, bias_groups=bg_overbias_needed)
        ##force wait time 3 mins for UFM to reach equalibrium 
        for sleep_index in range(3):
            time.sleep(60)
        safe_dc_biases = previous_dc_biases.copy()
        for replace_bg in bg_overbias_needed:
            safe_dc_biases[replace_bg] = cfg.dev.bias_groups[replace_bg]["testbed_100mK_bias_voltage"]
            
        S.set_tes_bias_bipolar_array(safe_dc_biases)  
        bsa_0 = bias_steps.take_bias_steps(S, cfg)
        bias_steps.plot_Rfrac(bsa_0)
        
        ## add in check if detectors are normal now
        percentage_rn_0 = bsa_0.R0/bsa_0.R_n_IV
        drop_from_normal = False
        for bl in bg_overbias_needed:
            mask_bg = np.where(bsa_0.bgmap==bl)[0]
            per_bl_percentage_rn_0 = percentage_rn_0[mask_bg]
            ## mask for normal
            mask_normal = np.where((per_bl_percentage_rn_0 > 0.9) | (per_bl_percentage_rn_0 < -0.9))
            ## if more than half of the detectors are normal
            if len(mask_normal[0]) > 0.5*len(mask_bg):
                bg_detectors_normal.append(bl)   
                drop_from_normal = True


    ##preparing for the 2nd step
    initial_dc_biases = S.get_tes_bias_bipolar_array()
    new_bias_voltage = initial_dc_biases.copy()

    ## load IV analysis result so can use dynamic step size
    iva = iv.IVAnalysis.load(bsa_0.meta['iv_file'])


    for bl in bias_groups:
        mask_bg = np.where(bsa_0.bgmap==bl)[0]

        per_bl_percentage_rn_0 = percentage_rn_0[mask_bg]
            
        med_per_bl_percentage_rn = np.nanmedian(per_bl_percentage_rn_0)
        
        ## use the latest IV to decide the step size, might be tricky to get a good step size in field
        v_sc = iva.v_bias[iva.idxs[[mask_bg], 0]]
        v_norm = iva.v_bias[iva.idxs[[mask_bg], 1]]
        v_spread = np.nanmedian(v_norm) - np.nanmedian(v_sc)
        ## take 15% of the spread as the step size, so the next biasstep will be taken from 0.25 ~ 0.75 percent Rn after considering the spread
        delta_dc_bias = 0.15 * v_spread
        
        
        ## if the current bias point is below 50% Rn, increase V_biaa
        if med_per_bl_percentage_rn < 0.5:
            new_bias_voltage[bl] =  initial_dc_biases[bl] + delta_dc_bias
        if med_per_bl_percentage_rn >= 0.5:
            new_bias_voltage[bl] =  initial_dc_biases[bl] - delta_dc_bias
        ## if the detector is normal, take a larger step
        if bl in bg_detectors_normal:
            if initial_dc_biases[bl] > cfg.dev.bias_groups[bl]["testbed_100mK_bias_voltage"]:
                initial_dc_biases[bl] = cfg.dev.bias_groups[bl]["testbed_100mK_bias_voltage"]
            new_bias_voltage[bl] =  initial_dc_biases[bl] - 0.5 * v_spread

    S.log("applying new voltage for 2nd biasstep")
    S.set_tes_bias_bipolar_array(new_bias_voltage)  

    ## taking the second bias step
    S.log("taking the 2nd biasstep")
    bsa_1 = bias_steps.take_bias_steps(S, cfg)
    if not math_only:
        bias_steps.plot_Rfrac(bsa_1)

    ## check for script failue
    for bl in bias_groups:
        mask_bg = np.where(bsa_1.bgmap==bl)[0]
        per_bl_R0 = bsa_1.R0[mask_bg]
        if np.isnan(np.nanmedian(per_bl_R0)):
            ## the yield is very low, could it be the biasstep analysis script failed? repeat again
            repeat_biasstep = True
            S.log("the biasstep script might have failed, going to repeat the measurement")

    if repeat_biasstep:
        bsa_1 = bias_steps.take_bias_steps(S, cfg)
        repeat_biasstep = False

    percentage_rn_1 = bsa_1.R0/bsa_1.R_n_IV

    ## add in check if detectors were normal are still normal
    if drop_from_normal:
        S.log(f"some biaslines were in normal state during 1st biasstep: {bg_detectors_normal}")
        drop_from_normal = False
        for bl in bg_detectors_normal:
            mask_bg = np.where(bsa_1.bgmap==bl)[0]
            per_bl_percentage_rn_1 = percentage_rn_1[mask_bg]
            ## mask for normal
            mask_normal = np.where((per_bl_percentage_rn_1 > 0.9) | (per_bl_percentage_rn_1 < -0.9))
            ## if more than half of the detectors are normal
            if len(mask_normal[0]) > 0.5*len(mask_bg):
                drop_from_normal = True

    while drop_from_normal:
        S.log(f"some biaslines are still in normal state: {bg_detectors_normal}")
        percentage_rn_1 = bsa_1.R0/bsa_1.R_n_IV
        S.log("droping detectors from normal to transition")
        previous_dc_biases_dfn = S.get_tes_bias_bipolar_array()

        S.log(f"current tes bias voltages are: {previous_dc_biases_dfn}")
        for bl in bg_detectors_normal:
            mask_bg = np.where(bsa_1.bgmap==bl)[0]
            v_sc = iva.v_bias[iva.idxs[[mask_bg], 0]]
            v_norm = iva.v_bias[iva.idxs[[mask_bg], 1]]
            v_spread = np.nanmedian(v_norm) - np.nanmedian(v_sc)
            previous_dc_biases_dfn[bl] =  previous_dc_biases_dfn[bl] - 0.5 * v_spread
        
        S.set_tes_bias_bipolar_array(previous_dc_biases_dfn)  
        bsa_1 = bias_steps.take_bias_steps(S, cfg)
        percentage_rn_1 = bsa_1.R0/bsa_1.R_n_IV
        drop_from_normal = False
        bg_detectors_normal = []
        for bl in bias_groups:
            mask_bg = np.where(bsa_1.bgmap==bl)[0]
            per_bl_percentage_rn_1 = percentage_rn_1[mask_bg]
            ## mask for normal
            mask_normal = np.where((per_bl_percentage_rn_1 > 0.9) | (per_bl_percentage_rn_1 < -0.9))
            ## if more than half of the detectors are normal
            if len(mask_normal[0]) > 0.5*len(mask_bg):
                drop_from_normal = True
                bg_detectors_normal.append(bl)
        for bl in bg_detectors_normal:
            if previous_dc_biases_dfn[bl] < 1:
                drop_from_normal = False

    target_percentage_rn_array = target_percentage_rn * np.ones(len(percentage_rn_0))
    ## getting result from the previous 2
    v0 = np.zeros(bsa_0.bgmap.shape[0])
    for bl in range(12):
        mask_bl = np.where(bsa_0.bgmap==bl)[0]
        v0[mask_bl] = initial_dc_biases[bl]

    v1 = np.zeros(bsa_1.bgmap.shape[0])
    for bl in range(12):
        mask_bl = np.where(bsa_1.bgmap==bl)[0]
        v1[mask_bl] = new_bias_voltage[bl]   
    ## deciding the new bias point
    vbias_estimate_array = (v0*(target_percentage_rn_array-percentage_rn_1) + v1*(percentage_rn_0 - target_percentage_rn_array)) / (percentage_rn_0 - percentage_rn_1)

    vbias_estimate = initial_dc_biases.copy()
    for bl in bias_groups:
        mask_bg = np.where(bsa_0.bgmap==bl)[0]

        per_bl_vbias_estimate = vbias_estimate_array[mask_bg]
        med_per_bl_vbias_estimate = np.nanmedian(per_bl_vbias_estimate)
        ## if the current bias point is below 50% Rn, increase V_biaa
        vbias_estimate[bl] = med_per_bl_vbias_estimate

    S.log("applying the new bias voltages")
    S.set_tes_bias_bipolar_array(vbias_estimate)  

    S.log("taking the final biasstep")
    bsa_2 = bias_steps.take_bias_steps(S, cfg)
    if not math_only:
        bias_steps.plot_Rfrac(bsa_2)
        
    return bsa_2








