#!/usr/bin/env python3

# author: zatkins
# desc: functions for generating the optical efficiency of given detectors from data

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit

def get_psat(data_dict, band, chan, level = 0.9, greedy = False):
    '''Returns the conventional P_sat from a SMuRF IV curve dictionary.

    Parameters
    ----------
    data_dict : dict
        The dictionary containing the IV curve data
    band : int
        The smurf band to extract from the dict
    chan : int
        The channel to extract from the band
    level : float
        The definition of P_sat, i.e. power when R = level*R_n
    greedy : bool, optional
        If True, will return -1000 if the R/Rn curve crosses the level more than once, by default False.
        If False, returns the power at the first instance when R/Rn crosses the level.

    Returns
    -------
    float
        The conventional P_sat
    '''
    chan_data = data_dict[band][chan]

    p = chan_data['p_tes']
    rn = chan_data['R']/chan_data['R_n']

    cross_idx = np.where(np.logical_and(rn - level >= 0, np.roll(rn - level, 1) < 0))[0]
    try:
        assert len(cross_idx) == 1
    except AssertionError:
        if greedy: return -1000
        else: cross_idx = cross_idx[:1]
    cross_idx = cross_idx[0]

    rn2p = interp1d(rn[cross_idx-1:cross_idx+1], p[cross_idx-1:cross_idx+1])
    return rn2p(level)

def eff_fit_func(p_opts, eta, c):
    return c - eta*p_opts

def eff_inv_fit_func(p_sats, eta, c):
    return (c - p_sats) / eta

def eff_fit(p_opts, p_sats, p_opts_errs = None, p0 = [1., 0.], bounds = None):
    assert len(p_opts) == len(p_sats)

    if p_opts_errs is None:
        absolute_sigma = False
    elif np.allclose(p_opts_errs, np.zeros(len(p_opts_errs)), rtol=0, atol=1e-16):
        absolute_sigma = False
    else:
        absolute_sigma = True
    
    if bounds is None: bounds = ((-np.inf, -np.inf), (np.inf, np.inf))

    mean, cov = curve_fit(eff_inv_fit_func, p_sats, p_opts, p0 = p0, sigma = p_opts_errs, \
        absolute_sigma = absolute_sigma, bounds = bounds)
    return mean, cov

