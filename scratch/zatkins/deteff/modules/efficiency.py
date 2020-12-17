#!/usr/bin/env python3

# author: zatkins
# desc: functions for generating the optical efficiency of given detectors from data

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from scipy import odr as curve_fit_2d

def eff_fit_func(beta, p_opts):
    eta, c = beta
    return c - eta*p_opts

def eff_inv_fit_func(p_sats, eta, c):
    return (c - p_sats) / eta

def eff_fit_1d(p_opts, p_sats, p_opt_errs = None, p0 = [1., 1.], bounds = None):
    assert len(p_opts) == len(p_sats)

    if p_opt_errs is None:
        absolute_sigma = False
    elif np.allclose(p_opt_errs, np.zeros(len(p_opt_errs)), rtol=0, atol=1e-18):
        absolute_sigma = False
    else:
        absolute_sigma = True
    
    if bounds is None: bounds = ((-np.inf, -np.inf), (np.inf, np.inf))

    mean, cov = curve_fit(eff_inv_fit_func, p_sats, p_opts, p0 = p0, sigma = p_opt_errs, \
        absolute_sigma = absolute_sigma, bounds = bounds)
    return mean, cov

def eff_fit_2d(p_opts, p_sats, p_opt_errs = None, p_sat_errs = None, p0 = [1., 1.]):
    
    assert len(p_opts) == len(p_sats)
    
    # because scipy.odr is found to reduce to eff_fit_1d when sx, sy = epsilon
    # but not if sx, sy = None or sy, sx = 0
    if p_opt_errs is None: p_opt_errs = 1e-18 + np.zeros(len(p_opts)) 
    if p_sat_errs is None: p_sat_errs = 1e-18 + np.zeros(len(p_sats))

    # if the passed errors are 0, make them epsilon
    p_opt_errs = p_opt_errs + 1e-18
    p_sat_errs = p_sat_errs + 1e-18

    linear = curve_fit_2d.Model(eff_fit_func)
    data = curve_fit_2d.RealData(p_opts, p_sats, sx = p_opt_errs, sy = p_sat_errs)
    odr = curve_fit_2d.ODR(data, linear, beta0 = p0)
    output = odr.run()

    mean = output.beta
    cov = output.cov_beta
    return mean, cov





