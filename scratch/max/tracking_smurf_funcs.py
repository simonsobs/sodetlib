#bin/sh
import matplotlib
matplotlib.use('Agg')

import pysmurf.client
import argparse
import numpy as np
import time
import sys


def tracking_resonator(S, band, reset_rate_khz, init_fraction_full_scale,
                        phiO_number, lms_gain = None):
    """
    Measure least mean square freq (lms_freq) with all channels currently tuned.
    Optimize for maximum number of integer phi0 and correspondingly
    frac_start. Return how many channels have been turned off and tracking
    arguments. Retune + rerun tracking setup not in measure mode with
    optimized parameters

    Returns: Optimized tracking parameters, number of channels on/off

    Args:
        S:
            Pysmurf control object
        band:
            500MHz band to run on
        reset_rate_khz:
            Width of the flux ramp in kHz, T = 1/reset_rate_khz, we use to start
            at 4 khz
        init_fraction_full_scale
            Fraction full scale is amplitude of the flux ramp, give an initial
            value then estimate the optimize value base on number of phi_0.
            The init fraction full scale value is [0,1].
        phiO_number
            Number of flux quantum in the SQUID. It is usually around 4 - 6
            Phi_0. If you set your flux ramp amplitude = 5 Phi_0, it means after
            5 Phi_0, you will reset the flux ramp amplitude, then the least mean
            frequency = 5*reset_rate_khz = 20 kHz
        lms_gain:
            Tracking frequency lock loop feedback loop gain.
    """
    frac_pp_optimize = False:
    while not(frac_pp_optimize):
        # run tracking_setup one time

        f,df,sync = S.tracking_setup(band=band,reset_rate_khz=reset_rate_khz,
                        lms_freq_hz=None, nsamp = 2**17,
                        fraction_full_scale=init_fraction_full_scale,
                        lms_gain = lms_gain, make_plot=False,show_plot=False,
                        save_plot=False, meas_lms_freq=True,channel=[],
                        return_data = True)

        df_std = np.std(df,0)
        f_span = np.max(f,0) - np.min(f,0)

        for c in S.which_on(band):
            if f_span < 0.03:
                S.channel_off(band,c)
            if f_span > 0.14:
                S.channel_off(band,c)

        f,df,sync = S.tracking_setup(band=band,reset_rate_khz=reset_rate_khz,
                        lms_freq_hz=None, nsamp = 2**17,
                        fraction_full_scale=init_fraction_full_scale,
                        lms_gain = lms_gain, make_plot=False,show_plot=False,
                        save_plot=False, meas_lms_freq=True,channel=[],
                        return_data = False)

        lms_meas = S.lms_freq_hz[band]
        frac_pp = init_fraction_full_scale*(reset_rate_khz*1e3*phiO_number/lms_meas)

        if (frac_pp >=0.99):
            print(f'Fraction full scale of {phiO_number} Phi0 <0.99 fraction_full_scale')
            print(f'Rerunning optimization for {phi0_number - 1} Phi0 ')
            phi0_number -= 1
        if (frac_pp < 0.99):
            print(f'Fraction full scale of {phiO_number} Phi0',
            f'= {np.round(frac_pp*100,3}% full scale')
            frac_pp_optimize = True

    S.relock(band)
    for i in range(2):
        S.run_serial_gradient_descent(band)
        S.run_serial_eta_scan(band)

    lms_freq_opt = phi0_number*reset_rate_khz*1e3
    # run tracking_setup firstly, then loop several time
    f,df,sync = S.tracking_setup(band=band,reset_rate_khz=reset_rate_khz,
                    lms_freq_hz=lms_freq_opt, lms_gain = lms_gain,
                    fraction_full_scale=frac_pp, nsamp = 2**17,
                    make_plot=True,show_plot=False,save_plot=True,
                    meas_lms_freq=False,channel=S.which_on(band),
                    return_data = True)

    #We use the amplitude of the frequency swing and the standard deviation of
    #the tracking error to turn off bad channels.
    #To check the channels

    df_std = np.std(df,0)
    f_span = np.max(f,0) - np.min(f,0)

    bad_track_chans = []
    for c in S.which_on(band):
        if f_span < 0.03:
            bad_track_chans.append(c)
            print('Low df_pp Cut')
            S.channel_off(band,c)
        if f_span > 0.14:
            bad_track_chans.append(c)
            print('High df_pp Cut')
            S.channel_off(band,c)
    return lms_freq_opt, frac_pp, bad_track_chans
