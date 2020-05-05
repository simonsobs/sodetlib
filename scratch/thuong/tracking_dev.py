#bin/sh
import matplotlib
matplotlib.use('Agg')

import pysmurf.client
import argparse
import numpy as np
import time
import sys


def tracking_resonator(S, band, reset_rate_khz, init_fraction_full_scale, phiO_number, optimize_number=None):
    """
    Measure least mean square freq (lms_freq) with all channels on 
    Run check_lock to kill poorly tracking channels or use tracking error to make own
    Optimize for maximum number of integer phi0 and correspondingly frac_start
    Return how many channels have been turned off and tracking arguments 
    Retune + rerun tracking setup not in measure mode with optimized parameters 
    Returns: Optimized tracking parameters, number of channels on/off
    Args:
        S: 
            Pysmurf control object
        reset_rate_khz:
            Width of the flux ramp in kHz, T = 1/reset_rate_khz, we use to start at 4 khz
        init_fraction_full_scale
            Fraction full scale is amplitude of the flux ramp, give an initial value then estimate the optimize value base on number of phi_0.
            The init fraction full scale value is [-1,1].
        phiO_number
            Number of flux quantum in the SQUID. It is usually around 4 - 6 Phi_0. If you set your flux ramp amplitude = 5 Phi_0, it means after 5 Phi_0, you will reset the flux ramp amplitude, then the least mean frequency = 5*reset_rate_khz = 20 kHz
        optimize_number
            The number of running tracking to optimize the fraction full scale
    """

    if optimize_number is None:
        
        S.tracking_setup(band=band,reset_rate_khz=reset_rate_khz,lms_freq_hz=None,fraction_full_scale=init_fraction_full_scale,make_plot=True,show_plot=False,save_plot=True,meas_lms_freq=True,channel=S.which_on(band))

        #lms_meas = np.max(S.lms_freq_hz)
        lms_meas = S.lms_freq_hz[band]
        frac_pp = init_fraction_full_scale*(reset_rate_khz*phiO_number/lms_meas)

        print('Fraction full sclae of '+str(phiO_number)+'Phi0 = ',frac_pp)

        if (frac_pp >=0.99) or (frac_pp <= -0.99):
            raise Exception('Change the phi0_number or initial fraction full scale to have fraction full scale [-1,1]')

    else:
        S.tracking_setup(band=band,reset_rate_khz=reset_rate_khz,lms_freq_hz=None,fraction_full_scale=init_fraction_full_scale,make_plot=True,show_plot=False,save_plot=True,meas_lms_freq=True,channel=S.which_on(band))
        lms_meas = np.max(S.lms_freq_hz)
        frac_pp = init_fraction_full_scale*(reset_rate_khz*phiO_number/lms_meas)

        for i in range(np.int(optimize_number)):
            S.tracking_setup(band=band,reset_rate_khz=reset_rate_khz,lms_freq_hz=reset_rate_khz*phiO_number,fraction_full_scale=frac_pp,make_plot=True,show_plot=False,save_plot=True,meas_lms_freq=True,channel=S.which_on(band))
            #lms_meas = np.max(S.lms_freq_hz)
            lms_meas = S.lms_freq_hz[band]
            frac_pp = init_fraction_full_scale*(reset_rate_khz*phiO_number/lms_meas)
            print('Fraction full scale of the '+str(i)+' optimize:',frac_pp)

        print('Fraction full sclae of '+str(phiO_number)+'Phi0 = ',frac_pp)

        if (frac_pp >=0.99) or (frac_pp <= -0.99):
            raise Exception('Change the phi0_number or initial fraction full scale to have fraction full scale [-1,1]')

    #We use the amplitude of the frequency swing and the standard deviation of the tracking error to turn off bad channels. 
    #To check the channels
    channels_off = S.check_lock(band) # return channels off
    channels_on = S.which_on(band)    # all of the channels on in a band at any given time
    S.tracking_setup(band=band,reset_rate_khz=reset_rate_khz,lms_freq_hz=reset_rate_khz*phiO_number,fraction_full_scale=frac_pp,make_plot=True,show_plot=False,save_plot=True,meas_lms_freq=True,channel=S.which_on(band))
    
    return frac_pp,channels_on, channels_off


if __name__=='__main__':    
    parser = argparse.ArgumentParser()

    # Arguments that are needed to create Pysmurf object
    parser.add_argument('--setup', action='store_true')
    parser.add_argument('--config-file', required=True)
    parser.add_argument('--epics-root', default='smurf_server_s2')

    # Custom arguments for this script
    parser.add_argument('--band', type=int, required=True, 
                        help='band (must be in range [0,3])')

    parser.add_argument('--reset_rate_khz', type=float, default=4, 
        help="Reset rate of flux ramp in khz")

    parser.add_argument('--init_fraction_full_scale', type=float, default=.7,
        help="Fraction amplitude of the flux ramp peak-peak"
    )
    parser.add_argument('--phiO_number', type=int, default=5,
        help="Number of Phi_0 per flux ramp"
    )

        parser.add_argument('--optimize_number', type=int, default=2,
        help="optimized fraction full scale parameter"
    )

    # Parse command line arguments
    args = parser.parse_args()

    S = pysmurf.client.SmurfControl(
            epics_root = args.epics_root,
            cfg_file = args.config_file,
            setup = args.setup, make_logfile=False,
    )
                                                    
    tracking_resonator(S, args.band, args.reset_rate_khz, args.init_fraction_full_scale, args.phiO_number,args.optimize_number)


