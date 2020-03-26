import matplotlib
matplotlib.use('Agg') # I put it in backend
import matplotlib.pyplot as plt

import pysmurf.client
import argparse
import numpy as np
import pickle as pkl
from scipy import signal
import os


def find_and_tune_freq(S, band, subband=np.arange(13,115), drive_power=None,
            n_read=2, make_plot=False, save_plot=True, plotname_append='',
            window=50, rolling_med=True, make_subband_plot=False,
            show_plot=False, grad_cut=.05, amp_cut=.25, pad=2, min_gap=2
            ):
    """
    Find_freqs to identify resonance, measure eta parameters + setup channels
    using setup_notches, run serial gradient + eta to refine

    Parameters
    ----------
    band : int
        band to find tuned frequencies on

    Optional parameters
    ----------
    subband : int
        An int array for the subbands
    drive_power : int
        The drive amplitude.  If none given, takes from cfg.
    n_read : int
        The number sweeps to do per subband
    make_plot : bool
        make the plot frequency sweep. Default False.
    save_plot : bool
        save the plot. Default True.
    plotname_append : string
        Appended to the default plot filename. Default ''.
    rolling_med : bool
        Whether to iterate on a rolling median or just the median of the whole
        sample.
    window : int
        The width of the rolling median window
    pad : int): number of samples to pad on either side of a resonance search
        window
    min_gap : int): minimum number of samples between resonances
    grad_cut : float): The value of the gradient of phase to look for
        resonances. Default is .05
    amp_cut : float): The fractional distance from the median value to decide
        whether there is a resonance. Default is .25.

    Returns
    -------
    found_min : str
        tells you if the minimum of the noise is within your range, yes if true, low if its at uc_atten = 30, and high if its at uc_atten = 0
    median_noise_min : float
        minimum median white noise level at the optimized uc_atten value
    atten_min : int
        attenuator value which minizes the median noise
    """

    print(band)

    S.find_freq(band,subband=subband, drive_power=drive_power,
                n_read=n_read, make_plot=make_plot, save_plot=aave_plot,
                plotname_append='',
                window=50, rolling_med=True, make_subband_plot=False,
                show_plot=False, grad_cut=.05, amp_cut=.25, pad=2, min_gap=2)'''
    S.setup_notches(band, drive=10, new_master_assignment=True)

    #set up tracking

    S.run_serial_gradient_descent(band);
    S.run_serial_eta_scan(band);
    S.tracking_setup(band,reset_rate_khz=4,fraction_full_scale=0.7, make_plot=True, save_plot=True, show_plot=False,channel=S.which_on(band), nsamp=2**18, lms_gain=6, lms_freq_hz=None, meas_lms_freq=True,feedback_start_frac=1/12,feedback_end_frac=1)

    #return found_min, np.min(median_noise), attens[med_min_arg]
    '''



if __name__=='__main__':
    parser = argparse.ArgumentParser()

    # Arguments that are needed to create Pysmurf object
    parser.add_argument('--setup', action='store_true')
    parser.add_argument('--config-file', required=True)
    parser.add_argument('--epics-root', default='smurf_server_s2')

    # Custom arguments for this script
    parser.add_argument('--band', type=int, required=True,
            help='band (must be in range [0,7])')
    """
    parser.add_argument('--tunefile', required=True,
        help='Tune file that you want to load for this analysis.')

    parser.add_argument('--dr', type=int, default = 12,
        help='Drive power at which to optimize the noise.')

    parser.add_argument('--frac-pp', type=float, required=True,
            help='Fraction full scale of your flux ramp used in tracking setup.')

    parser.add_argument('--lms-freq', type=float, required=True,
            help='Tracking frequency used in tracking setup.')
    """
    # Parse command line arguments
    args = parser.parse_args()

    # OFFLINE VERSION!
    S = pysmurf.client.SmurfControl(offline=True)
    '''
    S = pysmurf.client.SmurfControl(
            epics_root = args.epics_root,
            cfg_file = args.config_file,
            setup = args.setup, make_logfile=False,
    )'''

    #Put your script calls here
    #optimize_power_per_band(S,band = args.band,tunefile = args.tunefile,dr_start = args.dr,frac_pp = args.frac_pp,lms_freq = args.lms_freq)
    find_and_tune_freq(S, args.band)
	# TODO!!
    '''
    S.find_freq(band,drive_power=10,make_plot=True,subband=range(50,100))
    S.setup_notches(band, drive=10, new_master_assignment=True)

    #set up tracking

    S.run_serial_gradient_descent(band);
    S.run_serial_eta_scan(band);
    S.tracking_setup(band,reset_rate_khz=4,fraction_full_scale=0.7, make_plot=True, save_plot=True, show_plot=False,channel=S.which_on(band), nsamp=2**18, lms_gain=6, lms_freq_hz=None, meas_lms_freq=True,feedback_start_frac=1/12,feedback_end_frac=1)
    '''
