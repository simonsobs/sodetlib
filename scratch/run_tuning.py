import matplotlib
matplotlib.use('Agg') # I put it in backend
import matplotlib.pyplot as plt

import pysmurf.client
import argparse
import numpy as np
import pickle as pkl
from scipy import signal
import os


def find_and_tune_freq(S, bands, subband=np.arange(13,115), drive_power=None,
            n_read=2, make_plot=False, save_plot=True, plotname_append='',
            window=50, rolling_med=True, make_subband_plot=False,
            show_plot=False, grad_cut=.05, amp_cut=.25, pad=2, min_gap=2, # find_freq
            resonance=None, sweep_width=.3, df_sweep=.002, min_offset=0.1,
            delta_freq=None, new_master_assignment=False,
            lock_max_derivative=False, # setup_notches
            sync_group=True, timeout=240): # run_serial_gradient_descent & eta_scan
    """
    Find_freqs to identify resonance, measure eta parameters + setup channels
    using setup_notches, run serial gradient + eta to refine

    Parameters
    ----------
    S : pysmurf instance
        pysmurf instance to work with
    bands : [int]
        bands to find tuned frequencies on. In range [0,7].

    Optional parameters
    ----------
    subband : [int]
        An int array for the subbands
    drive_power : int
        The drive amplitude.  If none given, takes from cfg.
    n_read : int
        The number sweeps to do per subband
    make_plot : bool
        make the plot frequency sweep. Default False.
    save_plot : bool
        save the plot. Default True.
    plotname_append : str
        Appended to the default plot filename. Default ''.
    rolling_med : bool
        Whether to iterate on a rolling median or just the median of the whole
        sample.
    window : int
        The width of the rolling median window
    make_subband_plot : bool
        Make subband plots? Default False.
    show_plot : bool
        Show plots as they are made? Default False.
    pad : int
        number of samples to pad on either side of a resonance search
        window
    min_gap : int
        minimum number of samples between resonances
    grad_cut : float
        The value of the gradient of phase to look for
        resonances. Default is .05
    amp_cut : float
        The fractional distance from the median value to decide
        whether there is a resonance. Default is .25.
    resonance : [float]
        A 2 dimensional array with resonance
        frequencies and the subband they are in. If given, this will take
        precedent over the one in self.freq_resp.
    sweep_width : float
        The range to scan around the input resonance in
        units of MHz. Default .3
    df_sweep : float
        The sweep step size in MHz. Default .005
    min_offset : float
        Minimum distance in MHz between two resonators for assigning channels.
    delta_freq : float
        The frequency offset at which to measure
        the complex transmission to compute the eta parameters.
        Passed to eta_estimator.  Units are MHz.  If none supplied
        as an argument, takes value in config file.
    new_master_assignment : bool
        Whether to create a new master assignment
        file. This file defines the mapping between resonator frequency
        and channel number.
    lock_max_derivative : bool
        I'm not sure what this is, but it was in setup_notches when I wrote the
        OCS script (though with no documentation.) Defaults to False.
    sync_group : bool
        Whether to use the sync group to monitor
        the PV. Default is True.
    timeout : float
        The maximum amount of time to wait for the PV.

    Returns
    -------
    tune_file : str
        Name of the resultant tune file, currently loaded.
    resonators_on : int
        Number of channels (resonators) in this band that are on.
    """

    num_resonators_on = 0
    band_tune_file_dict = {}
    for band in bands:
        S.find_freq(band,subband=subband, drive_power=drive_power,
                    n_read=n_read, make_plot=make_plot, save_plot=save_plot,
                    plotname_append=plotname_append, window=window,
                    rolling_med=rolling_med, make_subband_plot=make_subband_plot,
                    show_plot=show_plot, grad_cut=grad_cut, amp_cut=amp_cut, pad=pad,
                    min_gap=min_gap)
        S.setup_notches(band, drive=drive_power, resonance=resonance,
                sweep_width=sweep_width, df_sweep=df_sweep, min_offset=min_offset,
                delta_freq=resonance, new_master_assignment=new_master_assignment,
                lock_max_derivative=new_master_assignment)
        #set up tracking
        S.run_serial_gradient_descent(band, sync_group=sync_group, timeout=timeout)
        S.run_serial_eta_scan(band, sync_group=sync_group, timeout=timeout)
        band_tune_file_dict[band]= S.tune_file
        num_resonators_on += S.which_on(band)
    # testing
    print("Total num resonators on: " + str(num_resonators_on))
    print("Dictionary of band tune file names: " +str(band_tune_file_dict))
    return (num_resonators_on, band_tune_file_dict)


if __name__=='__main__':
    parser = argparse.ArgumentParser()

    # Arguments that are needed to create Pysmurf object
    parser.add_argument('--setup', action='store_true')
    parser.add_argument('--config-file', required=True)
    parser.add_argument('--epics-root', default='smurf_server_s2')

    # Custom arguments for this script
    # it does ALL bands, thus doesn't need this one.
    parser.add_argument('--bands', nargs='+', type=int, required=True,
        help='input bands to tune as ints, separated by spaces (must be in range [0,7])')
    # find_freq optional arguments
    parser.add_argument('--subband', nargs='+', type=int, default=np.arange(13,115),
        help='An int array for the subbands. Type integers separated by spaces to input.')
    parser.add_argument('--drive-power', type=int, default=None,
        help='The drive amplitude. If none given, takes from cfg.')
    parser.add_argument('--n-read', type=int, default=2,
        help='The number sweeps to do per subband')
    parser.add_argument('--make-plot', type=bool, default=False,
        help='make the plot frequency sweep. Default False.')
    parser.add_argument('--save-plot', type=bool, default=True,
        help='save the plot. Default True.')
    parser.add_argument('--plotname-append', type=str, default='',
        help="Appended to the default plot filename. Default is ''.")
    parser.add_argument('--rolling-med', type=bool, default=True, # causing trouble for some reason
        help='Whether to iterate on a rolling median or just the median of the whole sample.')
    parser.add_argument('--window', type=int, default=50,
        help='The width of the rolling median window')
    parser.add_argument("--make-subband-plot", type=bool, default=False,
        help="Make subband plots? Default False.")
    parser.add_argument("--show-plot", type=bool, default=False,
        help="Show plots as they are made? Default False.")
    parser.add_argument('--grad-cut', type=float, default=.05,
        help='The value of the gradient of phase to look for resonances. Default is .05')
    parser.add_argument('--amp-cut', type=float, default=.25,
        help='The fractional distance from the median value to decide whether there is a resonance. Default is .25.')
    parser.add_argument("--pad", type=int, default=2,
        help="number of samples to pad on either side of a resonance search window")
    parser.add_argument("--min-gap", type=int, default=2,
        help="minimum number of samples between resonances")
    # setup_notches optional arguments
    parser.add_argument('--resonance', nargs='+', type=float, default=None,
        help='A 2 dimensional array with resonance frequencies and the subband they are in. If given, this will take precedent over the one in self.freq_resp. Type floats separated by spaces to input.')
    parser.add_argument('--sweep-width', type=float, default=.3,
        help='The range to scan around the input resonance in units of MHz. Default .3')
    parser.add_argument('--df-sweep', type=float, default=.002,
        help='The sweep step size in MHz. Default .005')
    parser.add_argument('--min-offset', type=float, default=0.1,
        help='Minimum distance in MHz between two resonators for assigning channels.')
    parser.add_argument("--delta-freq", type=float, default=None,
        help="The frequency offset at which to measure the complex transmission to compute the eta parameters. Passed to eta_estimator. Units are MHz. If none supplied as an argument, takes value in config file.")
    parser.add_argument("--lock-max-derivative", type=bool, default=False,
        help="I'm not sure what this is, but it was in setup_notches when I wrote the OCS script (though with no documentation.) Defaults to  False.")
    parser.add_argument('--new-master-assignment', type=bool, default=False,
        help='Whether to create a new master assignment file. This file defines the mapping between resonator frequency and channel number.')
    # run_serial_gradient_descent & run_serial_eta_scan optional args
    parser.add_argument('--sync-group', type=bool, default=True,
        help='Whether to use the sync group to monitor the PV. Defauult is True.')
    parser.add_argument('--timeout', type=float, default=240,
        help='The maximum amount of time to wait for the PV.')

    # Parse command line arguments
    args = parser.parse_args()

    # OFFLINE VERSION!
    # S = pysmurf.client.SmurfControl(offline=True)
    # Online version!
    S = pysmurf.client.SmurfControl(
        epics_root=args.epics_root,
        cfg_file=args.config_file,
        setup=args.setup, make_logfile=False,
    )

    # Put your script calls here
    # I still want to know why the for loop over bands HAS to be inside func
    find_and_tune_freq(
        S, args.bands, subband=args.subband,
        drive_power=args.drive_power, n_read=args.n_read,
        make_plot=args.make_plot, save_plot=args.save_plot,
        plotname_append=args.plotname_append, window=args.window,
        rolling_med=args.rolling_med,
        make_subband_plot=args.make_subband_plot,
        show_plot=args.show_plot, grad_cut=args.grad_cut,
        amp_cut=args.amp_cut, pad=args.pad,
        min_gap=args.min_gap,  # find_freq
        resonance=args.resonance, sweep_width=args.sweep_width,
        df_sweep=args.df_sweep, min_offset=args.min_offset,
        delta_freq=args.delta_freq,
        new_master_assignment=args.new_master_assignment,
        lock_max_derivative=args.lock_max_derivative,  # setup_notches
        sync_group=args.sync_group, timeout=args.timeout)
