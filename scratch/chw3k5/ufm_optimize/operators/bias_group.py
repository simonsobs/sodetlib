import os
import sys
import time
import argparse

# Here we append this python file's directory to the paths the python uses to look for imports.
# This is a temporary measure used as demonstration and testing tool.
basedir_this_file = os.path.basename(__file__)
sys.path.append(basedir_this_file)

from controler import LoadS
from smurf_band import SingleBand
from time_stream import TimeStreamData


class GroupedBiases:
    def __init__(self, S, cfg, band_group, verbose=True):
        """
        :param S: SMuRF controller object.
        :param cfg: SMuRF configuration object
        :param band_group: set - Needs to be some iterable that can be turned into a set of bands (bias lines).
                                 All the bands in this set will be operated on.
        :param verbose: bool - default is True: When True, toggles print() statements for the
                               various actions taken by the methods in this class. False,
                               runs silently.
        """
        self.S = S
        self.cfg = cfg
        self.verbose = verbose
        self.band_group = set(band_group)
        self.band_group_list = sorted(self.band_group)

    def overbias_tes(self, sleep_time=120, tes_bias=1):
        if self.verbose:
            print("Overbiasing TES")
        self.S.overbias_tes_all(band_groups=self.band_group_list,
                                overbias_wait=1, tes_bias=tes_bias, cool_wait=3, high_current_mode=False,
                                overbias_voltage=12)
        if self.verbose:
            print("  waiting for thermal environment get stabilized, sleeping {sleep_time} seconds.")
        time.sleep(sleep_time)


class AutoTune:
    """
    A class to tune parameters independently across bias lines, checking the white noise levels
    and making mappings with parameter values. These mappings are used to inform subsequent
    tunings.

    This class makes heavy use of the TimeStreamData class. An instance of that class is
    started in the __init__() method of this class, see self.time_stream_data.

    This is a bit experimental. There are many operational topologies for optimizing detectors,
    we will choose one that accounts for operational constraints of real devices.
    """
    def __init__(self, S, cfg, band_group, nperseg=2**12, verbose=True):
        """
        :param S: SMuRF controller object.
        :param cfg: SMuRF configuration object
        :param band_group: set - Needs to be some iterable that can be turned into a set of bands (bias lines).
                         All the bands in this set will be operated on.
        :param nperseg: nperseg int - default is 2**18 – The number of samples used in the
                                      PSD estimator. See scipy.signal.welch.
        :param verbose: bool - default is True: When True, toggles print() statements for the
                               various actions taken by the methods in this class. False,
                               runs silently.
        """
        # user defined variables and __init__ declared variables.
        self.S = S
        self.cfg = cfg
        self.verbose = verbose
        self.band_group = set(band_group)
        self.band_group_list = sorted(self.band_group)
        self.nperseg = nperseg

        # in method tuner()
        self.wl_medians = None
        self.noise_floors = None
        self.uc_atten_wl_median_tune_map_per_band = {}
        self.uc_atten_best_per_band = None
        self.lowest_wl_index_per_band = None
        self.wl_median_per_band = None

        self.time_stream_data = TimeStreamData(S=S, cfg=cfg, nperseg=self.nperseg, verbose=verbose)

    def single_time_stream(self, stream_time=120, do_plot=False, fmin=5, fmax=50):
        """
        Get a single time stream and use the TimeStreamData class to do some additional processing.

        :param stream_time: float - The default is 120 seconds. The sleep time in seconds to wait
                                    while acquiring time stream data.
        :param do_plot: bool - default is False. When True, a standard diagnostic plot is rendered
                               and saved for this time stream.
        :param fmin: float - default is float('-inf'). The minimum frequency to consider,
                             frequencies below this value will be masked during calculations.
        :param fmax: float - default is float('inf'). The maximum frequency to consider,
                             frequencies above this value will be masked during calculations.
        """
        self.time_stream_data.take_ts(stream_time=stream_time)
        if do_plot:
            self.time_stream_data.plot_ts()
        self.time_stream_data.get_median_bias_wl(fmin=fmin, fmax=fmax)
        if self.verbose:
            print(f'wl_median {self.time_stream_data.wl_median}')

    def tuner_up_atten(self, uc_attens_centers_per_band, tune_type_per_band=None, stream_time=120,
                       do_plots=False, fmin=5, fmax=50):
        """
        Acquire multiple time streams to test up conversation attenuation settings values for all bias lines.

        : param uc_attens_centers_per_band: dict - required. Expecting a dictionary with keys that
                                                   are bands (bias lines). The values for each key
                                                   should be up conversation attenuation setting
                                                   values, int. For bands where
                                                   tune_type_per_band[band] == 'single' then
                                                   uc_attens_centers_per_band[band] is the only
                                                   value measured this method. If
                                                   tune_type_per_band[band] in {'fine', 'rough'},
                                                   then the value of uc_attens_centers_per_band[band]
                                                   is the center value of a spread of a searches
                                                   over several time streams.
        :param tune_type_per_band: dict - default is None. Expecting a dictionary with keys that
                                          are bands (bias lines) and at least the all the keys
                                          also in uc_attens_centers_per_band.keys(). The values
                                          for each key should be the strings 'single' or 'fine',
                                          all other values trigger a 'rough' tuning. A single tuning
                                          does only one tuning a the up conversation attenuation
                                          at the value of uc_attens_centers_per_band[band].
                                          'fine' does a tuning of near by uc_atten values, while
                                          'rough' or the default tuning is using uc_atten values
                                          with a larger spread. None, the default, will set
                                          tune_type_per_band[band] = 'single' for every band in
                                          uc_attens_centers_per_band.keys()
        :param stream_time: float - The default is 120 seconds. Applies to all tunings started in
                                    this method.The sleep time in seconds to wait while acquiring
                                    time stream data.
        :param do_plots: bool - default is False. Applies to all tunings started in this method.
                                When True, a standard diagnostic plot is rendered and saved for each
                                time stream in this method.
        :param fmin: float - default is float('-inf'). Applies to all tunings started in this method.
                             The minimum frequency to consider, frequencies below this value will be
                             masked during calculations.
        :param fmax: float - default is float('inf'). Applies to all tunings started in this method.
                             The maximum frequency to consider, frequencies above this value will
                             be masked during calculations.
        """
        if tune_type_per_band is None:
            tune_type_per_band = {band: 'single' for band in uc_attens_centers_per_band.keys()}
        # determine the settings to test
        uc_attens_per_band = {}
        for band in uc_attens_centers_per_band:
            uc_attens_per_band[band] = []
            uc_attens_this_band = uc_attens_per_band[band]
            uc_atten_center_this_band = uc_attens_centers_per_band[band]
            uc_atten_wl_median_tune_map_this_band = self.uc_atten_wl_median_tune_map_per_band[band]
            tune_type_this_band = tune_type_per_band[band]
            tune_type_this_band = tune_type_this_band.lower()
            if tune_type_this_band is 'fine':
                diffs_from_center = [-2, -1, 0, 1, 2]
            elif tune_type_this_band is 'single':
                diffs_from_center = [0]
            else:
                diffs_from_center = [-10, -5, 0, 5, 10]

            for diff_from_center in diffs_from_center:
                test_setting = uc_atten_center_this_band + diff_from_center
                # try to use settings that have not yet been tested on.
                if tune_type_this_band != 'fine' and \
                        test_setting in uc_atten_wl_median_tune_map_this_band.keys():
                    # add or subtract a value to not repeat a measurements.
                    if diff_from_center < 0:
                        small_perturbations = [-1, 1, -2, 2]
                    else:
                        small_perturbations = [1, -1, 2, -2]
                    for small_perturbation in small_perturbations:
                        test_setting2 = test_setting + small_perturbation
                        if test_setting2 not in uc_atten_wl_median_tune_map_this_band.keys():
                            test_setting = test_setting2
                            break
                uc_attens_this_band.append(test_setting)
        # test for results
        bands = sorted(uc_attens_per_band.keys())
        bands_remaining = set(bands)
        self.wl_medians = {band: [] for band in bands}
        self.noise_floors = {band: [] for band in bands}
        index_counter = 0
        while bands_remaining != set():
            for band in sorted(bands_remaining):
                try:
                    uc_atten = uc_attens_per_band[band][index_counter]
                except IndexError:
                    bands_remaining.remove(band)
                else:
                    self.S.set_att_uc(band, uc_atten)
                    self.S.tracking_setup(band,
                                          reset_rate_khz=self.cfg.dev.bands[band]["flux_ramp_rate_khz"],
                                          fraction_full_scale=self.cfg.dev.bands[band]["frac_pp"],
                                          make_plot=False,
                                          save_plot=False,
                                          show_plot=False,
                                          channel=self.S.which_on(band),
                                          nsamp=2 ** 18,
                                          lms_freq_hz=None,
                                          meas_lms_freq=True,
                                          feedback_start_frac=self.cfg.dev.bands[band]["feedback_start_frac"],
                                          feedback_end_frac=self.cfg.dev.bands[band]["feedback_end_frac"],
                                          lms_gain=self.cfg.dev.bands[band]["lms_gain"])

            index_counter += 1
            # record the data
            self.single_time_stream(stream_time=stream_time, do_plot=do_plots, fmin=fmin, fmax=fmax)
            # collect the data after the time streams
            for band in bands:
                self.wl_medians[band].append(self.time_stream_data.wl_median[band])
                self.noise_floors[band].append(self.time_stream_data.wl_list[band])
        # analyze the results and set the optimal values.
        for band in bands:
            wl_medians_this_band = self.wl_medians[band]
            uc_attens__this_band = uc_attens_per_band[band]
            # print results once per band
            lowest_wl_index = wl_medians_this_band.index(min(wl_medians_this_band))
            wl_median = self.wl_medians[band][lowest_wl_index]
            uc_atten_best_this_band = uc_attens__this_band[lowest_wl_index]
            self.uc_atten_best_per_band[band] = uc_atten_best_this_band
            self.lowest_wl_index_per_band[band] = lowest_wl_index
            self.wl_median_per_band[band] = wl_median
            if self.verbose:
                print(f'T est results for "up conversion attenuation" band:{band} tune_type:{tune_type_per_band[band]}')
                print(f'  medians per up-conversation attenuation index:\n  {self.wl_medians[band]}')
                print(f'  {uc_atten_best_this_band} up-conversation attenuation index with the lowest median ' +
                      f'channel noise {wl_median}')
                channel_length = self.time_stream_data.stream_by_band_by_channel[band]
                print(f"  lowest WL: {wl_median} with {channel_length} channels")
            # set this tuning
            self.S.set_att_uc(band, uc_atten_best_this_band)
            # record the up conversation attenuation with noise level map
            self.uc_atten_wl_median_tune_map_per_band[band] = {uc_atten: wl_median for uc_atten, wl_median
                                                               in zip(uc_attens__this_band, wl_medians_this_band)}

    def tune_selector_up_atten(self, uc_attens_centers_per_band=None, loop_count_max=5,
                               stream_time=120, do_plots=False, fmin=5, fmax=50):
        """
        :param uc_attens_centers_per_band: dict - default is None. Expecting a dictionary with keys that
                                                  are bands (bias lines). The values for each key
                                                  should be up conversation attenuation setting
                                                  values, int. This sets the intial test value of
                                                  uc_atten for each band. None set all bands to
                                                  a uc_atten values that is a constant. Ideally,
                                                  this would read ina record best of value.
        :param loop_count_max: int - default is 5. The number of loops to attempt tuning before
                                     a failed loop exit.
        :param stream_time: float - The default is 120 seconds. Applies to all tunings started in
                                    this method.The sleep time in seconds to wait while acquiring
                                    time stream data.
        :param do_plots: bool - default is False. Applies to all tunings started in this method.
                                When True, a standard diagnostic plot is rendered and saved for each
                                time stream in this method.
        :param fmin: float - default is float('-inf'). Applies to all tunings started in this method.
                             The minimum frequency to consider, frequencies below this value will be
                             masked during calculations.
        :param fmax: float - default is float('inf'). Applies to all tunings started in this method.
                             The maximum frequency to consider, frequencies above this value will
                             be masked during calculations.
        """
        # do an initial of the up conversation attenuators
        if uc_attens_centers_per_band is None:
            uc_attens_centers_per_band = {band: 12 for band in self.band_group}
        self.tuner_up_atten(uc_attens_centers_per_band=uc_attens_centers_per_band, tune_type_per_band=None,
                            stream_time=stream_time, do_plots=do_plots, fmin=fmin, fmax=fmax)

        # check the tuning and do some optimization
        bias_bands_to_tune = set(uc_attens_centers_per_band.keys())
        loop_count = 0
        while bias_bands_to_tune == set():
            # break from the loop to indicate a non ideal exit
            if loop_count < loop_count_max:
                print(f'loop count: {loop_count}, is at it maximum allowed value.')
                break
            loop_count += 1
            # # the tuning selection process
            # determine the best tuning in the global map
            uc_attens_centers_per_band = {}
            tune_type_per_band = {}
            for band in sorted(bias_bands_to_tune):
                uc_atten_wl_median_this_band = self.uc_atten_wl_median_tune_map_per_band[band]
                # return the key (up conversation attenuation value) of the lowest value in an array
                uc_atten_min_wl_median = min(uc_atten_wl_median_this_band, key=uc_atten_wl_median_this_band.get)
                wl_median_this_band = self.uc_atten_wl_median_tune_map_per_band[band][uc_atten_min_wl_median]
                uc_attens_centers_per_band[band] = uc_atten_min_wl_median
                # rough and fine tuning need to be done per band.
                print(f"WL: {wl_median_this_band} for band {band}")
                if 150 < wl_median_this_band:
                    raise ValueError(f"WL: {wl_median_this_band} is greater then 150. Something might be wrong,\n" +
                                     f"Up and/or Down conversation attenuation may not be correctly set\n" +
                                     f"Please investigate")
                elif 60 < wl_median_this_band <= 150:
                    print(f"  Attempting rough tuning for band:{band}")
                    tune_type_per_band[band] = 'rough'
                elif wl_median_this_band < 60:
                    print(f"  Attempting fine tuning for band:{band}")
                    tune_type_per_band[band] = 'fine'
                # apply the tuning
                self.tuner_up_atten(uc_attens_centers_per_band=uc_attens_centers_per_band,
                                    tune_type_per_band=tune_type_per_band,
                                    stream_time=stream_time, do_plots=do_plots, fmin=fmin, fmax=fmax)


if __name__ == '__main__':
    # set up the parser for this Script
    parser = argparse.ArgumentParser(description='Parser for time_stream.py script.')
    parser.add_argument('slot_num', type=int, metavar='slot_num', nargs=1,
                        help='The SMuRF slot number (int) used for identification and control.')
    parser.add_argument('band', type=int, metavar='band', nargs=1,
                        help='The SMuRF band (int) for setup.')
    parser.add_argument('band_group', type=int, metavar='bands', nargs='+', action='append',
                        help='The SMuRF bands (ints) to optimize. This is expected to be a sequence of N integers.')
    # optional arguments
    parser.add_argument('--verbose', dest='verbose', action='store_true',
                        help="Turns on printed output from the script. The default is --verbose." +
                             "--no-verbose has minimal (or no) print statements.")
    parser.add_argument('--no-verbose', dest='verbose', action='store_false', default=True,
                        help="Turns off printed output from the script. The default is --verbose." +
                             "--no-verbose has minimal (or no) print statements.")
    parser.add_argument('--nperseg', dest='nperseg', type=int, default=2 ** 18,
                        help="int, optional, default is 2**18. The number of samples used in the PSD estimator. " +
                             "See scipy.signal.welch.")
    parser.add_argument('--streamtime', dest='stream_time', type=float, default=20.0,
                        help="float, optional, default is 20.0. The amount of time to stream data in seconds during " +
                             "tuning checks.")
    parser.add_argument('--loop-countmax', dest='loop_count_max', type=int, default=5,
                        help='int, optional, default is 5. The maximum number of times turning time streams to take ' +
                             'while trying to find acceptable performance for up-conversation attenuation.')
    parser.add_argument('--overbias-sleeptime', dest='overbias_sleep_time', type=float, default=120.0,
                        help="float, optional, default is 120.0. The amount of time to sleep in seconds" +
                             "after overbiasing the detectors.")
    parser.add_argument('--overbias-tesbias', dest='overbias_tes_bias', type=float, default=1.0,
                        help="float, optional, default is 1.0.")
    parser.add_argument('--fmin', dest='fmin', type=float, default=5.0,
                        help="float, optional, default is 5.0.")
    parser.add_argument('--fmax', dest='fmax', type=float, default=50.0,
                        help="float, optional, default is 50.0.")
    # parse the args for this script.
    args = parser.parse_args()


    # load a single S, or SMuRF controller instance for a given slot number
    load_s = LoadS(slot_nums=[args.slot_num], verbose=args.verbose)
    cfg = load_s.cfg_dict[args.slot_num]
    S = load_s.S_dict[args.slot_num]

    # configure a single band
    single_band = SingleBand(S=S, cfg=cfg, band=args.band, auto_startup=True, verbose=args.verbose)
    single_band.check_lock()

    # configure a collection of bands as a single bias group.
    grouped_biases = GroupedBiases(S=S, cfg=cfg, band_group=args.band_group, verbose=args.verbose)
    grouped_biases.overbias_tes(sleep_time=args.overbias_sleep_time, tes_bias=args.overbias_tes_bias)

    # acquire time stream data
    auto_tune = AutoTune(S=S, cfg=cfg, nperseg=args.nperseg, band_group=args.band_group, verbose=args.verbose)
    auto_tune.tune_selector_up_atten(uc_attens_centers_per_band=None, loop_count_max=args.loop_count_max,
                                     stream_time=args.stream_time, do_plots=False, fmin=args.fmin, fmax=args.fmax)
    # print the plotting directory
    if args.verbose:
        print(f"plotting directory is:\n{S.plot_dir}")
