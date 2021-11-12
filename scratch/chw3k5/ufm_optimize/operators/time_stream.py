"""
Acquire, analyze, and visualize Time Stream data that is optimized for detector characterization.
"""

import os
import sys
import time
import argparse

import numpy as np
import scipy.signal as signal
from matplotlib import pyplot as plt

# Here we append this python file's directory to the paths the python uses to look for imports.
# This is a temporary measure used as demonstration and testing tool.
basedir_this_file = os.path.basename(__file__)
sys.path.append(basedir_this_file)

from controler import LoadS


class TimeStreamData:
    """
    View and acquire time steam data from a single SMuRF controller object. This class
    is designed to be an acquisition tool and quick look only. More advance analysis
    may be available elsewhere.

    This can be used to view existing time stream files. See __init__() and read_ts() methods.

    Note: to load and view multiple time streams simultaneously use multiple instances
    of this class. This class only holds one time stream of data at a time. Reading
    additional time stream data will overwrite previously loaded time stream data.

    """

    def __init__(self, S, cfg, nperseg=2 ** 18, data_path=None, verbose=True):
        """
        :param S: SMuRF controller object.
        :param cfg: SMuRF configuration object
        :param nperseg: nperseg int - default is 2**18 – The number of samples used in the
                                      PSD estimator. See scipy.signal.welch.
        :param data_path: path (str) - default is None. When None, auto-loading of data from the
                                       __init__() is disabled. When a path sting is given, the
                                       method is read_ts() is called and time stream file located
                                       at path is read in and analyzed.
        :param verbose: bool - default is True: When True, toggles print() statements for the
                               various actions taken by the methods in this class. False,
                               runs silently.
        """
        self.S = S
        self.cfg = cfg
        self.verbose = verbose

        self.nperseg = nperseg
        self.detrend = 'constant'

        self.data_path = data_path
        self.start_time = None
        self.fs = None
        self.bands, self.channels = None, None
        self.phase, self.mask, self.tes_bias = None, None, None
        self.sample_nums = None
        self.t_array = None
        self.stream_by_band_by_channel = None

        # in method get_median_bias_wl()
        self.wl_list = None
        self.wl_median = None

        if self.data_path is not None:
            self.read_ts()

    def take_ts(self, stream_time=20):
        """
        Aquire time stream data.

        :param stream_time: float - the sleep time in seconds to wait while acquiring time stream data.
        """
        self.start_time = self.S.get_timestamp()

        # non blocking statement to start time stream and return the dat filename
        try:
            self.data_path = self.S.stream_data_on()
            # collect stream data
            if self.verbose:
                print(f'Sleeping for {stream_time} seconds to take time stream data...\n')
            time.sleep(stream_time)
            # end the time stream
        except:
            self.S.stream_data_off()
            print(f'Stream Off command sent')
            raise
        else:
            self.S.stream_data_off()
            if self.verbose:
                print(f'Stream Off command sent')
        # read the data that was taken
        self.read_ts()

    def read_ts(self, data_path=None):
        """
        Collect Data and transform it to a shape for plotting and analysis

        :param data_path: path (str) - default is None. When None, the path from the __init__()
                                       method at self.data_path is used. When this is a path like
                                       str, the instance variable self.data_path is overwritten
                                       and the time stream data at data_path is loaded.
        """
        if data_path is not None:
            self.data_path = data_path
        # Read the data
        timestamp, self.phase, self.mask, self.tes_bias = self.S.read_stream_data(self.data_path, return_tes_bias=True)
        self.fs = self.S.get_sample_frequency()
        dirname, basename = os.path.split(self.data_path)
        file_handle, file_extention = basename.rsplit('.', 1)
        self.start_time = file_handle
        print(f'loaded the .dat file at: {self.data_path}')

        # hard coded variables
        self.bands, self.channels = np.where(self.mask != -1)
        self.phase *= self.S.pA_per_phi0 / (2.0 * np.pi)  # uA
        self.sample_nums = np.arange(len(self.phase[0]))
        self.t_array = self.sample_nums / self.fs

        # reorganize the data by band then channel
        self.stream_by_band_by_channel = {}
        for band, channel in zip(self.bands, self.channels):
            if band not in self.stream_by_band_by_channel.keys():
                self.stream_by_band_by_channel[band] = {}
            ch_idx = self.mask[band, channel]
            self.stream_by_band_by_channel[band][channel] = self.phase[ch_idx]

    def get_median_bias_wl(self, fmin=float('-int'), fmax=(float('inf'))):
        """
        Do some per-band analysis on the time stream data.

        Populate the per-band dictionaries of self.wl_list (self.wl_list[band] is a list)
        and self.wl_median (self.wl_median[band] is a float).

        :param fmin: float - default is float('-inf'). The minimum frequency to consider,
                             frequencies below this value will be masked during calculations.
        :param fmax: float - default is float('inf'). The maximum frequency to consider,
                             frequencies above this value will be masked during calculations.
        """
        self.wl_list = {}
        self.wl_median = {}
        for band in sorted(self.stream_by_band_by_channel.keys()):
            self.wl_list[band] = []
            channels_this_band = self.stream_by_band_by_channel[band]
            for channel in sorted(channels_this_band.keys()):
                stream_single_channel = channels_this_band[channel]
                f, Pxx = signal.welch(stream_single_channel, nperseg=self.nperseg, fs=self.fs, detrend=self.detrend)
                Pxx = np.sqrt(Pxx)
                fmask = (fmin < f) & (f < fmax)

                wl = np.median(Pxx[fmask])
                self.wl_list[band].append(wl)
            self.wl_median = np.median(self.wl_list[band])

    def plot_ts(self):
        """
        Plot the time stream data.

        No user options, only hacking these values...
        """
        # # Plot layout and other defaults
        # Guide lines and hard coded plot elements
        psd_guild_lines_hz = [1.4, 60.0]
        psd_guild_line_colors = ['darkorchid', 'firebrick']
        psd_guild_line_alpha = 0.2
        psd_y_min_pa_roothz = 1.0
        psd_y_max_pa_roothz = 1.0e4

        # figure margins in figure coordinates
        frame_on = False
        left = 0.08
        bottom = 0.02
        right = 0.99
        top = 0.98
        figure_width_inches = 12
        figure_height_inches = 24
        inter_band_spacing_x = 0.05
        inter_band_spacing_y = 0.05
        phase_to_psd_height_ratio = 0.5
        between_phase_and_psd_spacing_y = 0.02
        # basic axis layout choices
        columns = 2
        # layout calculations
        sorted_bands = sorted(self.stream_by_band_by_channel.keys())
        num_of_bands = len(sorted_bands)
        rows = int(np.ceil(num_of_bands / float(columns)))
        single_band_width = (right - left - ((columns - 1) * inter_band_spacing_x)) / float(columns)
        single_band_height = (top - bottom - ((rows - 1) * inter_band_spacing_y)) / float(rows)
        available_band_height = single_band_height - between_phase_and_psd_spacing_y
        single_phase_height = available_band_height * phase_to_psd_height_ratio / (phase_to_psd_height_ratio + 1.0)
        single_psd_height = available_band_height - single_phase_height

        # figure and axis-handle setup
        fig = plt.figure(figsize=(figure_width_inches, figure_height_inches))
        ax_dict_phase = {}
        ax_dict_psd = {}
        left_ax_coord = left
        top_ax_phase_coord = top
        for counter, band in list(enumerate(sorted_bands)):
            # local calculations
            bottom_ax_phase_coord = top_ax_phase_coord - single_phase_height
            bottom_ax_psd_coord = bottom_ax_phase_coord - between_phase_and_psd_spacing_y - single_psd_height
            phase_coords = [left_ax_coord, bottom_ax_phase_coord, single_band_width, single_phase_height]
            psd_coords = [left_ax_coord, bottom_ax_psd_coord, single_band_width, single_psd_height]
            # create the axis handles
            ax_dict_phase[band] = fig.add_axes(phase_coords, frameon=frame_on)
            ax_dict_psd[band] = fig.add_axes(psd_coords, frameon=frame_on)
            # reset things for the next loop
            if ((counter + 1) % columns) == 0:
                # case where the next axis is on a new row
                left_ax_coord = left
                top_ax_phase_coord -= single_band_height + inter_band_spacing_y
            else:
                # case where the next axis is in to next column
                left_ax_coord += single_band_width + inter_band_spacing_x

        # plot the band channel data
        for counter2, band in list(enumerate(sorted_bands)):
            stream_single_band = self.stream_by_band_by_channel[band]
            ax_phase_this_band = ax_dict_phase[band]
            ax_psd_this_band = ax_dict_psd[band]
            for channel in sorted(stream_single_band.keys()):
                # phase
                stream_single_channel = stream_single_band[channel]
                stream_single_channel_norm = stream_single_channel - np.mean(stream_single_channel)
                ax_phase_this_band.plot(self.t_array, stream_single_channel_norm, color='C0', alpha=0.002)
                # psd
                f, Pxx = signal.welch(stream_single_channel, nperseg=self.nperseg, fs=self.fs,
                                      detrend=self.detrend)
                Pxx = np.sqrt(Pxx)
                ax_psd_this_band.loglog(f, Pxx, color='C0', alpha=0.002)

            # phase
            band_yield = len(stream_single_band)
            ax_phase_this_band.set_xlabel('time [s]')
            if counter2 % columns == 0:
                ax_phase_this_band.set_ylabel('Phase [pA]')
            ax_phase_this_band.grid()
            ax_phase_this_band.set_title(f'band {band} yield {band_yield}')
            ax_phase_this_band.set_ylim([-10000, 10000])
            # psd
            for line_hz, line_color in zip(psd_guild_lines_hz, psd_guild_line_colors):
                # add the guild lines to the plots
                ax_psd_this_band.plot([line_hz, line_hz],
                                      [psd_y_min_pa_roothz, psd_y_max_pa_roothz],
                                      color=line_color, alpha=psd_guild_line_alpha,
                                      ls='dashed')
                ax_psd_this_band.text(x=line_hz, y=psd_y_max_pa_roothz, s=f"{line_hz} Hz", color=line_color,
                                      rotation=315, alpha=0.6,
                                      ha='left', va='top')
            ax_psd_this_band.set_xlabel('Frequency [Hz]')
            if counter2 % columns == 0:
                ax_psd_this_band.set_ylabel('Amp [pA/rtHz]')
            ax_psd_this_band.grid()
            ax_psd_this_band.set_ylim([psd_y_min_pa_roothz, psd_y_max_pa_roothz])

        save_name = f'{self.start_time}_band_noise_stack.png'
        print(f'Saving plot to {os.path.join(self.S.plot_dir, save_name)}')
        plt.savefig(os.path.join(self.S.plot_dir, save_name))
        plt.close(fig=fig)


if __name__ == '__main__':
    """
    The code below will only run if the file is run directly, but not if elements from this file are imported.
    For example:
        python3 time_steams.py -args_for_argparse
    will have __name__ == '__main__' as True, and the code below will run locally.
    
    
    """
    # set up the parser for this Script
    parser = argparse.ArgumentParser(description='Parser for time_stream.py script.')
    parser.add_argument('slot_num', type=int, metavar='slot_num', nargs=1,
                        help='The SMuRF slot number (int) used for identification and control.')
    parser.add_argument('--verbose', dest='verbose', action='store_true',
                        help="Turns on printed output from the script. The default is --verbose." +
                             "--no-verbose has minimal (or no) print statements.")
    parser.add_argument('--no-verbose', dest='verbose', action='store_false', default=True,
                        help="Turns off printed output from the script. The default is --verbose." +
                             "--no-verbose has minimal (or no) print statements.")
    parser.add_argument('--plot', dest='plot', action='store_true', default=False,
                        help="Renders and timestream data as a dynamic plot and saves the result. " +
                             "The default is --no-plot  " +
                             "Warning: rendering plots can take significant resources and cause slow performance.")
    parser.add_argument('--no-plot', dest='plot', action='store_false',
                        help="No plots are rendered for the loaded time stream data." +
                             "The default is --no-plot.")
    parser.add_argument('--nperseg', dest='nperseg', type=int, default=2**18,
                        help="int, optional, default is 2**18. The number of samples used in the PSD estimator. " +
                             "See scipy.signal.welch.")
    parser.add_argument('--path', dest='data_path', default=None,
                        help="A path to existing time stream data. Default is None, " +
                             "which takes new timestream data and sets the resulting file as the path.")
    # parse the args for this script.
    args = parser.parse_args()

    # load a single S, or SMuRF controller instance for a given slot number
    load_s = LoadS(slot_nums=[args.slot_num], verbose=args.verbose)
    cfg = load_s.cfg_dict[args.slot_num]
    S = load_s.S_dict[args.slot_num]

    # Timestream data
    timestream_data = TimeStreamData(S, cfg, nperseg=args.nperseg, data_path=args.data_path, verbose=args.verbose)
    if args.plot:
        timestream_data.plot_ts()
