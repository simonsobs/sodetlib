

class TimeStreamData:
    def __init__(self, data_path=None):
        self.data_path = data_path
        self.start_time = None
        self.fs = None
        self.bands, self.channels = None, None
        self.phase, self.mask, self.tes_bias = None, None, None
        self.sample_nums = None
        self.t_array = None
        self.stream_by_band_by_channel = None

        if self.data_path is not None:
            self.read_ts()

    def take_ts(self, stream_time=20):
        self.start_time = S.get_timestamp()

        # non blocking statement to start time stream and return the dat filename
        try:
            self.data_path = S.stream_data_on()
            # collect stream data
            import time
            print(f'Sleeping for {stream_time} seconds to take time stream data...\n')
            time.sleep(stream_time)
            # end the time stream
        except:
            S.stream_data_off()
            print(f'Stream Off command sent')
            raise
        else:
            S.stream_data_off()
        # read the data that was taken
        self.read_ts()

    def read_ts(self):
        """
        Collect Data and transform it to a shape for plotting and analysis
        """
        # Read the data
        timestamp, self.phase, self.mask, self.tes_bias = S.read_stream_data(self.data_path, return_tes_bias=True)
        self.fs = S.get_sample_frequency()
        dirname, basename = os.path.split(self.data_path)
        file_handle, file_extention = basename.rsplit('.', 1)
        self.start_time = file_handle
        print(f'loaded the .dat file at: {self.data_path}')

        # hard coded variables
        self.bands, self.channels = np.where(self.mask != -1)
        self.phase *= S.pA_per_phi0 / (2.0 * np.pi)  # uA
        self.sample_nums = np.arange(len(self.phase[0]))
        self.t_array = self.sample_nums / self.fs

        # reorganize the data by band then channel
        self.stream_by_band_by_channel = {}
        for band, channel in zip(self.bands, self.channels):
            if band not in self.stream_by_band_by_channel.keys():
                self.stream_by_band_by_channel[band] = {}
            ch_idx = self.mask[band, channel]
            self.stream_by_band_by_channel[band][channel] = self.phase[ch_idx]

    def plot_ts(self):
        """
        Plot layout and other defaults
        """
        import scipy.signal as signal
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
        # PSD plot constants
        nperseg = 2 ** 12
        detrend = 'constant'
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
                f, Pxx = signal.welch(stream_single_channel, nperseg=nperseg, fs=self.fs, detrend=detrend)
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
        print(f'Saving plot to {os.path.join(S.plot_dir, save_name)}')
        plt.savefig(os.path.join(S.plot_dir, save_name))
        plt.close(fig=fig)


if __name__ == "__main__":
    """take new data."""
    time_stream_data = TimeStreamData(data_path=None)
    time_stream_data.take_ts(stream_time=120)
    time_stream_data.plot_ts()

    """read in existing data"""
    # time_stream_data2 = TimeStreamData(data_path=
    #                                        '/data/smurf_data/20211001/crate1slot4/1633107018/outputs/1633116804.dat')
    # time_stream_data2.plot_ts()
