"""
Princeton Detector Mapping.
Original Source: Kaiwen Zheng
Additional Author(s): Caleb Wheeler

Collect tune data and metadata from a variety of sources, to make useful associations, Include output CSV files
visualizations.
"""
import os
import numpy as np
from typing import Union
# custom packages
from sodetlib.detmap.simple_csv import read_csv
from sodetlib.detmap.meta_select import get_metadata_files
from sodetlib.detmap.layout_data import get_layout_data
from sodetlib.detmap.channel_assignment import OperateTuneData
from sodetlib.detmap.example.read_iv import read_psat  # soon to be deprecated


def assign_channel_from_vna(north_is_highband: bool, path_north_side_vna=None, path_south_side_vna=None,
                            shift_mhz=10.0) -> OperateTuneData:
    """Read VNA data and export it into an instance of OperateTuneData.

    The target paths for VNA data are expected to be a single column CSV file with the column header 'freq_mhz'.

    Parameters
    ----------
    north_is_highband :  bool
        A Toggle that solves the array mapping ambiguity. Consider this the question "is the 'North' side of a
        given detector array is the SMuRF highband?" True or False.
    path_north_side_vna : obj:`str`, optional
        An optional path str to measured frequency array file measured by a VNA for the 'North' side of a detector
        array. When `None`, it is expected that no frequency data is available for the North side. The target path
        is expected to be a single column CSV file with the column header 'freq_mhz'.
    path_south_side_vna : obj:`str`, optional
        An optional path str to measured frequency array file measured by a VNA for the 'South' side of a detector
        array. When `None`, it is expected that no frequency data is available for the South side. The target path
        is expected to be a single column CSV file with the column header 'freq_mhz'.
    shift_mhz : float, optional
        It is known that the measured frequencies of the SMuRF system are shifted compared to VNA measurements by
        approximately 10.0 MHz. This applies a shift to the VNA data to deliver projection of  the measured VNA
        frequencies into the reference frame of SMuRF data. The default shift is + 10.0 MHz added to the VNA data.
    Returns
    -------
    OperateTuneData
        An instance of the OperateTuneData data class for measurements of resonate frequencies on a single
        detector focal plane module. See: sodetlib/sodetlib/detmap/channel_assignment.py

    Raises
    -------
    FileNotFoundError
        if path_north_side_vna is None and path_south_side_vna is None. The user is not allowed to use this function
        make an instance of OperateTuneData that has no measured frequencies from a VNA.

    """
    if path_north_side_vna is None and path_south_side_vna is None:
        raise FileNotFoundError("Both North and South peak array files were None")
    # cast the file data into numpy arrays
    if path_north_side_vna is not None:
        north_res_mhz_column, _north_res_mhz_by_row = read_csv(path=path_north_side_vna)
        north_res_mhz = np.array(north_res_mhz_column['freq_mhz'])
    else:
        north_res_mhz = np.array([])

    if path_south_side_vna is not None:
        south_res_mhz_column, _south_res_mhz_by_row = read_csv(path=path_south_side_vna)
        south_res_mhz = np.array(south_res_mhz_column['freq_mhz'])
    else:
        south_res_mhz = np.array([])
    # determine which set goes into upper and lower smurf bands, respectively
    if north_is_highband:
        # North Side: to emulate the smurf tune file output we add 2000 MHz to the 'highband' (upper band)
        upper_res = north_res_mhz
        lower_res = south_res_mhz
        upper_res_is_north = True
    else:
        # South side: to emulate the smurf tune file output we add 2000 MHz to the 'highband' (upper band)
        upper_res = south_res_mhz
        lower_res = north_res_mhz
        upper_res_is_north = False
    # this is simple the opposite of the upper band bool value
    lower_res_is_north = not upper_res_is_north
    # put the data into bands and channels.
    upper_res_tune_data = OperateTuneData()
    upper_res_tune_data.from_peak_array(peak_array_mhz=upper_res, is_north=upper_res_is_north,
                                        is_highband=True, shift_mhz=shift_mhz, smurf_bands=None)
    lower_res_tune_data = OperateTuneData()
    lower_res_tune_data.from_peak_array(peak_array_mhz=lower_res, is_north=lower_res_is_north,
                                        is_highband=False, shift_mhz=shift_mhz, smurf_bands=None)
    return upper_res_tune_data + lower_res_tune_data


class MapMaker:
    output_csv_default_filename = 'pixel_freq_mapping.csv'

    def __init__(self, north_is_highband: bool, array_name=None,
                 mapping_strategy: Union[int, float, str, None] = 'map_by_freq', dark_bias_lines=None,
                 do_csv_output: bool = True, show_layout_plot: bool = False, save_layout_plot: bool = True,
                 output_dir=None,
                 abs_path_metadata_files=None,
                 verbose: bool = True):
        """Read in design metadata for a detector array and return instances data class OperateTuneData.

        Parameters
        ----------
        dark_bias_lines : obj: iter, optional
            An iterable, like a list, of integers corresponding to bias line numbers that a covered and not optically
            coupled for a given measurement.

        do_csv_output : bool, optional
            True makes a CSV output file using OperateTuneData.write_csv() method. False omits this step.
        show_layout_plot: bool, optional
            True displays an interactive layout plot using OperateTuneData.plot_with_layout() method.
            False, False does not show a plot and is the default. If save_layout_plot is also False, the computationally
            expensive plotting method is skipped.
        save_layout_plot: bool, optional
            True Saves an interactive layout plot using OperateTuneData.plot_with_layout() method.
            False, False does not save a plot. By default, a plot is saved. If show_layout_plot is also False, the
            computationally expensive plotting method is skipped.
        mapping_strategy : object: str, int, float, optional
            A variable used to select the strategy for mapping measured resonator frequency data to designed
            frequencies. This is mapping is not a one-to-one process and can be done in a number of ways.
            Strategies that require no additional metadata include:
            Mapping by resonator index, select with: mapping_strategy in {0, '0', 0.0, 'map_by_res_index'}
                This is the lowest computation effort mapping strategy that maps that simple maps resonators to design
                frequencies in the order in which they were measurer within each SMuRF band.
            Mapping by Frequency, select with: mapping_strategy in {1, 1.0, '1', 'map_by_freq'}
                This storage occurs on several levels of frequency corrections. The first is a linear (mx + b) remapping
                of measured frequencies across the whole range of designed frequencies. Minimizing the frequency distance
                between measured and design frequencies for 4 SMuRF bands. This process is then repeated with a second
                remapping of measured freelance but, no only considering a single smurf band. The final per-SMuRF band
                process is a healing stage to create a 1-1 mapping of measured to design resonators where measured
                resonators mapping to the same design resonator are moved to a nearest neighbor pushing the other mapped
                resonators to fill in the gaps where designed resonators previously did not have measured counterparts.
        """

        self.north_is_highband = north_is_highband
        self.array_name = array_name

        self.mapping_strategy = mapping_strategy
        self.dark_bias_lines = dark_bias_lines

        self.do_csv_output = do_csv_output
        self.show_layout_plot = show_layout_plot
        self.save_layout_plot = save_layout_plot
        self.output_dir = output_dir

        self.waferfile_path, self.designfile_path, self.mux_pos_to_mux_band_file_path = \
            get_metadata_files(array_name=self.array_name, abs_path_metadata_files=abs_path_metadata_files,
                               verbose=verbose)

        self.design_data, self.wafer_layout_data = self.load_metadata()

    def load_metadata(self, show_wafer_layout_plot=False) -> (OperateTuneData, dict):
        """
        Returns
        -------
        OperateTuneData, wafer_info

            OperateTuneData:
            An instance of the OperateTuneData data class for measurements of resonate frequencies on a single
            detector focal plane module. See: sodetlib/sodetlib/detmap/channel_assignment.py

            wafer_info:
            A two level dictionary with a primary key of mux_layout_position and a
            secondary key of bond_pad to access a dictionary of detector
            information. In particular, bandpass column indicates 90ghz, 150ghz,
            D for dark detectors which is 90ghz but has different property as optical
            ones, and NC for no-coupled resonators.
        """

        # get the design file for the resonators
        design_data = OperateTuneData(design_file_path=self.designfile_path,
                                      north_is_highband=self.north_is_highband,
                                      layout_position_path=self.mux_pos_to_mux_band_file_path)
        # get the UFM layout metadata (mux_layout_position and bond_pad mapping)
        wafer_layout_data = get_layout_data(self.waferfile_path, dark_bias_lines=self.dark_bias_lines,
                                            plot=show_wafer_layout_plot)
        return design_data, wafer_layout_data

    def add_metadata_and_get_output(self, tune_data: OperateTuneData, output_path_csv, layout_plot_path=None
                                    ) -> OperateTuneData:
        """General process for and instance of OperateTuneData (tune_data) with metadata (design_data, layout_data).

        Writes a CSV file with combined tune and metadata, and returns an instance of OperateTuneData that is fully
        populated with metadata.

        Parameters
        ----------
        tune_data : OperateTuneData
            An instance of the OperateTuneData data class for measurements of resonate frequencies on a single
            detector focal plane module. See: sodetlib/sodetlib/detmap/channel_assignment.py. This instance contains
            measured data.
        output_path_csv : str
            A string for an output path for a CSV file with combined tune and metadata
        layout_plot_path : object : str, optional
            A string for a full path and filename for plot created by the OperateTuneData.plot_with_layout() method.
            If None is given, a default filename is chosen.

        Returns
        -------
        OperateTuneData
            An instance of the OperateTuneData data class for measurements of resonate frequencies on a single
            detector focal plane module. See: sodetlib/sodetlib/detmap/channel_assignment.py.
            This instance is fully populated with metadata.
        """
        tune_data.map_design_data(design_data=self.design_data, mapping_strategy=self.mapping_strategy)
        # update the tune data to include the layout data.
        tune_data.map_layout_data(layout_data=self.wafer_layout_data)
        if self.show_layout_plot or self.save_layout_plot:
            tune_data.plot_with_layout(plot_path=layout_plot_path, show_plot=self.show_layout_plot,
                                       save_plot=self.save_layout_plot)
        if self.do_csv_output:
            # write a CSV file of this data
            tune_data.write_csv(output_path_csv=output_path_csv)
        return tune_data

    def get_output_csv_path(self, prefix, data_file):
        if self.output_dir is None:
            output_dir = os.path.dirname(data_file)
        else:
            output_dir = self.output_dir
        # put the output in the dame directory as the tune file.
        output_path_csv = os.path.join(output_dir, f'{prefix}_{self.output_csv_default_filename}')
        return output_path_csv

    def make_map_smurf(self, tunefile, output_path_csv=None, layout_plot_path=None) -> OperateTuneData:
        """A recipe for obtaining an instance of OperateTuneData from a SMuRF tunefile that is populated with metadata.

        ----------
        tunefile : obj: str
            A path string to a SMuRF tunefile, a specifically formatted binary pickle with .npy extension. This file stores
            aquired frequency data for a comb of resonator on a single detector focal plane module.
            north_is_highband : bool
            A Toggle that solves the array mapping ambiguity. Consider this the question "is the 'North' side of a
            given detector array is the SMuRF highband?" True or False.
        output_path_csv : obj: str, optional
            A string for the and output path for a CSV file with combined tune and metadata. By default, the output
            directory is the same as that of the `tunefile` argument above. The default filename is prepended with
            'smurf_' with a suffix determined by `output_csv_default_filename` in
            sodetlib/sodetlib/detmap/detmap_config.py
        layout_plot_path : object : str, optional
            A string for a full path and filename for plot created by the OperateTuneData.plot_with_layout() method.
            If None is given, a default filename is chosen.


        Returns
        -------
        OperateTuneData
            An instance of the OperateTuneData data class for measurements of resonate frequencies on a single
            detector focal plane module. The formatted of the measured data is expected to be derived from a SMuRF
            tunefile See: sodetlib/sodetlib/detmap/channel_assignment.py and OperateTuneData.read_tunefile() and within.
            This instance is of OperateTuneData is fully populated with available metadata.
        """
        if output_path_csv is None:
            output_path_csv = self.get_output_csv_path(prefix='smurf', data_file=tunefile)
        # get the tune file data from smurf
        tune_data_smurf = OperateTuneData(tune_path=tunefile, north_is_highband=self.north_is_highband)
        return self.add_metadata_and_get_output(tune_data=tune_data_smurf, output_path_csv=output_path_csv,
                                                layout_plot_path=layout_plot_path)

    def make_map_vna(self, tune_data_vna_intermediate_filename,
                     path_north_side_vna=None, path_south_side_vna=None,
                     shift_mhz=10.0,
                     output_path_csv=None, layout_plot_path=None) -> OperateTuneData:
        """A recipe for obtaining an instance of OperateTuneData from a pair of arrays of frequency data from a VNA.

            Two array of frequency data (north and south sides of an array) are allowed as input for a single detector focal
            plane module. The returned instance of Operate Tune data is fully populated with metadata.

        Parameters
        ----------
        tune_data_vna_intermediate_filename : obj: str
            The path str of an intermediate data product resulting from the read-in of two VNA array files,
            i.e. the targets of the arguments `path_north_side_vna` and `path_south_side_vna`
            If this file exists, then it is used as the primary data input of measured resonator frequencies. If the file
            does not exist then at least one of the optional arguments `path_north_side_vna` and `path_south_side_vna` must
            be specified. The targets of these files is then combined to a single instance of OperateTuneData which is then
            outputted as a CSV file with at the path specified by tune_data_vna_output_filename using the method
            OperateTuneData.write_csv(). This file stores aquired frequency data for a comb of resonator on a single
            detector focal plane module.
        path_north_side_vna : obj:`str`, optional
            An optional path str to measured frequency array file measured by a VNA for the 'North' side of a detector
            array. When `None`, it is expected that no frequency data is available for the North side. The target path
            is expected to be a single column CSV file with the column header 'freq_mhz'.
        path_south_side_vna : obj:`str`, optional
            An optional path str to measured frequency array file measured by a VNA for the 'South' side of a detector
            array. When `None`, it is expected that no frequency data is available for the South side. The target path
            is expected to be a single column CSV file with the column header 'freq_mhz'.
        shift_mhz : float, optional
            It is known that the measured frequencies of the SMuRF system are shifted compared to VNA measurements by
            approximately 10.0 MHz. This applies a shift to the VNA data to deliver projection of  the measured VNA
            frequencies into the reference frame of SMuRF data. The default shift is + 10.0 MHz added to the VNA data.
        output_path_csv : obj: str, optional
            A string for the and output path for a CSV file with combined tune and metadata. By default, the output
            directory is the same that of the `tune_data_vna_output_filename` argument above. The default filename is
            prepended with 'vna_' with a suffix determined by `output_csv_default_filename` in
            sodetlib/sodetlib/detmap/detmap_config.py.
        layout_plot_path : object : str, optional
            A string for a full path and filename for plot created by the OperateTuneData.plot_with_layout() method.
            If None is given, a default filename is chosen.

        Returns
        -------
        OperateTuneData
            An instance of the OperateTuneData data class for measurements of resonate frequencies on a single
            detector focal plane module. The formatted of the measured data is expected to be derived from a SMuRF
            a CSV file created by OperateTuneData.write_csv() or 1 or 2 csv files of measured frequency arrays, see the
            Parameters' documentation above for `tune_data_vna_output_filename` for more details.
            For more on OperateTuneData See: sodetlib/sodetlib/detmap/channel_assignment.py and
            OperateTuneData.from_peak_array() and/or read_csv.(). This instance is of OperateTuneData is fully populated
            with available metadata.

        """
        if output_path_csv is None:
            output_path_csv = self.get_output_csv_path(prefix='vna', data_file=tune_data_vna_intermediate_filename)
        # parse/get date from a Vector Network Analyzer
        if not os.path.exists(tune_data_vna_intermediate_filename):
            tune_data_raw_vna = assign_channel_from_vna(path_north_side_vna=path_north_side_vna,
                                                        path_south_side_vna=path_south_side_vna,
                                                        north_is_highband=self.north_is_highband, shift_mhz=shift_mhz)
            # write this data to skip this step next time and simply read in these results
            tune_data_raw_vna.write_csv(output_path_csv=tune_data_vna_intermediate_filename)
        # reload the tune data from the csv file, for constant behavior on first run or many runs
        tune_data_vna = OperateTuneData(tune_path=tune_data_vna_intermediate_filename,
                                        north_is_highband=self.north_is_highband)
        return self.add_metadata_and_get_output(tune_data=tune_data_vna, output_path_csv=output_path_csv,
                                                layout_plot_path=layout_plot_path)

    def make_map_g3_timestream(self, timestream, output_path_csv=None,  layout_plot_path=None) -> OperateTuneData:
        """A recipe for obtaining an instance of OperateTuneData from a SMuRF tunefile that is full populated with metadata.

        Parameters
        ----------
        timestream : obj: str
            A path string to a g3 time sttreamfile, a specifically formatted binary pickle with .npy extension.
            This file stores aquired frequency data for a comb of resonator on a single detector focal plane module.
        output_path_csv : obj: str, optional
            A string for the and output path for a CSV file with combined tune and metadata. By default, the output
            directory is the same as that of the `timestream` argument above. The default filename is prepended with 'g3ts_'
            with a suffix determined by `output_csv_default_filename` in sodetlib/sodetlib/detmap/detmap_config.py
        layout_plot_path : object : str, optional
            A string for a full path and filename for plot created by the OperateTuneData.plot_with_layout() method.
            If None is given, a default filename is chosen.

        Returns
        -------
        OperateTuneData
            An instance of the OperateTuneData data class for measurements of resonate frequencies on a single
            detector focal plane module. The formatted of the measured data is expected to be derived from a g3timestream
            See: sodetlib/sodetlib/detmap/channel_assignment.py and OperateTuneData.read_tunefile() and within.
            This instance is of OperateTuneData is fully populated with available metadata.

        """
        if output_path_csv is None:
            output_path_csv = self.get_output_csv_path(prefix='g3ts', data_file=timestream)
        # get the tune file data from smurf
        tune_data_g3ts = OperateTuneData(tune_path=timestream, is_g3timestream=True,
                                         north_is_highband=self.north_is_highband)
        return self.add_metadata_and_get_output(tune_data=tune_data_g3ts, output_path_csv=output_path_csv,
                                                layout_plot_path=layout_plot_path)

    @staticmethod
    def psat_map(tune_data: OperateTuneData, cold_ramp_file, temp_k=9.0, show_plot=False, save_plot=True):
        """ Color maps of psat (power of saturation) for each detector as a function of position on wafer.

        Parameters
        ----------
        tune_data : OperateTuneData
            An instance is of OperateTuneData that is fully populated with metadata, i.e. the outputs of one of the
            recipe functions make_map_smurf(), make_map_vna(), or make_map_g3_timestream() found in
            sodetlib/sodetlib/detmap/makemap.py and docstrings there in.
        cold_ramp_file : obj: str
            A specially formatted CSV file that point to IV data from which psat data is derived. This is to be deprecated
            and with an improved process for identifying IV data from the tunefile's filename.
        temp_k : float, optional
            The temperature in Kelvin at which the psat data was taken. Needs to match the available data set specified in
            `cold_ramp_file`.
        show_plot : bool, optional
            If True, the plotted data is displayed in a pop-up window that must be close be for proceeding to the next plot.
            The default is False.
        save_plot : bool, optional
            If True the plotted data is saved a folder a `plots` directory as specified in the `tune_data.plot_dir`.
            This is usually the same directory that contains the measured frequency data, and is determined in
            OperateTuneData.__init__(). The plot names are dynamically determined from the data's parameters.
            The default is True.
        """
        # Optical power data for validation of dark bias lines.
        _cold_ramp_data_by_column, cold_ramp_data_by_row = read_csv(path=cold_ramp_file)
        coldload_ivs = [data_row for data_row in cold_ramp_data_by_row if data_row['note'].lower() == 'iv']

        # read in the sample psat data.
        _psat_by_band_chan, psat_by_temp = read_psat(coldload_ivs=coldload_ivs, make_plot=False)

        # example plots
        for bandpass_target, psat_min, psat_max in [(90, 0.0, 3.0e-12),
                                                    (150, 0.0, 6.0e-12)]:
            tune_data.plot_with_psat(psat_by_temp=psat_by_temp, bandpass_target=bandpass_target,
                                     temp_k=temp_k, psat_min=psat_min, psat_max=psat_max,
                                     show_plot=show_plot, save_plot=save_plot)
