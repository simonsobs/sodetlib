"""
Princeton Detector Mapping.
Original Source: Kaiwen Zheng
Additional Author(s): Caleb Wheeler

Collect tune data and metadata from a variety of sources, to make useful associations, Include output CSV files
visualizations.
"""
import os
import numpy as np
# custom packages
from sodetlib.detmap.simple_csv import read_csv
from sodetlib.detmap.layout_data import get_layout_data
from sodetlib.detmap.channel_assignment import OperateTuneData
from sodetlib.detmap.example.read_iv import read_psat  # soon to be deprecated
from sodetlib.detmap.detmap_config import designfile_default_path, waferfile_default_path, \
    mux_pos_to_mux_band_file_default_path, output_csv_default_filename


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


def get_formatted_metadata(design_file=designfile_default_path, waferfile=waferfile_default_path,
                           dark_bias_lines=None) -> (OperateTuneData, dict):
    """Read in design metadata for a detector array and return instances data class OperateTuneData.

    Parameters
    ----------
    design_file : obj: str, optional
        A path string to the frequency-design file. The default frequency-design file is a versioned file located at
        sodetlib/sodetlib/detmap/meta/umux_32_map.pkl, a human readable version is also available at
        sodetlib/sodetlib/detmap/meta/umux_32_map.csv.
    waferfile : obj: str, optional
        A path string to the wafer-layout-design file. The default wafer-layout-design file is a versioned file located
        at sodetlib/sodetlib/detmap/meta/copper_map_corrected.csv.
    dark_bias_lines : obj: iter, optional
        An iterable, like a list, of integers corresponding to bias line numbers that a covered and not optically
        coupled for a given measurement.

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
    design_data = OperateTuneData(design_file_path=design_file)
    # get the UFM layout metadata (mux_layout_position and bond_pad mapping)
    layout_data = get_layout_data(waferfile, dark_bias_lines=dark_bias_lines, plot=False)
    return design_data, layout_data


def add_metadata_and_get_output(tune_data: OperateTuneData, design_data: OperateTuneData, layout_data: dict,
                                output_path_csv, do_csv_output=True,
                                mapping_strategy='map_by_res_index') -> OperateTuneData:
    """General process for and instance of OperateTuneData (tune_data) with metadata (design_data, layout_data).

    Writes a CSV file with combined tune and metadata, and returns an instance of OperateTuneData that is fully
    populated with metadata.

    Parameters
    ----------
    tune_data : OperateTuneData
        An instance of the OperateTuneData data class for measurements of resonate frequencies on a single
        detector focal plane module. See: sodetlib/sodetlib/detmap/channel_assignment.py. This instance contains
        measured data.
    design_data : OperateTuneData
        An instance of the OperateTuneData data class for measurements of resonate frequencies on a single
        detector focal plane module. See: sodetlib/sodetlib/detmap/channel_assignment.py. This instance contains
        design data.
    layout_data :  dict
        A two level dictionary with a primary key of mux_layout_position and a
        secondary key of bond_pad to access a dictionary of detector
        information. In particular, bandpass column indicates 90ghz, 150ghz,
        D for dark detectors which is 90ghz but has different property as optical
        ones, and NC for no-coupled resonators.
    output_path_csv : str
        A string for the and output path for a CSV file with combined tune and metadata
    do_csv_output : bool, optional
        True makes a CSV output file using OperateTuneData.write_csv() method. False omits this step.
    mapping_strategy : object: string, int, float, optional
        A variable used to select the strategy for mapping measured resonator frequency data to designed frequencies.
        This is mapping is not a one-to-one process and can be done in a number of ways.
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

    Returns
    -------
    OperateTuneData
        An instance of the OperateTuneData data class for measurements of resonate frequencies on a single
        detector focal plane module. See: sodetlib/sodetlib/detmap/channel_assignment.py.
        This instance is fully populated with metadata.
    """
    # update the tune_data collections to include design data.
    if design_data is not None:
        tune_data.map_design_data(design_data=design_data, mapping_strategy=mapping_strategy)
        # update the tune data to include the layout data.
        if layout_data is not None:
            tune_data.map_layout_data(layout_data=layout_data)
    if do_csv_output:
        # write a CSV file of this data
        tune_data.write_csv(output_path_csv=output_path_csv)
    return tune_data


def make_map_smurf(tunefile, north_is_highband: bool, design_file=designfile_default_path,
                   waferfile=waferfile_default_path, layout_position_path=mux_pos_to_mux_band_file_default_path,
                   dark_bias_lines=None, output_path_csv=None, do_csv_output=True,
                   mapping_strategy='map_by_res_index') -> OperateTuneData:
    """A recipe for obtaining an instance of OperateTuneData from a SMuRF tunefile that is full populated with metadata.

    Parameters
    ----------
    tunefile : obj: str
        A path string to a SMuRF tunefile, a specifically formatted binary pickle with .npy extension. This file stores
        aquired frequency data for a comb of resonator on a single detector focal plane module.
    north_is_highband : bool
        A Toggle that solves the array mapping ambiguity. Consider this the question "is the 'North' side of a
        given detector array is the SMuRF highband?" True or False.
    design_file : obj: str, optional
        A path string to the frequency-design file. The default frequency-design file is a versioned file located at
        sodetlib/sodetlib/detmap/meta/umux_32_map.pkl, a human-readable version is also available at
        sodetlib/sodetlib/detmap/meta/umux_32_map.csv.
    waferfile : obj: str, optional
        A path string to the wafer-layout-design file. The default wafer-layout-design file is a versioned file located
        at sodetlib/sodetlib/detmap/meta/copper_map_corrected.csv.
    layout_position_path : obj: str, optional
        A path string to the umux-position-on wafer to umux-band. This is required to give positional context to the
        wafer-layout-design metadata. The default file is a versioned file located at
        sodetlib/sodetlib/detmap/meta/mux_pos_num_to_mux_band_num.csv.
    dark_bias_lines : obj:iter, optional
        An iterable, like a list, of integers corresponding to bias line numbers that a covered and not optically
        coupled for a given measurement.
    output_path_csv : obj: str, optional
        A string for the and output path for a CSV file with combined tune and metadata. By default, the output
        directory is the same as that of the `tunefile` argument above. The default filename is prepended with 'smurf_'
        with a suffix determined by `output_csv_default_filename` in sodetlib/sodetlib/detmap/detmap_config.py
    do_csv_output : bool, optional
        True makes a CSV output file using OperateTuneData.write_csv() method. False omits this step.
    mapping_strategy : object: string, int, float, optional
        See the docstring in the function add_metadata_and_get_output().

    Returns
    -------
    OperateTuneData
        An instance of the OperateTuneData data class for measurements of resonate frequencies on a single
        detector focal plane module. The formatted of the measured data is expected to be derived from a SMuRF
        tunefile See: sodetlib/sodetlib/detmap/channel_assignment.py and OperateTuneData.read_tunefile() and within.
        This instance is of OperateTuneData is fully populated with available metadata.

    Example
    -------
    >>> make_map_vna(tunefile='sample_data/1632247315_tune.npy', north_is_highband=False)

    See sodetlib/sodetlib/detmap/examples/demo_simple.ipynb for a basic example like the one above

    See sodetlib/sodetlib/detmap/examples/demo_full.ipynb for useage in a more general example that read inputs from a
    yaml file and downloads a sample data set for experimentation.
    """
    design_data, layout_data = get_formatted_metadata(design_file=design_file, waferfile=waferfile,
                                                      dark_bias_lines=dark_bias_lines)
    if output_path_csv is None:
        # put the output in the dame directory as the tune file.
        output_path_csv = os.path.join(os.path.dirname(tunefile), f'smurf_{output_csv_default_filename}')
    # get the tune file data from smurf
    tune_data_smurf = OperateTuneData(tune_path=tunefile, north_is_highband=north_is_highband,
                                      layout_position_path=layout_position_path)
    return add_metadata_and_get_output(tune_data=tune_data_smurf, design_data=design_data, layout_data=layout_data,
                                       output_path_csv=output_path_csv, do_csv_output=do_csv_output,
                                       mapping_strategy=mapping_strategy)


def make_map_vna(tune_data_vna_output_filename,
                 north_is_highband: bool,
                 path_north_side_vna=None, path_south_side_vna=None,
                 shift_mhz=10.0,
                 design_file=designfile_default_path, waferfile=waferfile_default_path,
                 layout_position_path=mux_pos_to_mux_band_file_default_path, dark_bias_lines=None,
                 output_path_csv=None, do_csv_output=True, mapping_strategy='map_by_res_index') -> OperateTuneData:
    """A recipe for obtaining an instance of OperateTuneData from a pair of arrays of frequency data from a VNA.

        Two array of frequency data (north and south sides of an array) are allowed as input for a single detector focal
        plane module. The returned instance of Operate Tune data is fully populated with metadata.

    Parameters
    ----------
    tune_data_vna_output_filename : obj: str
        The path str of an intermediate data product resulting from the read-in of two VNA array files,
        i.e. the targets of the arguments `path_north_side_vna` and `path_south_side_vna`
        If this file exists, then it is used as the primary data input of measured resonator frequencies. If the file
        does not exist then at least one of the optional arguments `path_north_side_vna` and `path_south_side_vna` must
        be specified. The targets of these files is then combined to a single instance of OperateTuneData which is then
        outputted as a CSV file with at the path specified by tune_data_vna_output_filename using the method
        OperateTuneData.write_csv(). This file stores aquired frequency data for a comb of resonator on a single
        detector focal plane module.
    north_is_highband : bool
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
    design_file : obj: str, optional
        A path string to the frequency-design file. The default frequency-design file is a versioned file located at
        sodetlib/sodetlib/detmap/meta/umux_32_map.pkl, a human-readable version is also available at
        sodetlib/sodetlib/detmap/meta/umux_32_map.csv.
    waferfile : obj: str, optional
        A path string to the wafer-layout-design file. The default wafer-layout-design file is a versioned file located
        at sodetlib/sodetlib/detmap/meta/copper_map_corrected.csv.
    layout_position_path : obj: str, optional
        A path string to the umux-position-on wafer to umux-band. This is required to give positional context to the
        wafer-layout-design metadata. The default file is a versioned file located at
        sodetlib/sodetlib/detmap/meta/mux_pos_num_to_mux_band_num.csv.
    dark_bias_lines : obj:iter, optional
        An iterable, like a list, of integers corresponding to bias line numbers that a covered and not optically
        coupled for a given measurement.
    output_path_csv : obj: str, optional
        A string for the and output path for a CSV file with combined tune and metadata. By default, the output
        directory is the same that of the `tune_data_vna_output_filename` argument above. The default filename is
        prepended with 'vna_' with a suffix determined by `output_csv_default_filename` in
        sodetlib/sodetlib/detmap/detmap_config.py.
    do_csv_output : bool, optional
        True makes a CSV output file using OperateTuneData.write_csv() method. False omits this step.
    mapping_strategy : object: string, int, float, optional
        See the docstring in the function add_metadata_and_get_output().

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

    Example
    -------
    >>> make_map_vna(tune_data_vna_output_filename='sample_data/tune_data_vna.csv', north_is_highband=False,
                     path_north_side_vna='sample_data/north_side_vna_farray.csv',
                     path_south_side_vna='sample_data/south_side_vna_farray.csv')

    See sodetlib/sodetlib/detmap/examples/demo_full.ipynb for useage in a more general that read inputs from a yaml
    file and downloads a sample data set for experimentation.
    """
    if output_path_csv is None:
        # put the output in the dame directory as the tune file.
        output_path_csv = os.path.join(os.path.dirname(tune_data_vna_output_filename),
                                       f'vna_{output_csv_default_filename}')

    # # Metadata
    # get the design file for the resonators
    design_data = OperateTuneData(design_file_path=design_file)
    # get the UFM layout metadata (mux_layout_position and bond_pad mapping)
    layout_data = get_layout_data(waferfile, dark_bias_lines=dark_bias_lines, plot=False)

    # parse/get date from a Vector Network Analyzer
    if not os.path.exists(tune_data_vna_output_filename):
        # Run Kaiwen's peak finder and return a data structure that relates frequency to smurf band and channel number.
        tune_data_raw_vna = assign_channel_from_vna(path_north_side_vna=path_north_side_vna,
                                                    path_south_side_vna=path_south_side_vna,
                                                    north_is_highband=north_is_highband, shift_mhz=shift_mhz)
        # write this data to skip this step next time and simply read in these results
        tune_data_raw_vna.write_csv(output_path_csv=tune_data_vna_output_filename)
    # reload the tune data from the csv file, for constant behavior on first run or many runs
    tune_data_vna = OperateTuneData(tune_path=tune_data_vna_output_filename,
                                    layout_position_path=layout_position_path)

    return add_metadata_and_get_output(tune_data=tune_data_vna, design_data=design_data, layout_data=layout_data,
                                       output_path_csv=output_path_csv, do_csv_output=do_csv_output,
                                       mapping_strategy=mapping_strategy)


def make_map_g3_timestream(timestream, north_is_highband: bool, design_file=designfile_default_path,
                           waferfile=waferfile_default_path, layout_position_path=mux_pos_to_mux_band_file_default_path,
                           dark_bias_lines=None, output_path_csv=None, do_csv_output=True,
                           mapping_strategy='map_by_res_index') -> OperateTuneData:
    """A recipe for obtaining an instance of OperateTuneData from a SMuRF tunefile that is full populated with metadata.

    Parameters
    ----------
    timestream : obj: str
        A path string to a g3 time sttreamfile, a specifically formatted binary pickle with .npy extension.
        This file stores aquired frequency data for a comb of resonator on a single detector focal plane module.
    north_is_highband : bool
        A Toggle that solves the array mapping ambiguity. Consider this the question "is the 'North' side of a
        given detector array is the SMuRF highband?" True or False.
    design_file : obj: str, optional
        A path string to the frequency-design file. The default frequency-design file is a versioned file located at
        sodetlib/sodetlib/detmap/meta/umux_32_map.pkl, a human-readable version is also available at
        sodetlib/sodetlib/detmap/meta/umux_32_map.csv.
    waferfile : obj: str, optional
        A path string to the wafer-layout-design file. The default wafer-layout-design file is a versioned file located
        at sodetlib/sodetlib/detmap/meta/copper_map_corrected.csv.
    layout_position_path : obj: str, optional
        A path string to the umux-position-on wafer to umux-band. This is required to give positional context to the
        wafer-layout-design metadata. The default file is a versioned file located at
        sodetlib/sodetlib/detmap/meta/mux_pos_num_to_mux_band_num.csv.
    dark_bias_lines : obj:iter, optional
        An iterable, like a list, of integers corresponding to bias line numbers that a covered and not optically
        coupled for a given measurement.
    output_path_csv : obj: str, optional
        A string for the and output path for a CSV file with combined tune and metadata. By default, the output
        directory is the same as that of the `timestream` argument above. The default filename is prepended with 'g3ts_'
        with a suffix determined by `output_csv_default_filename` in sodetlib/sodetlib/detmap/detmap_config.py
    do_csv_output : bool, optional
        True makes a CSV output file using OperateTuneData.write_csv() method. False omits this step.
    mapping_strategy : object: string, int, float, optional
        See the docstring in the function add_metadata_and_get_output().

    Returns
    -------
    OperateTuneData
        An instance of the OperateTuneData data class for measurements of resonate frequencies on a single
        detector focal plane module. The formatted of the measured data is expected to be derived from a g3timestream
        See: sodetlib/sodetlib/detmap/channel_assignment.py and OperateTuneData.read_tunefile() and within.
        This instance is of OperateTuneData is fully populated with available metadata.

    Example
    -------
    >>> make_map_g3_timestream(timestream='sample_data/freq_map.npy', north_is_highband=False)

    See sodetlib/sodetlib/detmap/examples/demo_simple.ipynb for a basic example like the one above.
    """
    design_data, layout_data = get_formatted_metadata(design_file=design_file, waferfile=waferfile,
                                                      dark_bias_lines=dark_bias_lines)
    if output_path_csv is None:
        # put the output in the dame directory as the tune file.
        output_path_csv = os.path.join(os.path.dirname(timestream), f'g3ts_{output_csv_default_filename}')
    # get the tune file data from smurf
    tune_data_g3ts = OperateTuneData(tune_path=timestream, is_g3timestream=True,
                                     north_is_highband=north_is_highband,
                                     layout_position_path=layout_position_path)
    return add_metadata_and_get_output(tune_data=tune_data_g3ts, design_data=design_data, layout_data=layout_data,
                                       output_path_csv=output_path_csv, do_csv_output=do_csv_output,
                                       mapping_strategy=mapping_strategy)


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
