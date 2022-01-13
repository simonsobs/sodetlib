"""
Princeton Detector Mapping.
Original Source: Kaiwen Zheng
Additional Author(s): Caleb Wheeler

Collect tune data and metadata from a variety of sources, to make useful associations, including detector maps.
"""
import os
import numpy as np
# custom packages
from sodetlib.detmap.simple_csv import read_csv
from sodetlib.detmap.layout_data import get_layout_data
from sodetlib.detmap.channel_assignment import OperateTuneData
from sodetlib.detmap.example.read_iv import read_psat  # soon to be deprecated
from sodetlib.detmap.detmap_conifg import design_file_path, waferfile_path, mux_pos_num_to_mux_band_num_path


def assign_channel_from_vna(north_is_highband, path_north_side_vna=None, path_south_side_vna=None, shift_mhz=10.0):
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


def make_map_smurf(tunefile, north_is_highband=None, design_data=None, layout_position_path=None,
                   layout_data=None, csv_filename=None):
    # get the tune file data from smurf
    tune_data_smurf = OperateTuneData(tune_path=tunefile, north_is_highband=north_is_highband,
                                      layout_position_path=layout_position_path)
    # update the tune_data collections to include design data.
    if design_data is not None:
        tune_data_smurf.map_design_data(design_data=design_data, mapping_strategy='map_by_res_index')  # 'map_by_freq')
        # update the tune data to include the layout data.
        if layout_data is not None:
            tune_data_smurf.map_layout_data(layout_data=layout_data)
    if csv_filename is not None:
        # write a CSV file of this data
        tune_data_smurf.write_csv(output_path_csv=csv_filename)
    return tune_data_smurf


def make_map_vna(tune_data_vna_output_filename='tune_data_vna.csv',
                 path_north_side_vna=None, path_south_side_vna=None,
                 north_is_highband=None, shift_mhz=10.0,
                 design_data=None, layout_position_path=None, layout_data=None, csv_filename=None):

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

    if design_data is not None:
        # update the tune_data collections to include design data.
        tune_data_vna.map_design_data(design_data=design_data)
        if layout_data is not None:
            # update the tune data to include the layout data.
            tune_data_vna.map_layout_data(layout_data=layout_data)
    if csv_filename is not None:
        # write a CSV file of this data
        tune_data_vna.write_csv(output_path_csv=csv_filename)
    return tune_data_vna


def psat_map(tunefile, north_is_highband, output_filename_smurf=None,
             cold_ramp_file=None,
             design_file=design_file_path, waferfile=waferfile_path,
             mux_pos_num_to_mux_band_num=mux_pos_num_to_mux_band_num_path,
             dark_bias_lines=None,
             psat_temp_k=9.0, psat_show_plot=False, psat_save_plot=True):
    # # Metadata
    # get the design file for the resonators
    design_data_example = OperateTuneData(design_file_path=design_file)
    # get the UFM layout metadata (mux_layout_position and bond_pad mapping)
    layout_data_example = get_layout_data(waferfile, dark_bias_lines=dark_bias_lines, plot=False)

    # # Smurf Tune File
    # read the tunefile and initialize the data instance
    tune_data_smurf = make_map_smurf(tunefile=tunefile, north_is_highband=north_is_highband,
                                     design_data=design_data_example,
                                     layout_position_path=mux_pos_num_to_mux_band_num,
                                     layout_data=layout_data_example,
                                     csv_filename=output_filename_smurf)
    if cold_ramp_file is not None:
        # Optical power data from validation of dark bias lines.
        _cold_ramp_data_by_column, cold_ramp_data_by_row = read_csv(path=cold_ramp_file)
        coldload_ivs = [data_row for data_row in cold_ramp_data_by_row if data_row['note'].lower() == 'iv']

        # read in the sample psat data.
        _psat_by_band_chan, psat_by_temp = read_psat(coldload_ivs=coldload_ivs, make_plot=False)

        # example plots
        for bandpass_target, psat_min, psat_max in [(90, 0.0, 3.0e-12),
                                                    (150, 0.0, 6.0e-12)]:
            tune_data_smurf.plot_with_psat(psat_by_temp=psat_by_temp, bandpass_target=bandpass_target,
                                           temp_k=psat_temp_k, psat_min=psat_min, psat_max=psat_max,
                                           show_plot=psat_show_plot, save_plot=psat_save_plot)
