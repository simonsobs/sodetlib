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
from sodetlib.detmap.detmap_conifg import designfile_default_path, waferfile_default_path, \
    mux_pos_to_mux_band_file_default_path, output_csv_default_filename


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


def get_formatted_metadata(design_file=designfile_default_path, waferfile=waferfile_default_path,
                           dark_bias_lines=None):
    # get the design file for the resonators
    design_data = OperateTuneData(design_file_path=design_file)
    # get the UFM layout metadata (mux_layout_position and bond_pad mapping)
    layout_data = get_layout_data(waferfile, dark_bias_lines=dark_bias_lines, plot=False)
    return design_data, layout_data


def add_metadata_and_get_output(tune_data, design_data, layout_data, output_path_csv, do_csv_output=True):
    # update the tune_data collections to include design data.
    if design_data is not None:
        tune_data.map_design_data(design_data=design_data, mapping_strategy='map_by_res_index')  # 'map_by_freq')
        # update the tune data to include the layout data.
        if layout_data is not None:
            tune_data.map_layout_data(layout_data=layout_data)
    if do_csv_output:
        # write a CSV file of this data
        tune_data.write_csv(output_path_csv=output_path_csv)
    return tune_data


def make_map_smurf(tunefile, north_is_highband=None, design_file=designfile_default_path,
                   waferfile=waferfile_default_path, layout_position_path=mux_pos_to_mux_band_file_default_path,
                   dark_bias_lines=None, output_path_csv=None, do_csv_output=True):
    design_data, layout_data = get_formatted_metadata(design_file=design_file, waferfile=waferfile,
                                                      dark_bias_lines=dark_bias_lines)
    if output_path_csv is None:
        # put the output in the dame directory as the tune file.
        output_path_csv = os.path.join(os.path.dirname(tunefile), f'smurf_{output_csv_default_filename}')
    # get the tune file data from smurf
    tune_data_smurf = OperateTuneData(tune_path=tunefile, north_is_highband=north_is_highband,
                                      layout_position_path=layout_position_path)
    return add_metadata_and_get_output(tune_data=tune_data_smurf, design_data=design_data, layout_data=layout_data,
                                       output_path_csv=output_path_csv, do_csv_output=do_csv_output)


def make_map_vna(tune_data_vna_output_filename='tune_data_vna.csv',
                 path_north_side_vna=None, path_south_side_vna=None, north_is_highband=None, shift_mhz=10.0,
                 design_file=designfile_default_path, waferfile=waferfile_default_path,
                 layout_position_path=mux_pos_to_mux_band_file_default_path, dark_bias_lines=None,
                 output_path_csv=None, do_csv_output=True):

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
                                       output_path_csv=output_path_csv, do_csv_output=do_csv_output)


def make_map_g3_timestream(timestream, north_is_highband=None, design_file=designfile_default_path,
                           waferfile=waferfile_default_path, layout_position_path=mux_pos_to_mux_band_file_default_path,
                           dark_bias_lines=None, output_path_csv=None, do_csv_output=True):
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
                                       output_path_csv=output_path_csv, do_csv_output=do_csv_output)


def psat_map(tunefile, north_is_highband, output_filename_smurf=None,
             cold_ramp_file=None,
             design_file=designfile_default_path, waferfile=waferfile_default_path,
             mux_pos_num_to_mux_band_num=mux_pos_to_mux_band_file_default_path,
             dark_bias_lines=None, do_csv_output=True,
             psat_temp_k=9.0, psat_show_plot=False, psat_save_plot=True):

    # # Smurf Tune File
    # read the tunefile and initialize the data instance
    tune_data_smurf = make_map_smurf(tunefile=tunefile, north_is_highband=north_is_highband,
                                     design_file=design_file, waferfile=waferfile,
                                     layout_position_path=mux_pos_num_to_mux_band_num, dark_bias_lines=dark_bias_lines,
                                     output_path_csv=output_filename_smurf, do_csv_output=do_csv_output)
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
