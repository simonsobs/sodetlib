"""
Princeton Detector Mapping.
Original Source: Kaiwen Zheng
Additional Author(s): Caleb Wheeler

Collect tune data and metadata from a variety of sources, to make useful associations, including detector maps.
"""

import os

# custom packages
from read_iv import read_psat
from vna_func import get_peaks_from_vna

from simple_csv import read_csv
from channel_assignment import OperateTuneData
from layout_data import get_layout_data


def assign_channel_from_vna(south_raw_files, north_raw_files, north_is_highband, shift_mhz=10.0):
    # Kaiwen peak finding algorithms
    south_res_mhz = get_peaks_from_vna(south_raw_files) / 1.0e6
    print("South Side fit completed.")
    north_res_mhz = get_peaks_from_vna(north_raw_files) / 1.0e6
    print("North Side fit completed.")

    upper_res_tune_data = OperateTuneData()
    lower_res_tune_data = OperateTuneData()
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
    upper_res_tune_data.from_peak_array(peak_array_mhz=upper_res, is_north=upper_res_is_north,
                                        is_highband=True, shift_mhz=shift_mhz, smurf_bands=None)
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
                 north_raw_files=None, south_raw_files=None, north_is_highband=None, shift_mhz=10.0,
                 design_data=None, layout_position_path=None, layout_data=None,
                 redo_vna_tune=False, csv_filename=None):

    # parse/get date from a Vector Network Analyzer
    if redo_vna_tune or not os.path.exists(tune_data_vna_output_filename):
        if north_raw_files is None:
            north_raw_files = []
        if south_raw_files is None:
            south_raw_files = []
        if north_raw_files == [] and south_raw_files == []:
            raise FileNotFoundError("Both North and South VNA files were empty lists)")
        # Run Kaiwen's peak finder and return a data structure that relates frequency to smurf band and channel number.
        tune_data_raw_vna = assign_channel_from_vna(south_raw_files=south_raw_files, north_raw_files=north_raw_files,
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
        tune_data_smurf.write_csv(output_path_csv=csv_filename)
    return tune_data_vna


if __name__ == '__main__':
    # get a sample configuration to use with this example
    from config_files.detmap_conifg_example import N_files, S_files, cold_ramp_file, \
        north_is_highband, vna_shift_mhz, tunefile, dark_bias_lines, design_file, mux_pos_num_to_mux_band_num_path, \
        waferfile, output_filename, output_filename_vna, tune_data_vna_output_filename, redo_vna_tune
    # # Metadata
    # get the design file for the resonators
    design_data_example = OperateTuneData(design_file_path=design_file)
    # get the UFM layout metadata (mux_layout_position and bond_pad mapping)
    layout_data_example = get_layout_data(waferfile, dark_bias_lines=dark_bias_lines, plot=False)

    # # Smurf Tune File
    # read the tunefile and initialize the data instance
    tune_data_smurf = make_map_smurf(tunefile=tunefile, north_is_highband=north_is_highband,
                                     design_data=design_data_example,
                                     layout_position_path=mux_pos_num_to_mux_band_num_path,
                                     layout_data=layout_data_example,
                                     csv_filename=output_filename)

    # # VNA scans
    tune_data_vna = make_map_vna(tune_data_vna_output_filename=tune_data_vna_output_filename,
                                 north_raw_files=N_files, south_raw_files=S_files,
                                 north_is_highband=north_is_highband, shift_mhz=vna_shift_mhz,
                                 design_data=design_data_example,
                                 layout_position_path=mux_pos_num_to_mux_band_num_path,
                                 layout_data=layout_data_example,
                                 redo_vna_tune=redo_vna_tune, csv_filename=output_filename_vna)

    # Optical power data from validation of dark bias lines.
    _cold_ramp_data_by_column, cold_ramp_data_by_row = read_csv(path=cold_ramp_file)
    coldload_ivs = [data_row for data_row in cold_ramp_data_by_row if data_row['note'].lower() == 'iv']

    # read in the sample psat data.
    _psat_by_band_chan, psat_by_temp = read_psat(coldload_ivs=coldload_ivs, make_plot=False)

    # Plot variables
    temp_k = 9.0
    show_plot = False
    save_plot = True

    # example plots
    for bandpass_target, psat_min, psat_max in [(90, 0.0, 3.0e-12),
                                                (150, 0.0, 6.0e-12)]:
        tune_data_smurf.plot_with_psat(psat_by_temp=psat_by_temp, bandpass_target=bandpass_target,
                                       temp_k=temp_k, psat_min=psat_min, psat_max=psat_max,
                                       show_plot=show_plot, save_plot=save_plot)
