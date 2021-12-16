"""
Princeton Detector Mapping.
Original Source: Kaiwen Zheng
Additional Author(s): Caleb Wheeler

Data goes in, Maps come out.
"""

import os

import matplotlib.pyplot as plt

# custom packages
from read_iv import match_chan_map, read_psat

from vna_func import get_peaks_from_vna

from detector_map import get_pad_to_wafer, smurf_to_detector
from simple_csv import read_csv
from channel_assignment import OperateTuneData

allowed_highbands = {'N', 'S'}


def assign_channel_from_vna(south_files, north_files, north_is_highband, shift_mhz=10):
    # Kaiwen peak finding algorithms
    south_res_mhz = get_peaks_from_vna(south_files) / 1.0e6
    print("South Side fit completed.")
    north_res_mhz = get_peaks_from_vna(north_files) / 1.0e6
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


def automated_map(south_files, north_files, north_is_highband, shift_mhz, tunefile, dark_bias_lines, design_file,
                  mux_pos_num_to_mux_band_num_path, waferfile, design_threshold_mhz=0.1,
                  tune_data_vna_output_filename='tune_data_vna.csv', redo_vna_tune=False):
    # get the tune file data from smurf
    tune_data_smurf = OperateTuneData(tune_path=tunefile, north_is_highband=north_is_highband)

    # parse/get date from a Vector Network Analyzer
    if os.path.exists(tune_data_vna_output_filename) and not redo_vna_tune:
        tune_data_vna = OperateTuneData(tune_path=tune_data_vna_output_filename)
    else:
        if north_files == [] and south_files == []:
            raise FileNotFoundError("Both North and South VNA files were empty lists)")
        # Run Kaiwen's peak finder and return a data structure that relates frequency to smurf band and channel number.
        tune_data_vna = assign_channel_from_vna(south_files=south_files, north_files=north_files,
                                                north_is_highband=north_is_highband, shift_mhz=shift_mhz)
        # write this data to skip this step next time and simply read in these results
        tune_data_vna.write_csv(output_path_csv=tune_data_vna_output_filename)

    design_data = OperateTuneData(design_file_path=design_file, layout_position_path=mux_pos_num_to_mux_band_num_path)
    # update the tune_data collections to include design data.
    tune_data_smurf.map_design_data(design_data=design_data)
    tune_data_vna.map_design_data(design_data=design_data)

    wafer_info = get_pad_to_wafer(waferfile, dark_bias_lines=dark_bias_lines)

    # Not refactored below
    smurf2det = smurf_to_detector(smurf2padloc, wafer_info)

    return tune_data_smurf, tune_data_vna


if __name__ == '__main__':
    # get a sample configuration to use with this example
    from scratch.chw3k5.detmap.config_files.detmap_conifg_example import N_files, S_files, cold_ramp_file, \
        north_is_highband, shift, tunefile, dark_bias_lines, design_file, mux_pos_num_to_mux_band_num_path, \
        waferfile, output_filename, tune_data_vna_output_filename, redo_vna_tune

    # The main mapping file
    tune_data_smurf, tune_data_vna = automated_map(south_files=S_files, north_files=N_files,
                                                   north_is_highband=north_is_highband, shift_mhz=shift,
                                                   tunefile=tunefile, dark_bias_lines=dark_bias_lines,
                                                   design_file=design_file,
                                                   mux_pos_num_to_mux_band_num_path=mux_pos_num_to_mux_band_num_path,
                                                   waferfile=waferfile,
                                                   tune_data_vna_output_filename=tune_data_vna_output_filename,
                                                   redo_vna_tune=redo_vna_tune)
    # not refactored below
    smurf2det.to_csv(output_filename, index=False)

    data_by_column, data_by_row = read_csv(path=cold_ramp_file)

    coldload_ivs = [data_row for data_row in data_by_row if data_row['note'].lower() == 'iv']

    psat_data = read_psat(coldload_ivs=coldload_ivs, map_data=smurf2det, make_plot=True)

    pixel_info = match_chan_map(output_filename, psat_data)

    T = 9.0
    mi = 0
    ma = 3e-12

    for key in pixel_info.keys():
        if pixel_info[key]['det'][0]['freq'] == '90':
            plt.scatter(pixel_info[key]['det'][0]['det_x'], pixel_info[key]['det'][0]['det_y'],
                        c=pixel_info[key]['psat'][0], vmin=mi, vmax=ma)

    plt.title("90 GHz Psat at 100mK CL=9K, range=0-3 pW")
    plt.show()

    T = 9.0
    mi = 0
    ma = 6e-12
    for key in pixel_info.keys():
        if pixel_info[key]['det'][0]['freq'] == '150':
            plt.scatter(pixel_info[key]['det'][0]['det_x'], pixel_info[key]['det'][0]['det_y'],
                        c=pixel_info[key]['psat'][0], vmin=mi, vmax=ma)

    plt.title("150 GHz Psat at 100mK CL=9K, range=0-6 pW")
    plt.show()
