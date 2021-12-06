"""
Princeton Detector Mapping.
Original Source: Kaiwen Zheng
Additional Author(s): Caleb Wheeler

Data goes in, Maps come out.
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# custom packages
from read_iv import match_chan_map, read_psat

from vna_func import assign_freq_index, get_peaks_from_vna

from detector_map import vna_freq_to_muxpad, \
    smurf_to_mux, mux_band_to_mux_posn, get_pad_to_wafer, smurf_to_detector
from simple_csv import read_csv
from channel_assignment import read_tunefile, OperateTuneData

allowed_highbands = {'N', 'S'}


def assign_channel_from_vna(south_files, north_files, highband, dict_thru, shift_mhz=10):
    highband = highband.strip().upper()
    if highband not in allowed_highbands:
        raise KeyError(f'the value for highband that was received: "{highband}"\n' +
                       f'is not of the allow values: "{allowed_highbands}"')
    # Kaiwen peak finding algorithms
    south_res_mhz = get_peaks_from_vna(south_files) / 1.0e6
    print("South Side fit completed.")
    north_res_mhz = get_peaks_from_vna(north_files) / 1.0e6
    print("North Side fit completed.")

    upper_res_tune_data = OperateTuneData()
    lower_res_tune_data = OperateTuneData()
    if highband == 'N':
        # North Side: to emulate the smurf tune file output we add 2000 MHz to the 'highband' (upper band)
        upper_res = north_res_mhz + 2000
        upper_res_is_north = True
        lower_res = south_res_mhz
    else:
        # South side: to emulate the smurf tune file output we add 2000 MHz to the 'highband' (upper band)
        upper_res = south_res_mhz + 2000
        upper_res_is_north = False
        lower_res = north_res_mhz
    # this is simple the opposite of the upper band bool value
    lower_res_is_north = not upper_res_is_north
    # put the data into bands and channels.
    upper_res_tune_data.from_peak_array(peak_array_mhz=upper_res, is_north=upper_res_is_north,
                                        is_highband=True, shift_mhz=shift_mhz, smurf_bands=None)
    lower_res_tune_data.from_peak_array(peak_array_mhz=lower_res, is_north=lower_res_is_north,
                                        is_highband=False, shift_mhz=shift_mhz, smurf_bands=None)
    return upper_res_tune_data + lower_res_tune_data


def assign_channel_use_tune(tunefile, bands=np.arange(8), dict_thru=None, highband="S"):
    if dict_thru is None:
        dict_thru = {"N": [], "S": []}
    band_array = []
    channel_array = []
    chan_assign = read_tunefile(tunefile=tunefile, return_pandas_df=True)
    chan_assign = chan_assign.loc[chan_assign['channel'] != -1]
    # print("Tune file has ",len(chan_assign)," resonators\n")
    for band in bands:
        df_band = chan_assign.loc[chan_assign['smurf_band'] == band]
        # print("Band",band," has ",len(df_band)," resonators.")
        mux_band, mux_index, miss = assign_freq_index(band, df_band["freq_mhz"], dict_thru, highband)
        band_array = np.concatenate((band_array, mux_band))
        channel_array = np.concatenate((channel_array, mux_index))
    return pd.DataFrame({"Band": band_array, "channel": channel_array, "UFM Frequency": chan_assign["frequency"]})


def automated_map(south_files, north_files, highband, shift_mhz, dict_thru, tunefile, dark_bias_lines, design_file,
                  mux_pos_num_to_mux_band_num_path, waferfile, threshold=0.1,
                  tune_data_vna_output_filename='tune_data_vna.csv', redo_vna_tune=False):
    # get the tune file data from smurf
    tune_data_smurf = OperateTuneData(path=tunefile)
    # parse/get date from a Vector Network Analyzer
    if os.path.exists(tune_data_vna_output_filename) and not redo_vna_tune:
        tune_data_vna = OperateTuneData(path=tune_data_vna_output_filename)
    else:
        if north_files == [] and south_files == []:
            raise FileNotFoundError("Both North and South VNA files were empty lists)")
        # Kaiwen's VNA peak finding function
        # peak finding on the VNA data return a data structure that relates frequency to smurf band and channel number.
        tune_data_vna = assign_channel_from_vna(south_files, north_files, highband, dict_thru, shift_mhz=shift_mhz)
        # write this data to skip this step next time and simply read in these results
        tune_data_vna.write_csv(output_path_csv=tune_data_vna_output_filename)

    chan_assign = chan_assign[["smurf_band", "channel", "freq_mhz"]]
    chan_assign = chan_assign.rename(
        columns={"smurf_band": "smurf_band", "channel": "smurf_chan", "freq_mhz": "smurf_freq"}).reset_index(drop=True)

    df_low = df_vna.loc[df_vna["UFM Frequency"] < 6e3].drop_duplicates(subset=['Band', 'Index']).reset_index()
    df_high = df_vna.loc[(df_vna["UFM Frequency"] > 6e3) & (df_vna["UFM Frequency"] < 8e3)].drop_duplicates(
        subset=['Band', 'Index']).reset_index()

    df_pad_low = vna_freq_to_muxpad(df_low, design_file)
    df_pad_high = vna_freq_to_muxpad(df_high, design_file)
    pad = pd.concat([df_pad_low, df_pad_high]).reset_index()

    smurf2mux = smurf_to_mux(chan_assign, pad, threshold)
    smurf2padloc = mux_band_to_mux_posn(smurf2mux=smurf2mux,
                                        mux_pos_num_to_mux_band_num_path=mux_pos_num_to_mux_band_num_path)
    wafer_info = get_pad_to_wafer(waferfile, dark_bias_lines=dark_bias_lines)
    smurf2det = smurf_to_detector(smurf2padloc, wafer_info)

    return smurf2det


if __name__ == '__main__':
    # get a sample configuration to use with this example
    from scratch.chw3k5.detmap.config_files.detmap_conifg_example import N_files, S_files, cold_ramp_file, \
        highband, shift, dict_thru, tunefile, dark_bias_lines, design_file, mux_pos_num_to_mux_band_num_path, \
        waferfile, output_filename

    # The main mapping file
    smurf2det = automated_map(south_files=S_files, north_files=N_files,
                              highband=highband, shift_mhz=shift, dict_thru=dict_thru,
                              tunefile=tunefile, dark_bias_lines=dark_bias_lines,
                              design_file=design_file,
                              mux_pos_num_to_mux_band_num_path=mux_pos_num_to_mux_band_num_path, waferfile=waferfile)

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
