"""
Princeton Detector Mapping.
Original Source: Kaiwen Zheng
Additional Author(s): Caleb Wheeler

Data goes in, Maps come out.
"""

import os
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# custom packages
from read_iv import match_chan_map, read_psat, get_psat
from peak_finder_v2 import get_peaks_v2
from vna_func import read_vna_data_array, correct_trend, read_smurf_channels
from detector_map import smurf_chan_to_realized_freq, all_smurf_chan_to_realized_freq, vna_freq_to_muxpad, \
    smurf_to_mux, mux_band_to_mux_posn, get_pad_to_wafer, smurf_to_detector

# caleb's packages
from simple_csv import read_csv
from download_example_data import sample_data_init

# Check to see if the example data is avaible, if not it downloads it from a GoogleDrive host.
sample_data_init()

# start the example script
highband = "S"
shift = 10
waferfile = os.path.join("metadata", "copper_map_corrected.csv")
design_file = os.path.join("metadata", "umux_32_map.pkl")
bands = np.arange(8)
dict_thru = {"N": [7], "S": []}
dark_bias_lines = [4, 5, 6, 7, 8, 9, 10, 11]  # If certain sides are covered
smurf_tune = os.path.join('sample_data', '1632247315_tune.npy')
band2posn = pd.DataFrame(
    {'mux_band': np.array([1, 9, 4, 2, 0, 6, 8, 10, 12, 13, 11, 5, 3, 3, 5, 11, 13, 12, 10, 8, 6, 0, 2, 4, 9, 7, 1]),
     'mux_posn': np.concatenate((np.array([0]), np.arange(2, 28)))})

dir_N = os.path.join('sample_data', 'north_vna')
dir_S = os.path.join('sample_data', 'south_vna')
north_search_str = os.path.join(dir_N, '*.S2P')
N_band = sorted(glob.glob(north_search_str))

south_search_str = os.path.join(dir_S, '*.S2P')
S_band = sorted(glob.glob(south_search_str))

# hard cord sorting needs to fixed
N_files = N_band[3:-1]
S_files = S_band[2:]


def get_peaks_from_vna(vna_files):
    f, r, i = read_vna_data_array(vna_files)
    s21_corrected = correct_trend(f, r, i, avg_over=1000)
    f0s, resonance_s21, low, high = get_peaks_v2(f, s21_corrected, f_delta=1e5, det_num=1000, baseline=0.1)
    return f0s


def assign_freq_index(band, freqlist, dict_thru, highband):
    if ((highband == "S") & (band > 3)) | ((highband == "N") & (band <= 3)):
        missing_chip = dict_thru["S"]
    else:
        missing_chip = dict_thru["N"]
    counts = len(freqlist)
    mux_band = np.zeros(counts)
    mux_index = np.zeros(counts)
    init_band = np.floor((band % 4) * 3.5)
    miss = (3.5 * 66) - counts
    offset = 0
    if band % 2 == 1:
        offset = 33
    for i in np.arange(4):
        if (init_band + i) in missing_chip:
            if missing_chip in [3, 10]:
                miss -= 33
                offset = 33
            else:
                miss -= 66
                offset = 66
            continue
        start = max(0, 66 * i - offset)
        end = min(counts, (i + 1) * 66 - offset)
        mux_band[start:end] += i + init_band
        mux_index[start:end] = np.arange(end - start)
    print("Band %i misses %i resonators (%.2f percent)\n"
          % (band, miss, 100 * miss / (miss + len(freqlist))))
    return mux_band, mux_index, miss


def assign_index_use_vna(S_files, N_files, highband, dict_thru, shift=10):
    assert ((highband == "N") | (highband == "S"))
    south_res = get_peaks_from_vna(S_files)
    print("South Side fit completed.")
    north_res = get_peaks_from_vna(N_files)
    print("North Side fit completed.")
    south_res = south_res / 1e6 + 2000 * (highband == "S")
    north_res = north_res / 1e6 + 2000 * (highband == "N")
    band_array = []
    index_array = []
    freq_array = []
    for band in np.arange(8):
        if ((highband == "S") & (band > 3)) | ((highband == "N") & (band <= 3)):
            bandfreq = south_res[(south_res > band * 500 + 4000 + shift) & (south_res < band * 500 + 4500 + shift)]
        else:
            bandfreq = north_res[(north_res > band * 500 + 4000 + shift) & (north_res < band * 500 + 4500 + shift)]
        print("Band", band, " has ", len(bandfreq), " resonators.")
        mux_band, mux_index, miss = assign_freq_index(band, bandfreq, dict_thru, highband)
        band_array = np.concatenate((band_array, mux_band))
        index_array = np.concatenate((index_array, mux_index))
        freq_array = np.concatenate((freq_array, bandfreq))
    return pd.DataFrame({"Band": band_array, "Index": index_array, "UFM Frequency": freq_array})


def assign_index_use_tune(smurf_tune, bands=np.arange(8), dict_thru=None, highband="S"):
    if dict_thru is None:
        dict_thru = {"N": [], "S": []}
    band_array = []
    index_array = []
    chan_assign = read_smurf_channels(smurf_tune)
    chan_assign = chan_assign.loc[chan_assign['channel'] != -1]
    # print("Tune file has ",len(chan_assign)," resonators\n")
    for band in bands:
        df_band = chan_assign.loc[chan_assign['band'] == band]
        # print("Band",band," has ",len(df_band)," resonators.")
        mux_band, mux_index, miss = assign_freq_index(band, df_band["frequency"], dict_thru, highband)
        band_array = np.concatenate((band_array, mux_band))
        index_array = np.concatenate((index_array, mux_index))
    return pd.DataFrame({"Band": band_array, "Index": index_array, "UFM Frequency": chan_assign["frequency"]})


def automated_map(S_files, N_files, highband, shift, dict_thru, smurf_tune, dark_bias_lines, design_file, band2posn,
                  waferfile, threshold=0.1):
    chan_assign = read_smurf_channels(smurf_tune)

    if N_files == [] and S_files == []:
        raise FileNotFoundError("Both North and South VNA files were empty lists)")
    else:
        df_vna = assign_index_use_vna(S_files, N_files, highband, dict_thru, shift=10)

    chan_assign = chan_assign[["band", "channel", "frequency"]]
    chan_assign = chan_assign.rename(
        columns={"band": "smurf_band", "channel": "smurf_chan", "frequency": "smurf_freq"}).reset_index(drop=True)

    df_low = df_vna.loc[df_vna["UFM Frequency"] < 6e3].drop_duplicates(subset=['Band', 'Index']).reset_index()
    df_high = df_vna.loc[(df_vna["UFM Frequency"] > 6e3) & (df_vna["UFM Frequency"] < 8e3)].drop_duplicates(
        subset=['Band', 'Index']).reset_index()

    df_pad_low = vna_freq_to_muxpad(df_low, design_file)
    df_pad_high = vna_freq_to_muxpad(df_high, design_file)
    pad = pd.concat([df_pad_low, df_pad_high]).reset_index()

    smurf2mux = smurf_to_mux(chan_assign, pad, threshold)
    smurf2padloc = mux_band_to_mux_posn(smurf2mux, band2posn)
    wafer_info = get_pad_to_wafer(waferfile, dark_bias_lines=dark_bias_lines)
    smurf2det = smurf_to_detector(smurf2padloc, wafer_info)

    return smurf2det


smurf2det = automated_map(S_files, N_files, highband, shift, dict_thru, smurf_tune, dark_bias_lines, design_file,
                          band2posn, waferfile)


output_filename = "test_pixel_info.csv"
smurf2det.to_csv(output_filename, index=False)

cold_ramp_file = os.path.join('sample_data', 'coldloadramp_example.csv')
data_by_column, data_by_row = read_csv(path=os.path.join('sample_data', 'coldloadramp_example.csv'))

coldload_ivs = [data_row for data_row in data_by_row if data_row['note'].lower() == 'iv']

# coldload = pd.read_csv(cold_ramp_file, header=None)
# coldload.columns = ["bath_temp", "bias_voltage", "bias_line", "band", "data_path", "note"]
# coldload = coldload.loc[coldload["note"] == "IV"]
# coldload.reset_index(drop=True)
# coldload = coldload.rename(columns={'bath_temp': 'cold_load_temp'})
# coldload = coldload[['cold_load_temp', 'bias_line', 'band', 'data_path']]


psat_data = read_psat(coldload_ivs=coldload_ivs, map_data=smurf2det, make_plot=True)

pixel_info = match_chan_map(output_filename, psat_data)

T = 9.0
mi = 0
ma = 3e-12

for key in pixel_info.keys():
    if pixel_info[key]['det'][0]['freq'] == '90':
        try:
            plt.scatter(pixel_info[key]['det'][0]['det_x'], pixel_info[key]['det'][0]['det_y'],
                        c=pixel_info[key]['psat'][0], vmin=mi, vmax=ma)
        except:
            pass
plt.title("90 GHz Psat at 100mK CL=9K, range=0-3 pW")
plt.show()

T = 9.0
mi = 0
ma = 6e-12
for key in pixel_info.keys():
    if pixel_info[key]['det'][0]['freq'] == '150':
        try:
            plt.scatter(pixel_info[key]['det'][0]['det_x'], pixel_info[key]['det'][0]['det_y'],
                        c=pixel_info[key]['psat'][0], vmin=mi, vmax=ma)
        except:
            pass
plt.title("150 GHz Psat at 100mK CL=9K, range=0-6 pW")
plt.show()
