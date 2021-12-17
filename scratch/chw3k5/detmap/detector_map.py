# Author: Kaiwen Zheng
# Description: Functions to map smurf channels to detector UIDs
import re
import math

import numpy as np
import pandas as pd

from simple_csv import read_csv
from timer_wrap import timing


@timing
def smurf_chan_to_realized_freq(filename, band=None):
    smurf2Rfreq = pd.DataFrame({"smurf_band": {}, "smurf_chan": {}, "smurf_freq": {}})

    smurf_file = pd.read_csv(filename, header=None)
    freq = np.array(smurf_file[0])

    # Make sure band input, filename and frequency matches
    if band is not None:
        assert type(band) is int or float
        assert int(band) == math.floor((freq[0] - 4000) / 500)
    else:
        band = math.floor((freq[0] - 3998) / 500)

    searcher = re.search('_channel_assignment_b(.+).txt', filename)
    if searcher is None:
        pass
    else:
        assert band == int(filename[-5])

    # Correct for AMC frequency
    if max(freq) > 8005 or min(freq) < 3995:
        raise ValueError("Smurf frequency out of range")
    if np.mean(freq) > 6e3:
        freq = freq - 2000

    smurf2Rfreq["smurf_band"] = np.zeros(len(freq)) + band
    smurf2Rfreq["smurf_chan"] = np.array(smurf_file[2])
    smurf2Rfreq["smurf_freq"] = freq

    valid_smurf2Rfreq = smurf2Rfreq.loc[smurf2Rfreq["smurf_chan"] != -1]
    valid_smurf2Rfreq.index = np.arange(len(valid_smurf2Rfreq))

    return valid_smurf2Rfreq


@timing
def all_smurf_chan_to_realized_freq(filenames, band=None):
    """
    Returns a dataframe of smurf channel assignments
    Parameters
    ----------
    filenames : filename or an array of filenames
        Names for smurf channel assignments

    Returns
    -------
    smurf2Rfreq: DataFrame
        a table of smurf channel number and measured frequencies
        All frequency are corrected to be 4-6 GHz
    """
    smurf2Rfreq = pd.DataFrame({"smurf_band": {}, "smurf_chan": {}, "smurf_freq": {}})
    if band != None:
        assert np.array([filenames]).size == np.array([band]).size

    if np.array([filenames]).size == 1:
        smurf2Rfreq = smurf_chan_to_realized_freq(filename, band)
    else:
        for i, onefile in enumerate(filenames):

            if band != None:
                oneband = band[i]
            else:
                oneband = None

            df_oneband = smurf_chan_to_realized_freq(onefile, oneband)
            smurf2Rfreq = smurf2Rfreq.append(df_oneband, ignore_index=True)
    return smurf2Rfreq


@timing
def vna_freq_to_muxpad(vna2designfile, design_file="umux_32_map.pkl"):
    """
    Reads a map of mux chip realized-frequency-to-frequency-index, and output the 
    bondpad location of these frequencies
    ----------
    vna2designfile: string
        A csv file with header 
        | Band | Index | UFM Frequency |
        Band is the muxband which has integers between 0-13, 
        Index is the frequency order within a mux band which is integer between 0-65
        UFM Frequency is resonator frequency in MHz or Hz as measured in VNA
        This file needs to be input into the script as of April 2021

    design_file:
        filepath to the mux chip design map, for v3.2 this is can be found in 
        http://simonsobservatory.wikidot.com/local--files/muxdesigns/umux_32_map.pkl

    Returns:
    -------
    vna2pad: DataFrame
        a table of VNA frequency to mux band and mux bondpad
    """
    df_design = pd.read_pickle(design_file)

    df_design['Frequency(MHz)'] = df_design['Frequency(MHz)'] / 1e6  # A mistake in header,freq is in Hz
    if isinstance(vna2designfile, pd.DataFrame):
        df_vna = vna2designfile
    else:
        df_vna = pd.read_csv(vna2designfile)
    assert np.array(df_vna.duplicated(subset=['Band', 'Index'])).any() == False, "Duplicated!"

    freq = np.array(df_vna['UFM Frequency'])
    if ((2e9 < freq) & (freq < 8e9)).all():
        freq = freq / 1e6
        df_vna['UFM Frequency'] = freq

    assert ((2e3 < freq) & (freq < 8e3)).all()

    vna2pad = pd.DataFrame({"mux_band": {}, "pad": {}, "index": {}, "design_freq": {}, "vna_freq": {}})

    for i in np.arange(len(df_vna)):
        for j in np.arange(len(df_design)):
            if df_vna['Band'][i] == df_design['Band'][j]:
                if df_vna['Index'][i] == df_design['Freq-index'][j]:
                    vna2pad = vna2pad.append({"mux_band": df_vna['Band'][i], "pad": df_design['Pad'][j],
                                              "index": df_vna['Index'][i],
                                              "design_freq": df_design['Frequency(MHz)'][j],
                                              "vna_freq": df_vna['UFM Frequency'][i]}, ignore_index=True)
    return vna2pad


@timing
def smurf_to_mux(smurf2Rfreq, vna2pad, threshold=0.01):
    """
    Reads SmuRF information and VNA-2-bondpad information and produce a map from
    smurf channel to mux band and bondpads
    ----------
    smurf2Rfreq:
        DataFrame that includes SMuRF tuning frequency to smurf bands and channels
    vna2pad:
        DataFrane that includes mux chip frequency and bondpad information
    threshold:
        The expected difference between VNA and SmuRF found resonance frequency, in MHz.

    Returns:
    -------
    smurf2mux: DataFrame
        A table of smurf band and channel to mux band and pad
    """

    smurf2mux = pd.DataFrame({"smurf_band": {}, "smurf_chan": {}, "smurf_freq": {}, "mux_band": {}, "pad": {},
                              "index": {}, "design_freq": {}, "vna_freq": {}})

    for i, smurf_freq in enumerate(smurf2Rfreq["smurf_freq"]):
        found = False
        for j, vna_freq in enumerate(vna2pad["vna_freq"]):
            if found is False and abs(smurf_freq - vna_freq) < threshold:
                row = smurf2Rfreq[i:i + 1]
                row2 = vna2pad[j:j + 1]
                row2.index = row.index
                smurf2mux = smurf2mux.append(pd.concat([row, row2], axis=1))
                found = True
    return smurf2mux


@timing
def mux_band_to_mux_posn(smurf2mux, mux_pos_num_to_mux_band_num_path, highband='S'):
    """
    Find the wafer location index of each mux chip.
    ----------
    smurf2mux:
        A dataframe that includes information from each smurf band to mux band
    mux_pos_num_to_mux_band_num_path: str, path, required
        A path to a csv file with unique mux_pos_numbers (ints 0-27) keys mapping to mux_band_num (ints 0-13).
    highband:
        'S' or 'N', indicates which side of the wafer is connected to SMuRF band 4-7

    Returns:
    -------
    smurf2padloc: DataFrame
        A table of smurf band and channel to mux location
    """
    smurf2mux = smurf2mux.reset_index(drop=True)
    smurf2padloc = pd.concat([smurf2mux, pd.DataFrame({"mux_posn": []})], axis=1)

    # read-in for the mux_pos_num mapping to mux_band_num
    data_by_column, data_by_row = read_csv(mux_pos_num_to_mux_band_num_path)
    # the mux position numbers should be be between 0 and 27 and be a unique list of numbers
    mux_pos_nums = data_by_column['mux_pos_num']
    # get a unique list of mux_pos_nums
    mux_pos_nums_unique = set(mux_pos_nums)
    # check to make sure there are is unique list of mux_pos_nums
    if len(mux_pos_nums) != len(mux_pos_nums_unique):
        mux_pos_nums_test_set = set()
        mux_pos_nums_repeated = set()
        for mux_pos_num in mux_pos_nums:
            if mux_pos_num in mux_pos_nums_test_set:
                mux_pos_nums_repeated.add(mux_pos_num)
            mux_pos_nums_test_set.add(mux_pos_num)
        raise KeyError(f'The mux_pos_nums in the csv file {mux_pos_num_to_mux_band_num_path}\n' +
                       f'is required to be a unique set on integers,\n' +
                       f'the follow mux position numbers are repeated: {mux_pos_nums_repeated}')
    # check each mux_pos_num for compliance
    for mux_pos_num in mux_pos_nums:
        # check that the mux_pos_num is an integer
        if not isinstance(mux_pos_num, int):
            raise TypeError(f'mux_pos_num must be an int, got type: {type(mux_pos_num)}')
        # check that the mux_pos_num is an expected value between 0 and 27
        elif mux_pos_num < 0 or 27 < mux_pos_num:
            raise ValueError(f'mux_pos_num must be between 0-27 (inclusive), got mux_pos_num: {mux_pos_num}')

    # the mux_band_nums, they should be integers between 0 and 13
    mux_band_nums = data_by_column['mux_band_num']
    # check each mux_pos_num for compliance
    for mux_band_num in mux_band_nums:
        # check that the mux_pos_num is an integer
        if not isinstance(mux_band_num, int):
            raise TypeError(f'mux_band_num must be an int, got type: {type(mux_band_num)}')
        # check that the mux_pos_num is an expected value between 0 and 13
        elif mux_band_num < 0 or 13 < mux_band_num:
            raise ValueError(f'mux_band_num must be between 0-13 (inclusive), got mux_band_num: {mux_band_num}')

    # make a convenient dictionary to map mux_pos_num to mux_band_num
    mux_pos_num_to_mux_band_num = {mux_pos_num: mux_band_num for mux_pos_num, mux_band_num
                                   in zip(mux_pos_nums, mux_band_nums)}

    # start Kaiwen's original code
    for i, smurf_band in enumerate(smurf2mux["smurf_band"]):
        if highband.lower() == 'n':
            if smurf_band < 4:
                smurf_band_in_north = False
            else:
                smurf_band_in_north = True
        elif highband.lower() == 's':
            if smurf_band < 4:
                smurf_band_in_north = True
            else:
                smurf_band_in_north = False
        else:
            raise KeyError(f'The high band is not a required value [N/S], instead {highband} was received.')

        twosides = np.array(mux_pos_nums)[np.where(np.array(smurf2mux["mux_band"])[i] == np.array(mux_band_nums))]
        assert len(twosides) <= 2

        if smurf_band_in_north:
            mux_posn = [x for x in twosides if x > 13]
        else:
            mux_posn = [x for x in twosides if x <= 13]

        try:
            smurf2padloc["mux_posn"][i] = mux_posn[0]
        except:
            pass

    return smurf2padloc


@timing
def get_pad_to_wafer(filename, dark_bias_lines=None):
    """
    Extracts routing wafer to detector wafer map
    Mostly from Zach Atkin's script

    Upgraded for speed and converted to PEP-8 format by Caleb Wheeler Dec 2021
    ----------
    filename:
        Path to the detector-routing wafer map created by NIST and Princeton
    dark_bias_lines:
        Bias lines that are dark in a particular test

    Returns:
    -------
    wafer_info
        A table from mux chip position and bondpad to detector information
        In particular, freq column indicates 90ghz, 150ghz, D for dark 
        detectors which is 90ghz but has different property as optical ones,
        and NC for no-coupled resonators
    """
    if dark_bias_lines is None:
        dark_bias_lines = []
    wafer_file = pd.read_csv(filename)
    wafer_info = pd.DataFrame({"mux_posn": {}, "pad": {}, "biasline": {}, "pol": {}, "freq": {}, "det_row": {},
                               "det_col": {}, "rhomb": {}, "opt": {}, "det_x": {}, "det_y": {}})

    for index, row in wafer_file.iterrows():
        # string that search for data unique to the SQUID_PIN data column
        pad_re = 'SQ_(.+)_Ch_(.+)_\+'
        pad_str = row['SQUID_PIN']
        searcher = re.search(pad_re, pad_str)

        _, pad_str = searcher.groups()
        pad = int(pad_str)
        posn = row['Mux chip position']

        pol_str = row['DTPadlabel']
        pol_letter = pol_str[0].upper()
        if pol_letter in {'T', 'R'}:
            pol = 'A'
        elif pol_letter in {'B', 'L'}:
            pol = 'B'
        elif pol_letter == 'X':
            pol = 'D'
        else:
            raise KeyError(f'polarization character: {pol_letter} is not one of the expected types.')

        rhomb = row['DTPixelsection']

        bias_line = int(row['Bias line'])

        det_row = int(row['DTPixelrow'])
        det_col = int(row['DTPixelcolumn'])

        det_x = float(row['x']) / 1e3
        det_y = float(row['y']) / 1e3

        if (bias_line in dark_bias_lines) or (pol == 'D') or row['DTSignaldescription'] == 'NC':
            opt = False
        else:
            opt = True

        freq_re = '(.+)ghz'
        freq_str = row['DTSignaldescription']
        searcher = re.search(freq_re, freq_str)
        if searcher is None:
            if row['DTSignaldescription'] == 'NC':
                freq = 'NC'
            if pol == 'D':
                freq = 'D'
        else:
            freq_str = searcher.groups()
            freq = int(*freq_str)

        wafer_info = wafer_info.append(
            {"mux_posn": posn, "pad": pad, "biasline": bias_line, "pol": pol, "freq": freq, "det_row": det_row,
             "det_col": det_col, "rhomb": rhomb, "opt": opt, "det_x": det_x, "det_y": det_y}, ignore_index=True)
    return wafer_info


@timing
def smurf_to_detector(smurf2padloc, wafer_info):
    """
    Produces a map from smurf channel to detector information
    """

    smurf2det = pd.concat([smurf2padloc[:0], wafer_info[:0]], axis=1)
    for i, smurf_posn in enumerate(smurf2padloc["mux_posn"]):
        for j, wafer_posn in enumerate(wafer_info["mux_posn"]):
            if smurf_posn == wafer_posn and smurf2padloc["pad"][i] == wafer_info["pad"][j]:
                row = smurf2padloc[i:i + 1]
                row2 = wafer_info[j:j + 1]
                row2.index = row.index
                smurf2det = smurf2det.append(pd.concat([row, row2], axis=1))
    smurf2det = smurf2det.loc[:, ~smurf2det.columns.duplicated()]
    smurf2det = smurf2det.reindex(columns=["smurf_band", "smurf_chan", "smurf_freq", "vna_freq", "design_freq", "index",
                                           "mux_band", "pad", "mux_posn", "biasline", "pol", "freq", "det_row",
                                           "det_col",
                                           "rhomb", "opt", "det_x", "det_y"])
    return smurf2det
