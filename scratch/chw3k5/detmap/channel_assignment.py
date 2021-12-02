"""
Author: Caleb Wheeler, written by reading code originally writen by Kaiwen Zheng

This file handles the data structures, reading, and analysis of resonator frequency to smurf band-channel index pairs.
"""
from typing import NamedTuple
import pickle

import pandas as pd
import numpy as np


class TuneDatum(NamedTuple):
    freq_mhz: float
    smurf_band: int
    subband: int
    channel: int


class OperateTuneData:
    def __init__(self, path=None):
        self.path = path

        # initial values for variables the are populated in this class's methods.
        self.tune_data = None
        self.tune_data_by_band_and_channel_index = None
        self.pandas_data_frame = None

        if path is not None:
            self.read_tunefile()

    def read_tunefile(self):
        # initalize the data the variable that stores the data we are reading.
        self.tune_data = set()
        self.tune_data_by_band_and_channel_index = {}

        # read the
        tunefile_data = np.load(self.path, allow_pickle=True).item()

        for smurf_band in list(tunefile_data.keys()):
            data_this_band = tunefile_data[smurf_band]
            if 'resonances' in data_this_band.keys():
                # since this smurf band has resonates we will add it to the the tune data
                self.tune_data_by_band_and_channel_index[smurf_band] = {}
                # loop over the tune data in this band
                resonator_data_this_band = data_this_band['resonances']
                for channel_index in resonator_data_this_band.keys():
                    single_res = resonator_data_this_band[channel_index]
                    freq_mhz = single_res['freq']
                    channel = single_res['channel']
                    subband = single_res['subband']
                    tune_datum_this_res = TuneDatum(freq_mhz=freq_mhz, smurf_band=smurf_band,
                                                    subband=subband, channel=channel)
                    self.tune_data.add(tune_datum_this_res)
                    self.tune_data_by_band_and_channel_index[smurf_band][channel] = tune_datum_this_res

    def get_pandas_df(self):
        # make sure the tune data was load before this method was called.
        if self.tune_data_by_band_and_channel_index is None:
            raise IOError(f'No tune data has been loaded.')
        tune_datum_fields_list = TuneDatum._fields
        self.pandas_data_frame = pd.DataFrame({header_key: [] for header_key in tune_datum_fields_list})
        smurf_bands = sorted(self.tune_data_by_band_and_channel_index.keys())
        for smurf_band in smurf_bands:
            tune_data_this_band_by_index = self.tune_data_by_band_and_channel_index[smurf_band]
            channels = sorted(tune_data_this_band_by_index.keys())
            for channel in channels:
                tune_datum_this_res = tune_data_this_band_by_index[channel]
                datum_dict = {field_key: tune_datum_this_res.__getattribute__(field_key)
                              for field_key in tune_datum_fields_list}
                self.pandas_data_frame = self.pandas_data_frame.append(datum_dict, ignore_index=True)
        return self.pandas_data_frame


def read_tunefile(tunefile, return_pandas_df=False):
    operate_tune_data = OperateTuneData(path=tunefile)
    if return_pandas_df:
        return operate_tune_data.get_pandas_df()
    else:
        return operate_tune_data

