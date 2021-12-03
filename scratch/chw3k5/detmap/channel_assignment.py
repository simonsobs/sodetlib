"""
Author: Caleb Wheeler, written by reading code originally writen by Kaiwen Zheng

This file handles the data structures, reading, and analysis of resonator frequency to smurf band-channel index pairs.
"""
import os.path
from operator import attrgetter
from typing import NamedTuple, Optional

import numpy as np
import pandas as pd


"""
For the a single Tune datum (unique smurf_band and channel)
"""


class TuneDatum(NamedTuple):
    freq_mhz: float
    smurf_band: int
    channel: int
    is_north: Optional[bool] = None
    is_highband: Optional[bool] = None
    subband: Optional[int] = None

    def __str__(self):
        output_str = ''
        for column in list(self._fields):
            output_str += f'{self.__getattribute__(column)},'
        # the last comma is not needed
        final_output = output_str[:-1]
        return final_output

    def __iter__(self):
        for field_key in self._fields:
            yield self.__getattribute__(field_key)

    def dict(self):
        return {field_key: self.__getattribute__(field_key) for field_key in self._fields}

    def dict_without_none(self):
        return {field_key: self.__getattribute__(field_key) for field_key in self._fields
                if self.__getattribute__(field_key) is not None}


tune_data_column_names = list(TuneDatum._fields)
tune_data_header = f'{tune_data_column_names[0]}'
for column_name in tune_data_column_names[1:]:
    tune_data_header += f',{column_name}'


"""
Some file naming functions
"""


def filename(test_int, prefix='test', extension='csv'):
    return f'{prefix}_{test_int}.{extension}'


def operate_tune_data_csv_filename(test_int, prefix='tune_data'):
    return filename(test_int=test_int, prefix=prefix, extension='csv')


def get_filename(filename_func, **kwargs):
    test_int = 1
    while os.path.exists(filename_func(test_int, **kwargs)):
        test_int += 1
    last_int = test_int - 1
    if last_int == 0:
        last_filename = None
    else:
        last_filename = filename_func(test_int - 1, **kwargs)
    new_filename = filename_func(test_int, **kwargs)
    return last_filename, new_filename


class OperateTuneData:
    """
    For a measurement of tune data across a single UFM
    """
    def __init__(self, path=None):
        # read-in path
        self.path = path
        if self.path is None:
            extension = None
        else:
            basename = os.path.basename(self.path)
            _prefix, extension = basename.rsplit('.', 1)
            extension = extension.lower()

        # a default that is used when output_path_csv=None in the method self.write_csv()
        self.output_prefix = 'test_tune_data_vna'

        # initial values for variables the are populated in this class's methods.
        self.tune_data = None
        self.tune_data_by_band_and_channel_index = None
        self.pandas_data_frame = None

        # auto read in know file types
        if path is not None:
            if extension == 'npy':
                self.read_tunefile()
            elif extension == 'csv':
                self.read_csv()
            else:
                raise KeyError(f'File extension: "{extension}" is not recognized type.')

    def __iter__(self):
        """
        This defines the way data is outputted for this class, when certain built-in Python
        functions, like list(), are used.

        While the data is stored in unordered dictionaries and sets for quick searching and
        comparison, it is often desirable to have a constantly ordered output for analysis,
        debugging, and writing files. This is where we determine that ordering.
        """
        smurf_bands = sorted(self.tune_data_by_band_and_channel_index.keys())
        for smurf_band in smurf_bands:
            tune_data_this_band_by_index = self.tune_data_by_band_and_channel_index[smurf_band]
            channels = sorted(tune_data_this_band_by_index.keys())
            for channel in channels:
                tune_data_this_band_and_channel = tune_data_this_band_by_index[channel]
                freq_sorted_tune_data = sorted(tune_data_this_band_and_channel,
                                               key=attrgetter('freq_mhz'))
                for tune_datum in freq_sorted_tune_data:
                    yield tune_datum

    def read_tunefile(self):
        # initialize the data the variable that stores the data we are reading.
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

    def from_dataframe(self, data_frame, is_north=None, is_highband=None):
        # initialize the data the variable that stores the data we are reading.
        self.tune_data = set()
        self.tune_data_by_band_and_channel_index = {}
        # loop over the enter data frame. !Never do this for calculations, only fo casting as done below!
        # for more info https://stackoverflow.com/questions/16476924/how-to-iterate-over-rows-in-a-dataframe-in-pandas#:~:text=DataFrame%20in%20Pandas%3F-,Answer%3A%20DON%27T*!,-Iteration%20in%20Pandas
        for index, row in data_frame.iterrows():
            # These are always required. Need to be hashable and cast as integers for faster search nad comparison.
            smurf_band = int(row['smurf_band'])
            channel = int(row['channel'])
            # if this is the fist element in the dict/set, initialize the dict/set
            if smurf_band not in self.tune_data_by_band_and_channel_index.keys():
                self.tune_data_by_band_and_channel_index[smurf_band] = {}
            if channel not in self.tune_data_by_band_and_channel_index[smurf_band].keys():
                self.tune_data_by_band_and_channel_index[smurf_band][channel] = set()
            # set the tune datum for this row, note ** is a kwargs variable unpacking.
            tune_datum_this_res = TuneDatum(**row, is_north=is_north, is_highband=is_highband)
            self.tune_data.add(tune_datum_this_res)
            self.tune_data_by_band_and_channel_index[smurf_band][channel].add(tune_datum_this_res)

    def get_pandas_df(self):
        # make sure the tune data was load before this method was called.
        if self.tune_data_by_band_and_channel_index is None:
            raise IOError(f'No tune data has been loaded.')
        # initialize the pandas data frame
        self.pandas_data_frame = pd.DataFrame({header_key: [] for header_key in tune_data_column_names})
        # loop over the all the data in this class, not the behavior of list(self) is set by the __iter__ method, above
        for tune_datum_this_res in list(self):
            datum_dict = tune_datum_this_res.dict()
            self.pandas_data_frame = self.pandas_data_frame.append(datum_dict, ignore_index=True)
        return self.pandas_data_frame

    def write_csv(self, output_path_csv=None):
        # if no file name is specified make a new unique output file name
        if output_path_csv is None:
            _last_filename, new_filename = get_filename(filename_func=operate_tune_data_csv_filename,
                                                        prefix=self.output_prefix)
            output_path_csv = new_filename
        # write the data line by line
        with open(output_path_csv, 'w') as f:
            f.write(f'{tune_data_header}\n')
            # loop over the all the data in this class, not the behavior of list(self) is set by the __iter__ method
            for tune_datum_this_res in list(self):
                f.write(f'{tune_datum_this_res}\n')

    def read_csv(self):
        if self.path is None:
            last_filename, _new_filename = get_filename(filename_func=operate_tune_data_csv_filename,
                                                        prefix=self.output_prefix)
            if last_filename is None:
                raise FileNotFoundError
            self.path = last_filename
        with open(self.path, 'r') as f:
            pass



def read_tunefile(tunefile, return_pandas_df=False):
    operate_tune_data = OperateTuneData(path=tunefile)
    if return_pandas_df:
        return operate_tune_data.get_pandas_df()
    else:
        return operate_tune_data

