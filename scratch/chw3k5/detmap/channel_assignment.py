"""
Author: Caleb Wheeler, written by reading code originally writen by Kaiwen Zheng

This file handles the data structures, reading, and analysis of resonator frequency to smurf band-channel index pairs.
"""
import os.path
from copy import deepcopy
from operator import attrgetter
from typing import NamedTuple, Optional

import numpy as np
import pandas as pd

from simple_csv import read_csv
from vna_to_smurf import emulate_smurf_bands


"""
For the a single Tune datum (unique smurf_band and channel)
"""


class TuneDatum(NamedTuple):
    smurf_band: int
    channel: int
    freq_mhz: float
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

    def __add__(self, other):
        """
        Two instances of this class to be combined in a binary operation. These instances ar denoted self and other.

        Returns: A new combined data instance of this class that includes the data from both input instances.
        -------

        """
        # test to see if 'self' and the 'other' instances have any copies of the same data
        tune_data_overlap = self.tune_data & other.tune_data
        # raise an error if overlapping data is found
        if tune_data_overlap != set():
            raise KeyError(f'Adding two instances of OperateTuneData requires that there is no overlapping data,\n' +
                           f'{len(tune_data_overlap)} values for tunedata, TuneDatum(s), were found to be the same, ' +
                           f'namely:\n{tune_data_overlap}')
        # spawn a new instance of this class to contain the combined data
        result = OperateTuneData()
        # combine and the 'self' and 'other' instances and combine the into the new 'result' instance
        result.tune_data = deepcopy(self.tune_data | other.tune_data)
        # do a data organization and validation
        result.tune_data_organization_and_validation()
        return result

    def __len__(self):
        if self.tune_data is None:
            raise ValueError(f'No tune data is set for this instance, so length has no meaning in this context.')
        else:
            return len(self.tune_data)

    def tune_data_organization_and_validation(self):
        """
        Populates the self.tune_data_by_band_and_channel_index instance variable. Checks for validation should be
        conducted here. This method is meant to be used internally with in this class' methods, and should not be
        required for user understanding.

        Returns
        -------
        """

        self.tune_data_by_band_and_channel_index = {}
        # loop over all the TuneDatum(s), here we sort the data to get consistent results for debugging
        for tune_datum in sorted(self.tune_data, key=attrgetter('smurf_band', 'channel', 'freq_mhz')):
            smurf_band_this_datum = tune_datum.smurf_band
            channel_this_tune_datum = tune_datum.channel
            # make a dictionary instance if this is the first resonance found in this band.
            if smurf_band_this_datum not in self.tune_data_by_band_and_channel_index.keys():
                self.tune_data_by_band_and_channel_index[smurf_band_this_datum] = {}
            # make a new set instance is this is the first resonance in this band-channel pair.
            if channel_this_tune_datum not in self.tune_data_by_band_and_channel_index[smurf_band_this_datum].keys():
                self.tune_data_by_band_and_channel_index[smurf_band_this_datum][channel_this_tune_datum] = set()
            # add the tune datum to this mapping
            self.tune_data_by_band_and_channel_index[smurf_band_this_datum][channel_this_tune_datum].add(tune_datum)

    def read_tunefile(self):
        # initialize the data the variable that stores the data we are reading.
        self.tune_data = set()
        # read the
        tunefile_data = np.load(self.path, allow_pickle=True).item()

        for smurf_band in list(tunefile_data.keys()):
            data_this_band = tunefile_data[smurf_band]
            if 'resonances' in data_this_band.keys():
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
        # do a data organization and validation
        self.tune_data_organization_and_validation()

    def from_dataframe(self, data_frame, is_north=None, is_highband=None):
        # initialize the data the variable that stores the data we are reading.
        self.tune_data = set()
        # loop over the enter data frame. !Never do this for calculations, only fo casting as done below!
        # for more info https://stackoverflow.com/questions/16476924/how-to-iterate-over-rows-in-a-dataframe-in-pandas#:~:text=DataFrame%20in%20Pandas%3F-,Answer%3A%20DON%27T*!,-Iteration%20in%20Pandas
        for index, row in data_frame.iterrows():
            # These are always required. Need to be hashable and cast as integers for faster search nad comparison.
            smurf_band = int(row['smurf_band'])
            channel = int(row['channel'])
            # set the tune datum for this row, note ** is a kwargs variable unpacking.
            tune_datum_this_res = TuneDatum(**row, is_north=is_north, is_highband=is_highband)
            self.tune_data.add(tune_datum_this_res)
        # do a data organization and validation
        self.tune_data_organization_and_validation()

    def from_peak_array(self, peak_array_mhz, is_north=None, is_highband=None,
                        shift_mhz=10, smurf_bands=None):
        self.tune_data = set()
        _real_band_bounds_mhz, _all_data_band_bounds_mhz, all_data_lower_band_bounds_mhz, \
            all_data_upper_band_bounds_mhz = emulate_smurf_bands(shift_mhz=shift_mhz, smurf_bands=smurf_bands)
        if is_highband:
            band_bounds_mhz = all_data_upper_band_bounds_mhz
        else:
            band_bounds_mhz = all_data_lower_band_bounds_mhz
        # initialize a counter used to determine channel number
        channel_count_by_band = {smurf_band: 0 for smurf_band in band_bounds_mhz}
        # this is the outer loop to be extra sure there are no repeats, it is sorted by freq_mhz for the channel count
        for res_freq_mhz in sorted(peak_array_mhz):
            for smurf_band in band_bounds_mhz.keys():
                lower_bound_mhz, upper_bound_mhz = band_bounds_mhz[smurf_band]
                if lower_bound_mhz <= res_freq_mhz < upper_bound_mhz:
                    channel = channel_count_by_band[smurf_band]
                    tune_datum_this_res = TuneDatum(freq_mhz=res_freq_mhz, smurf_band=smurf_band,
                                                    channel=channel, is_north=is_north, is_highband=is_highband)
                    # assign the tune datum to the storage variable
                    self.tune_data.add(tune_datum_this_res)
                    # iterate the channel counter
                    channel_count_by_band[smurf_band] += 1
                    # no need keep searching after we find the correct band
                    break
        # do a data organization and validation
        self.tune_data_organization_and_validation()

    def return_pandas_df(self):
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
                tune_datum_str = str(tune_datum_this_res)
                tune_datum_str_pandas_null = tune_datum_str.replace('None', 'null')
                f.write(f'{tune_datum_str_pandas_null}\n')

    def read_csv(self):
        if self.path is None:
            last_filename, _new_filename = get_filename(filename_func=operate_tune_data_csv_filename,
                                                        prefix=self.output_prefix)
            if last_filename is None:
                raise FileNotFoundError
            self.path = last_filename
        # read in the data
        data_by_column, data_by_row = read_csv(path=self.path)
        # set the data in the standard format for this class
        self.tune_data = {TuneDatum(**row_dict) for row_dict in data_by_row}
        # do a data organization and validation
        self.tune_data_organization_and_validation()


def read_tunefile(tunefile, return_pandas_df=False):
    operate_tune_data = OperateTuneData(path=tunefile)
    if return_pandas_df:
        return operate_tune_data.return_pandas_df()
    else:
        return operate_tune_data

