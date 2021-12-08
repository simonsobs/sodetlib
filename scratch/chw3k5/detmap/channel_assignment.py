"""
Author: Caleb Wheeler, written by reading code originally writen by Kaiwen Zheng

This file handles the data structures, reading, and analysis of resonator frequency to smurf band-channel index pairs.
"""
import os.path
from copy import deepcopy
from operator import attrgetter, itemgetter
from typing import NamedTuple, Optional, Union

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
    pad_num: Optional[int] = None
    mux_band: Optional[int] = None
    mux_channel: Optional[int] = None
    mux_subband: Optional[str] = None
    design_freq_mhz: Optional[float] = None

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


"""
Data Conversion
"""


def design_pickle_to_csv(design_file_path, design_filename_csv,
                         design_file_pickle_to_csv_header):
    # creat the human-readable csf file from the Pandas data frame.
    design_df = pd.read_pickle(design_file_path)
    # A mistake in header,freq is in Hz
    design_df['Frequency(MHz)'] = design_df['Frequency(MHz)'] / 1e6
    # loop over the data frame. !Never do this for calculations, only for casting as done below!
    # for more info https://stackoverflow.com/questions/16476924/how-to-iterate-over-rows-in-a-dataframe-in-pandas#:~:text=DataFrame%20in%20Pandas%3F-,Answer%3A%20DON%27T*!,-Iteration%20in%20Pandas
    design_data = sorted([row for index, row in design_df.iterrows()], key=attrgetter('Frequency(MHz)'))
    csv_file_column_names = [csv_column for pandas_column, csv_column in design_file_pickle_to_csv_header]
    csv_file_column_name_to_pandas = {csv_column: pandas_column for pandas_column, csv_column in
                                      design_file_pickle_to_csv_header}
    # tune the data into text and write it to disk
    with open(design_filename_csv, 'w') as f:
        # generate the header for the csv file
        header_str = ''
        for csv_file_column_name in csv_file_column_names:
            header_str += f'{csv_file_column_name},'
        # write the header
        f.write(f'{header_str[:-1]}\n')
        # loop over the frequency ordered data
        for data_row in design_data:
            # generate a row string each line of data
            row_str = ''
            for csv_file_column_name in csv_file_column_names:
                single_value = data_row[csv_file_column_name_to_pandas[csv_file_column_name]]
                row_str += f'{single_value},'
            #  write the row string
            f.write(f'{row_str[:-1]}\n')


class OperateTuneData:
    """
    For a measurement of tune data across a single UFM
    """

    # hard coded variables that may need to be changed in the future
    # the order of this list determines the order of csv file created from the design pickle
    design_file_pickle_to_csv_header = [('Band', 'mux_band'), ('Freq-index', 'mux_channel'),
                                        ('Frequency(MHz)', 'freq_mhz'), ('Subband', 'mux_subband'), ('Pad', 'pad_num')]
    # a default that is used when output_path_csv=None in the method self.write_csv()
    output_prefix = 'test_tune_data_vna'

    def __init__(self, tune_path=None, design_file_path=None):
        # read-in path for tune files
        self.tune_path = tune_path
        if self.tune_path is None:
            extension = None
        else:
            basename = os.path.basename(self.tune_path)
            _prefix, extension = basename.rsplit('.', 1)
            extension = extension.lower()
        # read-in path for the design files
        self.design_file_path = design_file_path

        # initial values for variables that are populated in this class's methods.
        self.tune_data = None
        self.tune_data_by_band_and_channel_index = None
        self.pandas_data_frame = None

        # auto read in known file types
        if tune_path is not None:
            if extension == 'npy':
                self.read_tunefile()
            elif extension == 'csv':
                self.read_csv()
            else:
                raise KeyError(f'File extension: "{extension}" is not recognized type.')
        if self.design_file_path is not None:
            self.read_design_file()

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
                tune_datum_this_band_and_channel = tune_data_this_band_by_index[channel]
                if channel == -1:
                    # special handling for channel == -1
                    freq_sorted_tune_data = sorted(tune_datum_this_band_and_channel, key=attrgetter('freq_mhz'))
                    for tune_datum in freq_sorted_tune_data:
                        yield tune_datum
                else:
                    # these are a required have a single TuneDatum for band-channel pairs
                    yield tune_datum_this_band_and_channel

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
            tune_dict_this_band = self.tune_data_by_band_and_channel_index[smurf_band_this_datum]
            # map the data to a dictionary structure
            if channel_this_tune_datum == -1:
                # special handling for the channel value -1
                if -1 not in tune_dict_this_band.keys():
                    self.tune_data_by_band_and_channel_index[smurf_band_this_datum][-1] = set()
                self.tune_data_by_band_and_channel_index[smurf_band_this_datum][channel_this_tune_datum].add(tune_datum)

            elif channel_this_tune_datum in tune_dict_this_band.keys():
                # make a new set instance is this is the first resonance in this band-channel pair.
                existing_tune_datum = tune_dict_this_band[channel_this_tune_datum]
                raise KeyError(f'Only Unique band-channel are allowed. ' +
                               f'For smurf_band: {smurf_band_this_datum} and channel {channel_this_tune_datum} ' +
                               f'The existing datum: {existing_tune_datum} ' +
                               f'uses has the same band-channel data as the new: {tune_datum}')
            else:
                # add the tune datum to this mapping
                self.tune_data_by_band_and_channel_index[smurf_band_this_datum][channel_this_tune_datum] = tune_datum

    def from_dataframe(self, data_frame, is_north=None, is_highband=None):
        # initialize the data the variable that stores the data we are reading.
        self.tune_data = set()
        # loop over the data frame. !Never do this for calculations, only for casting as done below!
        # for more info https://stackoverflow.com/questions/16476924/how-to-iterate-over-rows-in-a-dataframe-in-pandas#:~:text=DataFrame%20in%20Pandas%3F-,Answer%3A%20DON%27T*!,-Iteration%20in%20Pandas
        for index, row in data_frame.iterrows():
            # These are always required. Need to be hashable and cast as integers for faster search nad comparison.
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
        if self.tune_path is None:
            last_filename, _new_filename = get_filename(filename_func=operate_tune_data_csv_filename,
                                                        prefix=self.output_prefix)
            if last_filename is None:
                raise FileNotFoundError
            self.tune_path = last_filename
        # read in the data
        data_by_column, data_by_row = read_csv(path=self.tune_path)
        # set the data in the standard format for this class
        self.tune_data = {TuneDatum(**row_dict) for row_dict in data_by_row}
        # do a data organization and validation
        self.tune_data_organization_and_validation()

    def read_tunefile(self):
        # initialize the data the variable that stores the data we are reading.
        self.tune_data = set()
        # read the tune file
        tunefile_data = np.load(self.tune_path, allow_pickle=True).item()
        # loop of the bands in order
        for smurf_band in sorted(list(tunefile_data.keys())):
            data_this_band = tunefile_data[smurf_band]
            # resonances may not be present in all bands
            if 'resonances' in data_this_band.keys():
                # loop over the tune data in this band
                resonator_data_this_band = data_this_band['resonances']
                for channel_index in sorted(resonator_data_this_band.keys()):
                    single_res = resonator_data_this_band[channel_index]
                    freq_mhz = single_res['freq']
                    channel = single_res['channel']
                    subband = single_res['subband']
                    tune_datum_this_res = TuneDatum(freq_mhz=freq_mhz, smurf_band=smurf_band,
                                                    subband=subband, channel=channel)
                    self.tune_data.add(tune_datum_this_res)
        # do a data organization and validation
        self.tune_data_organization_and_validation()

    def read_design_file(self):
        if self.design_file_path is None:
            raise ValueError(f'Design file not specified, i.e. self.design_file_path is None')
        # determine if this is a pickle file from the file's extension
        prefix, extension = self.design_file_path.rsplit('.', 1)
        extension = extension.lower()
        # if this is a pickle file we will read that in first and write it to a csv file, the desired format
        if extension in {'pkl', 'pickle'}:
            # We will use the csv file if it exists, else we will creat it here
            design_filename_csv = prefix + '.csv'
            if not os.path.exists(design_filename_csv):
                design_pickle_to_csv(design_file_path=self.design_file_path, design_filename_csv=design_filename_csv,
                                     design_file_pickle_to_csv_header=self.design_file_pickle_to_csv_header)
                print(f'csv design file: {design_filename_csv}\nwritten from pickle file: {self.design_file_path}.')
            # with the csv file written or found we will now reset the self.design_file_path
            self.design_file_path = design_filename_csv
        # read the design file csv
        data_by_column, data_by_row = read_csv(path=self.design_file_path)
        # get the smurf band data to classify the design data.
        real_band_bounds_mhz, _all_data_band_bounds_mhz, _all_data_lower_band_bounds_mhz, \
            _all_data_upper_band_bounds_mhz = emulate_smurf_bands(shift_mhz=0.0, smurf_bands=None)
        # counter for the smurf channel
        # initialize a counter used to determine channel number
        channel_count_by_band = {smurf_band: 0 for smurf_band in real_band_bounds_mhz.keys()}
        # set the data in the standard format for this class
        self.tune_data = set()
        for row_dict in sorted(data_by_row, key=itemgetter('freq_mhz')):
            if row_dict['mux_band'] > 14:
                row_dict['mux_band'] =- 14
            # determine the smurf band
            for smurf_band in sorted(real_band_bounds_mhz.keys()):
                lower_bound_mhz, upper_bound_mhz = real_band_bounds_mhz[smurf_band]
                if lower_bound_mhz <= row_dict['freq_mhz'] < upper_bound_mhz:
                    row_dict['smurf_band'] = smurf_band
                    if smurf_band > 3:
                        row_dict['is_highband'] = True
                    else:
                        row_dict['is_highband'] = False
                    # set the smurf channel
                    row_dict['channel'] = channel_count_by_band[smurf_band]
                    # integrate the counter
                    channel_count_by_band[smurf_band] += 1
                    # no need to keep searching when we find what we are looking for
                    break
            else:
                # Force a smurf band to be set, this happens when the break statement was not reached
                raise KeyError(f"Design frequency {row_dict['freq_mhz']} MHz, was not found to be withing the " +
                               f"the real smurf bands, {real_band_bounds_mhz}, this should not be possible! " +
                               f"Check the design file: {self.design_file_path}")
            self.tune_data.add(TuneDatum(**row_dict))

        # do a data organization and validation
        self.tune_data_organization_and_validation()

    def map_design_data(self):
        pass


def read_tunefile(tunefile, return_pandas_df=False):
    operate_tune_data = OperateTuneData(tune_path=tunefile, design_file_path=None)
    if return_pandas_df:
        return operate_tune_data.return_pandas_df()
    else:
        return operate_tune_data

