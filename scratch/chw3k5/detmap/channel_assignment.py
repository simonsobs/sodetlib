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
import matplotlib as mpl
mpl.use(backend='TkAgg')
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import matplotlib.colors as colors
import matplotlib.cm as cm

from simple_csv import read_csv
from vna_to_smurf import emulate_smurf_bands


"""
For the a single Tune datum (unique smurf_band and channel)
"""


class TuneDatum(NamedTuple):
    smurf_band: int
    channel_index: int
    freq_mhz: float
    is_north: Optional[bool] = None
    is_highband: Optional[bool] = None
    smurf_channel: Optional[int] = None
    smurf_subband: Optional[int] = None
    bond_pad: Optional[int] = None
    mux_band: Optional[int] = None
    mux_channel: Optional[int] = None
    mux_subband: Optional[str] = None
    mux_layout_position: Optional[int] = None
    design_freq_mhz: Optional[float] = None
    bias_line: Optional[int] = None
    pol: Optional[str] = None
    freq_obs_ghz: Optional[Union[int, str]] = None
    det_row: Optional[int] = None
    det_col: Optional[int] = None
    rhomb: Optional[str] = None
    is_optical: Optional[bool] = None
    det_x: Optional[float] = None
    det_y: Optional[float] = None

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
                                        ('Frequency(MHz)', 'freq_mhz'), ('Subband', 'mux_subband'), ('Pad', 'bond_pad')]
    # a default that is used when output_path_csv=None in the method self.write_csv()
    output_prefix = 'test_tune_data_vna'

    # the attributes of design data that are mapped into TuneDatum, not freq_mhz is handled differently
    design_attributes = {'bond_pad', 'mux_band', 'mux_channel', 'mux_subband', 'mux_layout_position'}
    # the attributes of layout data that are mapper into TuneDatum
    layout_attributes = {"bias_line", "pol", "freq_obs_ghz", "det_row", "det_col", "rhomb", "is_optical",
                         "det_x", "det_y"}

    # interation order for values that are allowed for TuneDatum.is_north
    is_north_iter_order = [True, False, None]

    # the directory for output plots
    plot_dir = os.path.join('plots')

    def __init__(self, tune_path=None, design_file_path=None, layout_position_path=None, north_is_highband=None):
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
        # read-in path for the layout position csv file, how the mux wafers are layout on a UFM
        self.layout_position_path = layout_position_path
        # True is the north side the high band, False if not, None if not applicable or unknown
        self.north_is_highband = north_is_highband

        # initial values for variables that are populated in this class's methods.
        self.is_smurf = False
        self.tune_data = None
        self.tune_data_side_band_channel = None
        self.tune_data_smurf_band_and_channel = None
        self.pandas_data_frame = None
        self.tune_data_with_design_data = None
        self.tune_data_without_design_data = None
        self.tune_data_with_layout_data = None
        self.tune_data_without_layout_data = None

        # make the plot directory if it does not exist
        if not os.path.exists(self.plot_dir):
            os.mkdir(self.plot_dir)

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

        for is_north in sorted(self.tune_data_side_band_channel.keys(), key=self.is_north_rank_key):
            tune_data_this_side = self.tune_data_side_band_channel[is_north]
            smurf_bands = sorted(tune_data_this_side.keys())
            for smurf_band in smurf_bands:
                tune_data_this_band = tune_data_this_side[smurf_band]
                channels_indexes = sorted(tune_data_this_band.keys())
                for channel_index in channels_indexes:
                    tune_datum_this_band_and_channel = tune_data_this_band[channel_index]
                    if channel_index == -1:
                        # special handling for channel == -1
                        freq_sorted_tune_data = sorted(tune_datum_this_band_and_channel, key=attrgetter('freq_mhz'))
                        for tune_datum in freq_sorted_tune_data:
                            yield tune_datum
                    else:
                        # these are a required have a single TuneDatum for side-band-channel combinations
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
        result.tune_data = deepcopy(self.tune_data.union(other.tune_data))
        # do a data organization and validation
        result.tune_data_organization_and_validation()
        return result

    def __len__(self):
        if self.tune_data is None:
            raise ValueError(f'No tune data is set for this instance, so length has no meaning in this context.')
        else:
            return len(self.tune_data)

    def is_north_rank_key(self, is_north):
        for count, is_north_state in list(enumerate(self.is_north_iter_order)):
            if is_north == is_north_state:
                return count
        else:
            return float('inf')

    def tune_data_organization_and_validation(self):
        """
        Populates the self.tune_data_side_band_channel instance variable. Checks for validation should be
         conducted here. This method is meant to be used internally with in this class' methods, and should not be
         required for user understanding.

        Also, this method does a data validation, raising exceptions for unexpected data sets before the data is
         probated to analysis, matching, where it could cause unexpected results.


        Returns: self.tune_data_side_band_channel a three level dictionary tree that uses a unique set of keys to lead
                 to a single TuneDatum. These keys are is_north, smurf_band, channel (smurf_channel). The allowed
                 values for these keys are: 1) is_north {True, False, None}
                                            2) smurf_band ints {-1, 0, 1, 2, 3, 4, 5, 6, 7}
                                            3) channels {-1, 0, 1, 2, ... , ~280}.

                There is special handling channel -1, with is denotes a found but untracked/unused resonator
                by the SMuRF. First, when channel == -1, then smurf_band is always also -1. Many unused resonatotors
                given this designation, instead of these keys leading to a unique TuneDatum, they lead to a set() of
                TuneDatum with smurf_band == -1 and channel == -1.
        -------
        """

        self.tune_data_side_band_channel = {}
        # loop over all the TuneDatum(s), here we sort the data to get consistent results for debugging
        for tune_datum in sorted(self.tune_data, key=attrgetter('smurf_band', 'channel_index', 'freq_mhz')):
            smurf_band_this_datum = tune_datum.smurf_band
            channel_index = tune_datum.channel_index
            smurf_channel = tune_datum.smurf_channel
            is_north = tune_datum.is_north
            # make a new dictionary instance if this is the first resonance found for this side.
            if is_north not in self.tune_data_side_band_channel.keys():
                self.tune_data_side_band_channel[is_north] = {}
            # make a new dictionary instance if this is the first resonance found in this band.
            if smurf_band_this_datum not in self.tune_data_side_band_channel[is_north].keys():
                self.tune_data_side_band_channel[is_north][smurf_band_this_datum] = {}
            # map the data to a dictionary structure
            if smurf_channel == -1:
                # special handling for the smurf_channel value -1
                if -1 not in self.tune_data_side_band_channel[is_north][smurf_band_this_datum].keys():
                    self.tune_data_side_band_channel[is_north][smurf_band_this_datum][-1] = set()
                self.tune_data_side_band_channel[is_north][smurf_band_this_datum][smurf_channel].add(tune_datum)
            elif channel_index in self.tune_data_side_band_channel[is_north][smurf_band_this_datum].keys():
                # This is happens if there is already TuneDatum for this combination of side-smurf_band-channel
                existing_tune_datum = self.tune_data_side_band_channel[is_north][smurf_band_this_datum][channel_index]
                raise KeyError(f'Only Unique side-band-channel combinations are allowed. ' +
                               f'For side: {is_north} smurf_band: {smurf_band_this_datum} ' +
                               f'and channel: {channel_index} ' +
                               f'The existing datum: {existing_tune_datum} ' +
                               f'uses has the same band-channel data as the new: {tune_datum}')
            else:
                # add the tune datum to this mapping
                self.tune_data_side_band_channel[is_north][smurf_band_this_datum][channel_index] = tune_datum
        if self.is_smurf:
            # do a second mapping by smurf channel, used for mapping p-sat data, also from smurf
            self.tune_data_smurf_band_and_channel = {}
            for tune_datum in list(self):
                smurf_band = tune_datum.smurf_band
                smurf_channel = tune_datum.smurf_channel
                if smurf_band not in self.tune_data_smurf_band_and_channel.keys():
                    self.tune_data_smurf_band_and_channel[smurf_band] = {}
                if smurf_channel == -1:
                    if smurf_channel not in self.tune_data_smurf_band_and_channel[smurf_band].keys():
                        self.tune_data_smurf_band_and_channel[smurf_band][smurf_channel] = set()
                    self.tune_data_smurf_band_and_channel[smurf_band][smurf_channel].add(tune_datum)
                else:
                    if smurf_channel in self.tune_data_smurf_band_and_channel[smurf_band].keys():
                        # This is happens if there is already TuneDatum for this combination of side-smurf_band-channel
                        existing_tune_datum = self.tune_data_smurf_band_and_channel[smurf_band][smurf_channel]
                        raise KeyError(f'Only Unique side-band-channel combinations are allowed. ' +
                                       f'For smurf_band: {smurf_band} ' +
                                       f'and channel: {smurf_channel} ' +
                                       f'The existing datum: {existing_tune_datum} ' +
                                       f'uses has the same band-channel data as the new: {tune_datum}')
                    else:
                        self.tune_data_smurf_band_and_channel[smurf_band][smurf_channel] = tune_datum

    def from_dataframe(self, data_frame, is_highband=None, is_north=None):
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
            _all_data_upper_band_bounds_mhz = emulate_smurf_bands(shift_mhz=shift_mhz, smurf_bands=smurf_bands)
        # Highband is an artificial designation, only low band resonators were ever fabricated in practice.
        band_bounds_mhz = all_data_lower_band_bounds_mhz
        # initialize a counter used to determine channel number
        channel_count_by_band = {smurf_band: 0 for smurf_band in band_bounds_mhz}
        # this is the outer loop to be extra sure there are no repeats, it is sorted by freq_mhz for the channel count
        for res_freq_mhz in sorted(peak_array_mhz):
            for smurf_band in band_bounds_mhz.keys():
                lower_bound_mhz, upper_bound_mhz = band_bounds_mhz[smurf_band]
                if lower_bound_mhz <= res_freq_mhz < upper_bound_mhz:
                    channel_index = channel_count_by_band[smurf_band]
                    # this is how the SMuRF reports the Highband designation
                    if is_highband:
                        smurf_band_emulated = smurf_band + 4
                    else:
                        smurf_band_emulated = smurf_band
                    # record the tune data
                    tune_datum_this_res = TuneDatum(freq_mhz=res_freq_mhz, smurf_band=smurf_band_emulated,
                                                    channel_index=channel_index, is_north=is_north, is_highband=is_highband)
                    # assign the tune datum to the storage variable
                    self.tune_data.add(tune_datum_this_res)
                    # iterate the channel counter
                    channel_count_by_band[smurf_band] += 1
                    # no need keep searching after we find the correct band
                    break
        # do a data organization and validation
        self.tune_data_organization_and_validation()

    @staticmethod
    def from_tune_datums(tune_data, north_is_highband):
        new_operate_tune_data = OperateTuneData(north_is_highband=north_is_highband)
        new_operate_tune_data.tune_data = tune_data
        new_operate_tune_data.tune_data_organization_and_validation()
        return new_operate_tune_data

    def return_pandas_df(self):
        # make sure the tune data was load before this method was called.
        if self.tune_data_side_band_channel is None:
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
        self.is_smurf = True
        # initialize the data the variable that stores the data we are reading.
        self.tune_data = set()
        # read the tune file
        tunefile_data = np.load(self.tune_path, allow_pickle=True).item()
        # loop of the bands in order
        for smurf_band in sorted(list(tunefile_data.keys())):
            if -1 < smurf_band < 4:
                # This is the low-band, smurf low-band is bands 0, 1, 2, 3
                is_highband = False
                if self.north_is_highband is None:
                    is_north = None
                elif self.north_is_highband:
                    # this is False since this is the low-band and the north side is the highband
                    is_north = False
                else:
                    # this is True since this is the low-band and the north side is *not* the highband
                    is_north = True
            elif 3 < smurf_band < 8:
                # This is the high-band, smurf high-band is bands 4, 5, 6, 7
                is_highband = True
                if self.north_is_highband is None:
                    is_north = None
                elif self.north_is_highband:
                    # this is True since this is the high-band and the north side is the highband
                    is_north = True
                else:
                    # this is False since this is the high-band and the north side is *not* the highband
                    is_north = False
            else:
                is_highband = None
                is_north = None
            data_this_band = tunefile_data[smurf_band]
            # resonances may not be present in all bands
            if 'resonances' in data_this_band.keys():
                # loop over the tune data in this band
                resonator_data_this_band = data_this_band['resonances']
                available_channel_indexes = sorted(resonator_data_this_band.keys())
                for channel_index in available_channel_indexes:
                    single_res = resonator_data_this_band[channel_index]
                    if is_highband:
                        # the real frequency is 2000.0 less than what is reported for high band data
                        freq_mhz = single_res['freq'] - 2000.0
                    else:
                        freq_mhz = single_res['freq']
                    smurf_channel = single_res['channel']
                    smurf_subband = single_res['subband']
                    tune_datum_this_res = TuneDatum(freq_mhz=freq_mhz, smurf_band=smurf_band, channel_index=channel_index,
                                                    smurf_channel=smurf_channel, smurf_subband=smurf_subband,
                                                    is_north=is_north, is_highband=is_highband)
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
        # counter for the smurf channel, initialize a counter used to determine channel number
        channel_count_by_band = {smurf_band: 0 for smurf_band in real_band_bounds_mhz.keys()}
        # set the data in the standard format for this class
        self.tune_data = set()
        for row_dict in sorted(data_by_row, key=itemgetter('freq_mhz')):
            # determine the smurf band
            for smurf_band in sorted(real_band_bounds_mhz.keys()):
                lower_bound_mhz, upper_bound_mhz = real_band_bounds_mhz[smurf_band]
                if lower_bound_mhz <= row_dict['freq_mhz'] < upper_bound_mhz:
                    row_dict['smurf_band'] = smurf_band
                    # the design data file is outdated, bands 4-7 are is a repeat of 0-3, there is no highband design
                    if smurf_band < 4:
                        # this would happen automatically,
                        # but we are explicitly stating here "that is_highband has no meaning for design data"
                        row_dict['is_highband'] = None
                        # set the smurf channel
                        row_dict['channel_index'] = channel_count_by_band[smurf_band]
                        # iterate the counter
                        channel_count_by_band[smurf_band] += 1
                        # add this datum
                        self.tune_data.add(TuneDatum(**row_dict))
                        # no need to keep searching when we find what we are looking for
                        break
                    elif 3 < smurf_band:
                        # ignore the highband design, those resonators were never fabricated
                        break
            else:
                # Force a smurf band to be set, this happens when the break statement was not reached
                raise KeyError(f"Design frequency {row_dict['freq_mhz']} MHz, was not found to be withing the " +
                               f"the real smurf bands, {real_band_bounds_mhz}, this should not be possible! " +
                               f"Check the design file: {self.design_file_path}")

        # the highband design resonators are simply a copy of the low band design, now we do that in software
        for tune_datum_low_band in list(self.tune_data):
            tune_datum_high_band_copy = tune_datum_low_band.dict()
            tune_datum_high_band_copy['smurf_band'] = tune_datum_high_band_copy['smurf_band'] + 4
            # This would happen automatically,
            # but we are explicitly stating here "that is_highband has no meaning for design data"
            tune_datum_high_band_copy['is_highband'] = None
            self.tune_data.add(TuneDatum(**tune_datum_high_band_copy))

        # do a data organization and validation
        self.tune_data_organization_and_validation()

    def map_design_data(self, design_data, layout_position_path=None):
        # the layout_position_path can be set in this method or the __init__() method
        if layout_position_path is not None:
            self.layout_position_path = layout_position_path
        # get positional layout data for the mux chips if it is available
        if self.layout_position_path is not None:
            mux_layout_position_by_column, _mux_layout_position_by_row = read_csv(path=self.layout_position_path)
            # initialize the mux_band_to_mux_pos_dict, True and False keys denote the is_north
            mux_band_to_mux_pos_dict = {True: {}, False: {}}
            for mux_band, mux_pos, is_north in zip(mux_layout_position_by_column['mux_band_num'],
                                                   mux_layout_position_by_column['mux_pos_num'],
                                                   mux_layout_position_by_column['is_north']):
                mux_band_to_mux_pos_dict[is_north][mux_band] = mux_pos
        else:
            mux_band_to_mux_pos_dict = None
        # make a new set to hold the tune data that is updated with design data
        tune_data_new = set()
        # track TuneDatums that are mapped to a design record
        tune_data_with_design_data = set()
        # track TuneDatums that are *not* mapped to a design record
        tune_data_without_design_data = set()
        # is_highband has no meaning for design data
        design_tune_data_by_band_channel = design_data.tune_data_side_band_channel[None]
        # loop overall the data
        for tune_datum in list(self):
            # pull these properties out to access the design data
            is_north = tune_datum.is_north
            is_highband = tune_datum.is_highband
            smurf_band = tune_datum.smurf_band
            channel_index = tune_datum.channel_index
            # the high-band resonators were never fabricated, highband is really a positional designation for SMuRF
            if is_highband:
                design_band = smurf_band - 4
            else:
                design_band = smurf_band
            # see if there is design data available for this band-channel pair
            if design_band in design_tune_data_by_band_channel.keys() and \
                    channel_index in design_tune_data_by_band_channel[design_band].keys():
                design_datum = design_tune_data_by_band_channel[design_band][channel_index]
                # we can extract specific parameters from the design_datum and build a dictionary
                design_dict = {design_key: design_datum.__getattribute__(design_key)
                               for design_key in self.design_attributes}
                # set the mux_layout_position is available
                if mux_band_to_mux_pos_dict is not None:
                    mux_band = design_dict['mux_band']
                    if mux_band in mux_band_to_mux_pos_dict[is_north].keys():
                        design_dict['mux_layout_position'] = mux_band_to_mux_pos_dict[is_north][mux_band]
                # move the design frequency to the appropriate attribute
                design_dict['design_freq_mhz'] = design_datum.freq_mhz
                # get a mutable dictionary for the tune datum
                tune_dict = tune_datum.dict()
                # update the tune dict with the design parameters
                tune_dict.update(design_dict)
                # reassign these key-value pairs to the immutable TuneDatum
                tune_datum_with_design_data = TuneDatum(**tune_dict)
                # add this to the new data set
                tune_data_new.add(tune_datum_with_design_data)
                # track which datums have designs info
                tune_data_with_design_data.add(tune_datum_with_design_data)
            else:
                # if there is no available design data then we pass the unchanged tune_datum to the new set
                tune_data_new.add(tune_datum)
                # we also track what datums were not able to matched with design data.
                tune_data_without_design_data.add(tune_datum)
        # if everything was successful, assign the combined mapping to this instances self.tune_data
        self.tune_data = tune_data_new
        # do a data organization and validation
        self.tune_data_organization_and_validation()
        # make new instances of this class with the design found and the design not found data
        self.tune_data_with_design_data = self.from_tune_datums(tune_data=tune_data_with_design_data,
                                                                north_is_highband=self.north_is_highband)
        self.tune_data_without_design_data = self.from_tune_datums(tune_data=tune_data_without_design_data,
                                                                   north_is_highband=self.north_is_highband)

    def map_layout_data(self, layout_data):
        # make a new set to hold the tune data that is updated with design data
        tune_data_new = set()
        # track TuneDatums that are mapped to a design record
        tune_data_with_layout_data = set()
        # track TuneDatums that are *not* mapped to a design record
        tune_data_without_layout_data = set()
        for tune_datum in list(self):
            # get the primary key for layout data
            mux_layout_position = tune_datum.mux_layout_position
            # get the secondary key for layout data
            bond_pad = tune_datum.bond_pad
            # check for layout data with this key pair, the key values may be None
            if mux_layout_position in layout_data.keys() and bond_pad in layout_data[mux_layout_position].keys():
                # the layout information for this tune_datum
                layout_datum = layout_data[mux_layout_position][bond_pad]
                # get a mutable version of the tune_datum to update
                tune_datum_updated = tune_datum.dict()
                # update the mutable dictionary with the requested layout data specified in this class
                tune_datum_updated.update({layout_key: layout_datum[layout_key]
                                           for layout_key in self.layout_attributes})
                # map this to a new tune_datum, which does a data type validation
                tune_datum_with_layout = TuneDatum(**tune_datum_updated)
                # save the results
                tune_data_new.add(tune_datum_with_layout)
                tune_data_with_layout_data.add(tune_datum_with_layout)
            else:
                tune_data_new.add(tune_datum)
                # track what does not get layout data
                tune_data_without_layout_data.add(tune_datum)
        # reset this instances tune_data
        self.tune_data = tune_data_new
        # do a validation on this new set of data with added map layout data
        self.tune_data_organization_and_validation()
        self.tune_data_with_layout_data = self.from_tune_datums(tune_data=tune_data_with_layout_data,
                                                                north_is_highband=self.north_is_highband)
        self.tune_data_without_layout_data = self.from_tune_datums(tune_data=tune_data_without_layout_data,
                                                                   north_is_highband=self.north_is_highband)

    def plot_with_psat(self, psat_by_temp, freq_obs_ghz_target, temp_k, psat_min=0.0, psat_max=3.0e-12,
                       show_plot=False, save_plot=False):
        # # Plot layout initialization
        # colorbar across the bottom, key/legend on the top, A B and D polarization maps across the middle
        left = 0.05
        bottom = 0.08
        right = 0.99
        top = 0.85
        x_total = right - left
        y_total = top - bottom

        x_between_plots = 0.00
        y_between_plots = 0.00

        x_polar = (x_total - 2.0 * x_between_plots) / 3.0

        y_colorbar = 0.12
        y_legend = 0.0
        y_polar = y_total - y_colorbar - y_legend - 2.0 * y_between_plots

        bottom_colorbar = bottom
        top_colorbar = bottom_colorbar + y_colorbar
        bottom_polar = top_colorbar + y_between_plots

        bottom_legend = top - y_legend
        coord_legend = [left, bottom_legend, x_total, y_legend]
        coord_colorbar = [left, bottom_colorbar, x_total, y_colorbar]

        polar_coords = {}
        left_polar = left

        for polar_letter in ['A', 'B', 'D']:
            right_polar = left_polar + x_polar
            polar_coords[polar_letter] = [left_polar, bottom_polar, x_polar, y_polar]
            left_polar = right_polar + x_between_plots

        # initialize the plot
        fig = plt.figure(figsize=(16, 6))

        # ax_legend = fig.add_axes(coord_legend, frameon=False)
        # ax_legend.tick_params(axis='x',  # changes apply to the x-axis
        #                       which='both',  # both major and minor ticks are affected
        #                       bottom=False,  # ticks along the bottom edge are off
        #                       top=False,  # ticks along the top edge are off
        #                       labelbottom=False)
        # ax_legend.tick_params(axis='y',  # changes apply to the x-axis
        #                       which='both',  # both major and minor ticks are affected
        #                       left=False,  # ticks along the bottom edge are off
        #                       right=False,  # ticks along the top edge are off
        #                       labelleft=False)

        ax_colorbar = fig.add_axes(coord_colorbar, frameon=False)
        ax_colorbar.tick_params(axis='x',  # changes apply to the x-axis
                                which='both',  # both major and minor ticks are affected
                                bottom=False,  # ticks along the bottom edge are off
                                top=False,  # ticks along the top edge are off
                                labelbottom=False)
        ax_colorbar.tick_params(axis='y',  # changes apply to the x-axis
                                which='both',  # both major and minor ticks are affected
                                left=False,  # ticks along the bottom edge are off
                                right=False,  # ticks along the top edge are off
                                labelleft=False)

        ax = {}
        for polar_letter in ['A', 'B', 'D']:
            ax[polar_letter] = fig.add_axes(polar_coords[polar_letter], frameon=False)
            ax[polar_letter].tick_params(axis='x',  # changes apply to the x-axis
                                         which='both',  # both major and minor ticks are affected
                                         bottom=False,  # ticks along the bottom edge are off
                                         top=False,  # ticks along the top edge are off
                                         labelbottom=False)
            ax[polar_letter].tick_params(axis='y',  # changes apply to the x-axis
                                         which='both',  # both major and minor ticks are affected
                                         left=False,  # ticks along the bottom edge are off
                                         right=False,  # ticks along the top edge are off
                                         labelleft=False)

        # # Data mapping and handling
        # select the data to be plot at a single temperature
        psat_data_at_temp = psat_by_temp[temp_k]
        # initialize the scatter plot data arrays per-pole
        det_x_data = {'A': [], 'B': [], 'D': []}
        det_y_data = {'A': [], 'B': [], 'D': []}
        det_psat_data = {'A': [], 'B': [], 'D': []}
        # set the color scale
        norm = colors.Normalize(vmin=psat_min, vmax=psat_max)
        cmap = plt.get_cmap('gist_rainbow_r')
        scalar_map = cm.ScalarMappable(norm=norm, cmap=cmap)
        fig.colorbar(scalar_map, ax=ax_colorbar, orientation='horizontal', fraction=1.0)
        # get all the data to render a scatter plot
        for tune_datum in list(self):
            smurf_band = tune_datum.smurf_band
            smurf_channel = tune_datum.smurf_channel
            det_x = tune_datum.det_x
            det_y = tune_datum.det_y
            pol = tune_datum.pol
            freq_obs_ghz = tune_datum.freq_obs_ghz
            if all([smurf_channel != -1, det_x is not None, det_y is not None, smurf_band in psat_data_at_temp.keys(),
                    smurf_channel in psat_data_at_temp[smurf_band].keys(), freq_obs_ghz == freq_obs_ghz_target]):
                det_x_data[pol].append(det_x)
                det_y_data[pol].append(det_y)
                det_psat_data[pol].append(psat_data_at_temp[smurf_band][smurf_channel])
        # Rendering the plot data
        for pol in ['A', 'B', 'D']:
            color_vals = [scalar_map.to_rgba(det_psat) for det_psat in det_psat_data[pol]]
            ax[pol].scatter(det_x_data[pol], det_y_data[pol], c=color_vals, vmin=psat_min, vmax=psat_max)
            ax[pol].set_title(f"Polarization '{pol}'")
        # Plot components for whole plot
        title_str = f"{freq_obs_ghz_target} GHz Psat at 100mK CL={temp_k}K, " + \
                    f"range={psat_min / 1.0e-12}-{psat_max / 1.0e-12} pW"
        fig.suptitle(title_str)
        # Display and saving
        if save_plot:
            plot_filename = os.path.join(self.plot_dir, title_str.replace(' ', '_') + '.png')
            plt.savefig(plot_filename)
            print(f'Plot saved at: {plot_filename}')
        if show_plot:
            plt.show()


def read_tunefile(tunefile, return_pandas_df=False):
    operate_tune_data = OperateTuneData(tune_path=tunefile, design_file_path=None)
    if return_pandas_df:
        return operate_tune_data.return_pandas_df()
    else:
        return operate_tune_data

