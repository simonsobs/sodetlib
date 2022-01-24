"""
Author: Caleb Wheeler, written by reading code originally writen by Kaiwen Zheng

This file handles the data structures, reading, and analysis of resonator frequency to smurf band-channel index pairs.
"""
import os.path
from copy import deepcopy
from getpass import getuser
from operator import attrgetter, itemgetter

import numpy as np
import pandas
import pandas as pd
if getuser() in {'chw3k5', 'cwheeler'}:
    # this only happens on Caleb's computers
    import matplotlib as mpl
    # an interactive backend to render the plots, allows for zooming/panning and other interactions
    mpl.use(backend='TkAgg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm

from sodetlib.detmap.simple_csv import read_csv
from sodetlib.detmap.vna_to_smurf import emulate_smurf_bands
from sodetlib.detmap.detmap_config import abs_path_detmap
from sodetlib.detmap.design_mapping import design_pickle_to_csv, operate_tune_data_csv_filename, get_filename, \
    map_by_res_index, map_by_freq, order_smurf_band_res_index
from sodetlib.detmap.single_tune import TuneDatum, tune_data_header, tune_data_column_names


class OperateTuneData:
    """For storing and operating on measurements of resonator frequencies across a single focal plane module.

    Mapping of metadata types requires one-to-one relationships of data and metadata. Incorrectly formatted and
    ambiguous data will raise exceptions directly after read-in. Data Hygiene is the priority, as this is a critical
    step in a long chain of contextualizing what was actually observed. See the method
    tune_data_organization_and_validation().

    Metadata is mapped to the design data in two steps first in the map_design_data() method and then in the
    map_layout_data() method.

    This class is fast because it uses dictionaries, dict, to store the data. Dictionaries use hash tables instead of
    direct value comparison in addition to a hierarchical structure the reduces the scope for searching and comparison.
    If you find this structure is new or confusing for direct access to the data, consider using some data output
    methods described below.

    This class supports casting from a number of data formats.
        See methods with the `from_` prefix casting from Python objects like peak_array, pandas_dataframes, and
        tune_datums.
        See methods with the `read_` prefix file read in from tunefiles (.npy extentions), design_files, and CSV
        files that are in the same format as the write_csv() method.

    Output methods/formats include:
        - Writing output in CSV, write_csv().
        - Export as a Pandas dataframe, return_pandas_dataframe.
        - Since this class defines an iterable method, __iter__(), and instance of the class can be cast into a list or
        other iterable (duck-typing). For example given and instance of OperateTuneData called operate_tune_data
        a list of the data can be obtained from list_of_tune_datums = list(operate_tune_data). This is used
        extensively with in this class's method via list(self) to iterate over and operate on all the existing data.

    Support of the `+` is available (see __add__()) which is good for adding two instances of OperateTuneData,
    for example the North and South sides of a focal plane array, into a single instance of OperateTuneData.


    Class Variables
    ---------------
    design_file_pickle_to_csv_header : list
        List of header conversions as 2-element tuples. The first element of the tuple the column name in the pickle
        file, and the second element is how the column is renamed in the subsequent CSV file. The order of this list
        determines the order of columns in the CSV file created from the design pickle file.
    output_prefix: str
        A default filename prefix that is used when `output_path_csv=None` in the method write_csv()
    design_attributes: set
        The attributes of design data that are mapped into a TuneDatum, not freq_mhz which is handled differently.
        Note: `set`s use hash tables, which are fast for comparison and sorting.
    layout_attributes: set
        The attributes of layout data that are mapped into a TuneDatum.
        Note: `set`s use hash tables, which are fast for comparison and sorting.
    is_north_iter_order: list
        This controls the interation order in __iter__() for values that are allowed for TuneDatum.is_north, the
        outermost loop of the __iter__() methods. This is done to cause consistent outputs when building lists and other
        iterables.
    allowed_instance_vars: set
        This controls what is allowed to be set as OperateTuneData attributes that are instances (subsets) of the
        parent instance in the method update_tunes().


    Attributes
    ----------
    tune_path : object: str, optional
        The file path for a tunefile or g3stream file, with .npy extension, or for a CSV file that was writing with the
        write_csv() method. This is specified on initialization, and triggers an automatic data read-in using the method
        read_tunefile() or read_csv(), depending on the file's extension. None is also an expected value, which does not
        trigger a data read-in.
    is_g3timestream : bool
        G3-timestream data has a slightly different format that needs a different read-in process. This bool variable
        toggles that process in the read_tunefile() method. This is set via the __init__() method and is False by
        default. This variable does nothing unless the tune_path value has an extension of .npy.
    design_file_path : object: str, optional
        The file path for resonator design data. Design data has a similar data topology to tunefiles, so this class
        is also use to read-in and operate on resonator design data. When tune_path is None and design_file_path is
        a path, auto read-in is triggered using the method read_design_file(). None is the default value.
    layout_position_path : object: str, optional
        The file path for a CSV file that sets the mapping of uMUX chip position on the focal plane array to the
        uMUX-band (frequency band for the uMUX resonators design) for each uMUX chip. This can be specified in the
        __init__() method or when calling the map_design_data() method where this data is read-in and applied.
    north_is_highband : object : str, optional
        Since 2 SMURF systems must be used to read a single focal plane array, we use this variable to distinguish
        resonator data on the north side of the and the south side. This is done satirically with in a tunefile
        by simply adding 2000 MHz to one side and calling it SMuRF band 4-7, the high bands, and 0-3 the low bands.
        None is also an exceptable input, and is what is required for the design data, which does not participate in the
        fiction of high and low bands.
    is_smurf : bool
        We make additional data structures when we have SMuRF shaped data. This attribute determines if additional data
        structures can be made in the tune_data_organization_and_validation() method. It initializes as False and is
        made True when SMuRF data is read-in read_tunefile().
    imported_design_data : bool
        Once the design data is imported and mapped, we can make additional data structures in the
        tune_data_organization_and_validation() method. Initialized as False, it is turned to True in the
        map_design_data() method.
    tune_data : object : set
        The fundamental data object for this class that collects the TuneDatum() objects when data is read-in or
        otherwise cast into this class. Has an initial value of None, but is redefined as a set() when a 'read_' or
        'from_' method is invoked.
    tune_data_side_band_res_index : object : dict
        A hierarchical dict (3-level) mapping of TuneDatum() objects. As the name suggests, the dictionary hierarchy is
        by array side answering the question 'is north?' (False, True, None),
        then SMuRF-band (0, 1, 2, 3, 4, 5, 6, 7, 8), and finally by resonator index (integers from 0 to ~280 and -1).
        Each dictionary key triplicate of (side, band, index) uniquely maps to a single TuneDatum(), except for
        res_index==-1, which has special handling.
    tune_data_smurf_band_channel: object : dict
        A hierarchical dict (2-level) mapping of TuneDatum() objects. As the name suggests, the dictionary hierarchy is
        by SMuRF-band (0, 1, 2, 3, 4, 5, 6, 7, 8) and SMuRF channel (integers from 0 to ~500 and -1). Each dictionary
        key pair of (band, channel) uniquely maps to a single TuneDatum(), except for channel==-1, which has special
        handling.
    tune_data_muxpos_bondpad : object : dict
        A hierarchical dict (2-level) mapping of TuneDatum() objects. As the name suggests, the dictionary hierarchy is
        by uMUX position on a focal plane array (integers 0-27 and None) and bond pad position on-chip (integers 0-65). Each
        dictionary key pair of (muxpos, bondpad) uniquely maps to a single TuneDatum(), execpt for None, None which
        has special handling.
    pandas_data_frame : object : pandas.DataFrame
        When data is exported using the method return_pandas_dataframe(), the dataframe is returned but also stored in
        this attribute. This attribute is None until the return_pandas_dataframe() method is called.
    tune_data_with_design_data:  object : OperateTuneData
        When the method map_design_data() is called, we make another instance of this class with only the TuneDatum()
        objects that were successfully mapped to design data. Since this is an instance of this class, it includes
        all the same data exporting methods and analysis to answer the question of why was this mapping successful and
        on what data? The attribute is None until map_design_data() is called.
    tune_data_without_design_data:  object : OperateTuneData
        When the method map_design_data() is called, we make another instance of this class with only the TuneDatum()
        objects that were not-successfully mapped to design data. Since this is an instance of this class, it includes
        all the same data exporting methods and analysis to answer the question of why was this mapping not-successful
        and on what data? The attribute is None until map_design_data() is called.
    tune_data_with_layout_data:  object : OperateTuneData
        When the method map_layout_data() is called, we make another instance of this class with only the TuneDatum()
        objects that were successfully mapped to layout data. Since this is an instance of this class, it includes
        all the same data exporting methods and analysis to answer the question of why was this mapping successful and
        on what data? The attribute is None until map_layout_data() is called.
    tune_data_without_layout_data:  object : OperateTuneData
        When the method map_layout_data() is called, we make another instance of this class with only the TuneDatum()
        objects that were not-successfully mapped to layout data. Since this is an instance of this class, it includes
        all the same data exporting methods and analysis to answer the question of why was this mapping not-successful
        and on what data? The attribute is None until map_layout_data() is called.
    plot_dir: str
        The path to where the 'plot' directory is located. This is the directory where plots are saved. This is
        determined automatically in the __init__() method from the parent directory of tune_path or it tune
        path is None, it is defined as sodetlib/sodetlib/detmap/plots/.
    """
    design_file_pickle_to_csv_header = [('Band', 'mux_band'), ('Freq-index', 'mux_channel'),
                                        ('Frequency(MHz)', 'freq_mhz'), ('Subband', 'mux_subband'), ('Pad', 'bond_pad')]
    output_prefix = 'test_operate_tune_data'
    design_attributes = {'bond_pad', 'mux_band', 'mux_channel', 'mux_subband', 'mux_layout_position'}
    layout_attributes = {"bias_line", "pol", "bandpass", "det_row", "det_col", "rhomb", "is_optical",
                         "det_x", "det_y"}
    is_north_iter_order = [True, False, None]
    allowed_instance_vars = {'design', 'layout'}

    def __init__(self, tune_path=None, is_g3timestream=False, design_file_path=None, layout_position_path=None,
                 north_is_highband=None):
        """Initialization that includes auto read-in for a few supported file formats of resonator frequencies.

        Parameters
        ----------
        tune_path : object: str, optional
            Sets the attribute by the same name. The file path for a tunefile or g3stream file, with .npy extension, or
            for a CSV file that was writing with the write_csv() method. This is specified on initialization, and
            triggers an automatic data read-in, using the method read_tunefile() or read_csv() depending on the file's
            extension. None is also an expected value, which does not trigger a data read in.
        is_g3timestream : bool, optional
            Sets the attribute by the same name. G3-timestream data has a slightly different format that needs a
            different read-in process. This bool variable toggles that process in the read_tunefile() method. This is
            set to False by default. This variable does nothing unless the tune_path value has an extension of .npy.
        design_file_path : object: str, optional
            Sets the attribute by the same name. The file path for resonator design data. Design data has a similar data
            topology to tunefiles, so this class is also use to read-in and operate on resonator design data. When
            tune_path is None and design_file_path is a path, auto read-in is triggered using the method
            read_design_file(). None is the default value.
        layout_position_path : object: str, optional
            Sets the attribute by the same name. The file path for a CSV file that sets the mapping of uMUX chip
            position on the focal plane array to the uMUX-band (frequency band for the uMUX resonators design) for each
            uMUX chip. This can be specified in method map_design_data() method where this data is read-in and applied.
        north_is_highband : object : str, optional
            Sets the attribute by the same name. Since 2 SMURF systems must be used to read a single focal plane array,
            we use this variable to distinguish resonator data on the north side of the and the south side. This is done
            artificially with in a tunefile by simply adding 2000 MHz to one side and calling it SMuRF band 4-7,
            the high bands, whereas 0-3 are the low bands. None is also an exceptable input, and is what is required for
             the design data, which does not participate in the fiction of high and low bands.
        """
        self.tune_path = tune_path
        if self.tune_path is None:
            extension = None
        else:
            basename = os.path.basename(self.tune_path)
            _prefix, extension = basename.rsplit('.', 1)
            extension = extension.lower()
        self.is_g3timestream = is_g3timestream
        self.design_file_path = design_file_path
        self.layout_position_path = layout_position_path
        self.north_is_highband = north_is_highband

        # we make additional data structures when we have SMuRF shaped data, triggered on data read-in
        self.is_smurf = False
        # we can make additional data structures when we have the design data imported into measured tunes
        self.imported_design_data = False

        # initial values for variables that are populated in this class's methods.
        self.tune_data = None
        self.tune_data_side_band_res_index = None
        self.tune_data_smurf_band_channel = None
        self.tune_data_muxpos_bondpad = None
        self.pandas_data_frame = None
        self.tune_data_with_design_data = None
        self.tune_data_without_design_data = None
        self.tune_data_with_layout_data = None
        self.tune_data_without_layout_data = None

        # the directory for output plots
        if self.tune_path is None:
            self.plot_dir = os.path.join(os.path.dirname(abs_path_detmap), 'plots')
        else:
            self.plot_dir = os.path.join(os.path.dirname(self.tune_path), 'plots')
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
        elif self.design_file_path is not None:
            self.read_design_file()

    def __iter__(self):
        """
        This defines the way data is outputted for this class, when certain built-in Python
        functions, like list(), are used.

        While the data is stored in unordered dictionaries and sets for quick searching and
        comparison, it is often desirable to have a constantly ordered output for analysis,
        debugging, and writing files. This is where we determine that ordering.
        """

        for is_north in sorted(self.tune_data_side_band_res_index.keys(), key=self.is_north_rank_key):
            tune_data_this_side = self.tune_data_side_band_res_index[is_north]
            for tune_datum in order_smurf_band_res_index(tune_data_band_index=tune_data_this_side):
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
        Populates the self.tune_data_side_band_res_index instance variable. Checks for validation should be
         conducted here. This method is meant to be used internally with in this class' methods, and should not be
         required for user understanding.

        Also, this method does a data validation, raising exceptions for unexpected data sets before the data is
         probated to analysis, matching, where it could cause unexpected results.


        Returns: self.tune_data_side_band_res_index a three level dictionary tree that uses a unique set of keys to lead
                 to a single TuneDatum. These keys are is_north, smurf_band, res_index. The allowed
                 values for these keys are: 1) is_north {True, False, None}
                                            2) smurf_band ints {-1, 0, 1, 2, 3, 4, 5, 6, 7}
                                            3) res_index {-1, 0, 1, 2, ... , ~280}.

                There is special handling channel -1, with is denotes a found but untracked/unused resonator
                by the SMuRF. First, when channel == -1, then smurf_band is always also -1. Many unused resonatotors
                given this designation, instead of these keys leading to a unique TuneDatum, they lead to a set() of
                TuneDatum with smurf_band == -1 and channel == -1.
        -------
        """

        self.tune_data_side_band_res_index = {}
        # loop over all the TuneDatum(s), here we sort the data to get consistent results for debugging
        for tune_datum in sorted(self.tune_data, key=attrgetter('smurf_band', 'res_index', 'freq_mhz')):
            smurf_band_this_datum = tune_datum.smurf_band
            res_index = tune_datum.res_index
            smurf_channel = tune_datum.smurf_channel
            is_north = tune_datum.is_north
            # make a new dictionary instance if this is the first resonance found for this side.
            if is_north not in self.tune_data_side_band_res_index.keys():
                self.tune_data_side_band_res_index[is_north] = {}
            # make a new dictionary instance if this is the first resonance found in this band.
            if smurf_band_this_datum not in self.tune_data_side_band_res_index[is_north].keys():
                self.tune_data_side_band_res_index[is_north][smurf_band_this_datum] = {}
            # map the data to a dictionary structure
            if smurf_channel == -1:
                # special handling for the smurf_channel value -1
                if -1 not in self.tune_data_side_band_res_index[is_north][smurf_band_this_datum].keys():
                    self.tune_data_side_band_res_index[is_north][smurf_band_this_datum][-1] = set()
                self.tune_data_side_band_res_index[is_north][smurf_band_this_datum][smurf_channel].add(tune_datum)
            elif res_index in self.tune_data_side_band_res_index[is_north][smurf_band_this_datum].keys():
                # This is happens if there is already TuneDatum for this combination of side-smurf_band-res_index
                existing_tune_datum = self.tune_data_side_band_res_index[is_north][smurf_band_this_datum][res_index]
                raise KeyError(f'Only Unique side-band-res_index combinations are allowed. ' +
                               f'For side: {is_north} smurf_band: {smurf_band_this_datum} ' +
                               f'and res_index: {res_index} ' +
                               f'The existing datum: {existing_tune_datum} ' +
                               f'uses has the same band-channel data as the new: {tune_datum}')
            else:
                # add the tune datum to this mapping
                self.tune_data_side_band_res_index[is_north][smurf_band_this_datum][res_index] = tune_datum
        if self.is_smurf:
            # do a second mapping by smurf channel, used for mapping p-sat data, also from smurf
            self.tune_data_smurf_band_channel = {}
            for tune_datum in list(self):
                smurf_band = tune_datum.smurf_band
                smurf_channel = tune_datum.smurf_channel
                if smurf_band not in self.tune_data_smurf_band_channel.keys():
                    self.tune_data_smurf_band_channel[smurf_band] = {}
                if smurf_channel == -1:
                    if smurf_channel not in self.tune_data_smurf_band_channel[smurf_band].keys():
                        self.tune_data_smurf_band_channel[smurf_band][smurf_channel] = set()
                    self.tune_data_smurf_band_channel[smurf_band][smurf_channel].add(tune_datum)
                else:
                    if smurf_channel in self.tune_data_smurf_band_channel[smurf_band].keys():
                        # This is happens if there is already TuneDatum for this pair of smurf_band-smurf_channel
                        existing_tune_datum = self.tune_data_smurf_band_channel[smurf_band][smurf_channel]
                        raise KeyError(f'Only Unique pairs of smurf_band-smurf_channel are allowed. ' +
                                       f'For smurf_band: {smurf_band} ' +
                                       f'and smurf_channel: {smurf_channel} ' +
                                       f'The existing datum: {existing_tune_datum} ' +
                                       f'uses has the same band-channel data as the new: {tune_datum}')
                    else:
                        self.tune_data_smurf_band_channel[smurf_band][smurf_channel] = tune_datum
        if self.imported_design_data:
            # with design data imported we can do a mapping useful for hardware data linking/organization
            self.tune_data_muxpos_bondpad = {}
            for tune_datum in list(self):
                mux_layout_position = tune_datum.mux_layout_position
                bond_pad = tune_datum.bond_pad
                if mux_layout_position not in self.tune_data_muxpos_bondpad.keys():
                    self.tune_data_muxpos_bondpad[mux_layout_position] = {}
                if bond_pad not in self.tune_data_muxpos_bondpad[mux_layout_position].keys():
                    if bond_pad is None:
                        self.tune_data_muxpos_bondpad[mux_layout_position][None] = set()
                elif bond_pad is not None:
                    existing_tune_datum = self.tune_data_muxpos_bondpad[mux_layout_position][bond_pad]
                    raise KeyError(f'Only Unique mux_layout_position-bond_pad combinations are allowed. ' +
                                   f'For mux_layout_position: {mux_layout_position} ' +
                                   f'and bond_pad: {bond_pad} ' +
                                   f'The existing datum: {existing_tune_datum} ' +
                                   f'uses has the same band-channel data as the new: {tune_datum}')
                if bond_pad is None:
                    self.tune_data_muxpos_bondpad[mux_layout_position][None].add(tune_datum)
                else:
                    self.tune_data_muxpos_bondpad[mux_layout_position][bond_pad] = tune_datum

    def update_tunes(self, tune_data_new: set, var_str: str, tune_data_with: set, tune_data_without: set):
        """ A base method used by both map_design_data() and map_layout_data() methods.

        This first updates this instance with TuneDatum Objects that have had new metadata added by the
        map_design_data() and map_layout_data() methods.

        This class then makes two new instances of OperateTuneData one with TuneDatums that had successfully had
        metadata mapped to them and one that did not use the patterns 'tune_data_with_{var_str}_data' and
        'tune_data_without_{var_str}_data' respectively. Since these are instances of OperateTuneData(), both instances
        they have all the same access to data export and analysis tools.

        Parameters
        ----------
        tune_data_new : set
            A set of TuneDatum() objects to reset the attribute of self.tune_data in this class.
        var_str : str
            Needed to set the attribute name, must be in str in the class variable
            OperateTuneData.allowed_instance_vars. This is used to digitising that data's source, i.e. layout or design
            mappings.
        tune_data_with : set
            A set of TuneDatum() objects where mapping *was* successful.
        tune_data_without : set
            A set of TuneDatum() objects where mapping *was not* successful.
        """
        # if everything was successful, assign the combined mapping to this instances self.tune_data
        self.tune_data = tune_data_new
        # do a data organization and validation
        self.tune_data_organization_and_validation()
        # make new instances of this class (within the variables of this class!)
        if var_str in self.allowed_instance_vars:
            # with the data found and the not found data associated with var_str
            self.__setattr__(f'tune_data_with_{var_str}_data',
                             from_tune_datums(tune_data=tune_data_with, north_is_highband=self.north_is_highband,
                                              is_smurf=self.is_smurf, imported_design_data=self.imported_design_data))
            self.__setattr__(f'tune_data_without_{var_str}_data',
                             from_tune_datums(tune_data=tune_data_without, north_is_highband=self.north_is_highband,
                                              is_smurf=self.is_smurf, imported_design_data=self.imported_design_data))
        else:
            raise KeyError(f'var_str: {var_str} is not one of the values allowed in ' +
                           f'OperateTuneData.allowed_instance_vars: {self.allowed_instance_vars}.')

    def from_pandas_dataframe(self, data_frame: pandas.DataFrame, is_highband=None, is_north=None):
        """ Import data into this class from a Pandas.DataFrame() instance.

        Parameters
        ----------
        data_frame : pandas.DataFrame
            https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html
        is_highband : object : bool, optional
            Bool or None, answers the question 'is this data equivalent to the SMuRF highband?' If you need to import
            both high and low band data, create two instances of OperateTuneData, one for highband (is_highband=True)
            and one for lowband (is_highband=False), then combine them using the + operator, see the __add__() method in
            this class.
        is_north : object : bool, optional
            Bool or None, answers the question 'is this data from resonators on the North side of a focal plane array?'
            If you need to import both is_north=False data and is_north=True data, create two instances of
            OperateTuneData, one for False and one for True and then combine them using the + operator, see the
            __add__() method in this class.
        """
        # initialize the data the variable that stores the data we are reading.
        self.tune_data = set()
        # loop over the data frame. !Never do this for calculations, only for casting as done below!
        # for more info:
        # https://stackoverflow.com/questions/16476924/how-to-iterate-over-rows-in-a-dataframe-in-pandas#:~:text=DataFrame%20in%20Pandas%3F-,Answer%3A%20DON%27T*!,-Iteration%20in%20Pandas
        for index, row in data_frame.iterrows():
            # These are always required. Need to be hashable and cast as integers for faster search nad comparison.
            # set the tune datum for this row, note ** is a kwargs variable unpacking.
            tune_datum_this_res = TuneDatum(**row, is_north=is_north, is_highband=is_highband)
            self.tune_data.add(tune_datum_this_res)
        # do a data organization and validation
        self.tune_data_organization_and_validation()

    def from_peak_array(self, peak_array_mhz, is_north=None, is_highband=None,
                        shift_mhz=10.0, smurf_bands=None):
        """Import data into this class from an iterable of resonator values in MHz.

        Parameters
        ----------
        peak_array_mhz : iterable
            Resonator values in MHz, objects can be anything handled be the sorted function that returns a list of
            float values in MHz i.e. sorted(peak_array_mhz)
        is_north : object : bool, optional
            is_north : object : bool, optional
            Bool or None, answers the question 'is this data from resonators on the North side of a focal plane array?'
            If you need to import both is_north=False data and is_north=True data, create two instances of
            OperateTuneData, one for False and one for True and then combine them using the + operator, see the
            __add__() method in this class.
        is_highband : object : bool, optional
            Bool or None, answers the question 'is this data equivalent to the SMuRF highband?' If you need to import
            both high and low band data, create two instances of OperateTuneData, one for highband (is_highband=True)
            and one for lowband (is_highband=False), then combine them using the + operator, see the __add__() method in
            this class.
        shift_mhz : float, optional
            It is known that the measured frequencies of the SMuRF system are shifted compared to VNA measurements by
            approximately 10.0 MHz. This applies a shift to the VNA data to deliver projection of  the measured VNA
            frequencies into the reference frame of SMuRF data. The default shift is + 10.0 MHz added to the VNA data.
        smurf_bands : object : iterable, optional
            Default of None set the SMuRF bands to be range(8) -> [0, 1, 2, 3, 4, 5, 6, 7]. But if your data
            is only a subset of the smurf bands -> [4, 5, 6, 7], or a single SMuRF band -> [7] you cna indicate that
            here.
        """
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
                    res_index = channel_count_by_band[smurf_band]
                    # this is how the SMuRF reports the Highband designation
                    if is_highband:
                        smurf_band_emulated = smurf_band + 4
                    else:
                        smurf_band_emulated = smurf_band
                    # record the tune data
                    tune_datum_this_res = TuneDatum(freq_mhz=res_freq_mhz, smurf_band=smurf_band_emulated,
                                                    res_index=res_index, is_north=is_north, is_highband=is_highband)
                    # assign the tune datum to the storage variable
                    self.tune_data.add(tune_datum_this_res)
                    # iterate the channel counter
                    channel_count_by_band[smurf_band] += 1
                    # no need keep searching after we find the correct band
                    break
        # do a data organization and validation
        self.tune_data_organization_and_validation()

    def return_pandas_dataframe(self):
        """ Get a pandas.DataFrame from the tune_data.

        Returns
        -------
        data_frame : pandas.DataFrame
            https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html
        """
        # make sure the tune data was load before this method was called.
        if self.tune_data_side_band_res_index is None:
            raise IOError(f'No tune data has been loaded.')
        # initialize the pandas data frame
        self.pandas_data_frame = pd.DataFrame({header_key: [] for header_key in tune_data_column_names})
        # loop over the all the data in this class, not the behavior of list(self) is set by the __iter__ method, above
        for tune_datum_this_res in list(self):
            datum_dict = tune_datum_this_res.dict()
            self.pandas_data_frame = self.pandas_data_frame.append(datum_dict, ignore_index=True)
        return self.pandas_data_frame

    def write_csv(self, output_path_csv=None):
        """ Write a CSV file from the tune_data.

        Parameters
        ----------
        output_path_csv : object : str, optional
            The path string for a CSV file created from the current data state in this instance. None will auto
            generate an output filename.
        """
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
        print(f'Output CSV written at: {output_path_csv}')

    def read_csv(self):
        """ Read a CSV file writen by an instance of this class that called the write_csv() method.

        self.tune_path is set via the __init__() method and this method is called automatically in __init__().
        """
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
        """ Read a .npy file that has tune_data, either a tunefile or g3timestream.

        self.tune_path is set via the __init__() method and this method is called automatically in __init__().
        """
        self.is_smurf = True
        # initialize the data the variable that stores the data we are reading.
        self.tune_data = set()
        # read the tune file
        if self.is_g3timestream:
            raw_data = np.load(self.tune_path, allow_pickle=True)
            tunefile_data = {}
            for smurf_band in range(8):
                res_index_dict_this_band = {}
                res_index_counter = 0
                for channel_num, possible_freq in list(enumerate(raw_data[smurf_band])):
                    if not np.isnan(possible_freq):
                        smurf_channel = {'freq': possible_freq, 'channel': channel_num, 'subband': None}
                        res_index_dict_this_band[res_index_counter] = smurf_channel
                        res_index_counter += 1
                if res_index_dict_this_band:
                    tunefile_data[smurf_band] = {'resonances': res_index_dict_this_band}
        else:
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
                available_res_indexes = sorted(resonator_data_this_band.keys())
                for res_index in available_res_indexes:
                    single_res = resonator_data_this_band[res_index]
                    if is_highband:
                        # the real frequency is 2000.0 less than what is reported for high band data
                        freq_mhz = single_res['freq'] - 2000.0
                    else:
                        freq_mhz = single_res['freq']
                    smurf_channel = single_res['channel']
                    smurf_subband = single_res['subband']
                    tune_datum_this_res = TuneDatum(freq_mhz=freq_mhz, smurf_band=smurf_band, res_index=res_index,
                                                    smurf_channel=smurf_channel, smurf_subband=smurf_subband,
                                                    is_north=is_north, is_highband=is_highband)
                    self.tune_data.add(tune_datum_this_res)
        # do a data organization and validation
        self.tune_data_organization_and_validation()

    def read_design_file(self):
        """ Read a design file (pickle, or CSV) that has designed resonator frequency data.

        If available, a CSV file is used. If only a pickle file is available, a human-readable CSV file is created,
        than that file is read in and used on subsequent read-in calls.

        self.design_file_path is set via the __init__() method and this method is called automatically in __init__().
        """
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
                        row_dict['res_index'] = channel_count_by_band[smurf_band]
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

    def map_design_data(self, design_data, layout_position_path=None, mapping_strategy='map_by_res_index'):
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
        # choose a mapping strategy
        if mapping_strategy in {0, '0', 0.0, 'map_by_res_index'}:
            tune_data_new, tune_data_with_design_data, tune_data_without_design_data = \
                map_by_res_index(tune_data=self.tune_data, design_attributes=self.design_attributes,
                                 design_data=design_data, mux_band_to_mux_pos_dict=mux_band_to_mux_pos_dict)
        elif mapping_strategy in {1, 1.0, '1', 'map_by_freq'}:
            tune_data_new, tune_data_with_design_data, tune_data_without_design_data = \
                map_by_freq(tune_data_side_band_res_index=self.tune_data_side_band_res_index,
                            design_attributes=self.design_attributes,
                            design_data=design_data, mux_band_to_mux_pos_dict=mux_band_to_mux_pos_dict)
        else:
            raise KeyError(f'Mapping Strategy: {mapping_strategy}, is not recognized.')
        # we can make additional data structures when we have the design data imported into measured tunes
        self.imported_design_data = True
        # reset this instance's tune_data
        self.update_tunes(tune_data_new=tune_data_new, var_str='design', tune_data_with=tune_data_with_design_data,
                          tune_data_without=tune_data_without_design_data)

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
        # reset this instance's tune_data
        self.update_tunes(tune_data_new=tune_data_new, var_str='layout',
                          tune_data_with=tune_data_with_layout_data, tune_data_without=tune_data_without_layout_data)

    def plot_with_psat(self, psat_by_temp, bandpass_target, temp_k, psat_min=0.0, psat_max=6.0e-12,
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
            bandpass = tune_datum.bandpass
            if all([smurf_channel != -1, det_x is not None, det_y is not None, smurf_band in psat_data_at_temp.keys(),
                    smurf_channel in psat_data_at_temp[smurf_band].keys(), bandpass == bandpass_target]):
                det_x_data[pol].append(det_x)
                det_y_data[pol].append(det_y)
                det_psat_data[pol].append(psat_data_at_temp[smurf_band][smurf_channel])
        # Rendering the plot data
        for pol in ['A', 'B', 'D']:
            color_vals = [scalar_map.to_rgba(det_psat) for det_psat in det_psat_data[pol]]
            ax[pol].scatter(det_x_data[pol], det_y_data[pol], c=color_vals, vmin=psat_min, vmax=psat_max)
            ax[pol].set_title(f"Polarization '{pol}'")
        # Plot components for whole plot
        title_str = f"{bandpass_target} GHz Psat at 100mK CL={temp_k}K, " + \
                    f"range={psat_min / 1.0e-12}-{psat_max / 1.0e-12} pW"
        fig.suptitle(title_str)
        # Display and saving
        if save_plot:
            plot_filename = os.path.join(self.plot_dir, title_str.replace(' ', '_') + '.png')
            plt.savefig(plot_filename)
            print(f'Plot saved at: {plot_filename}')
        if show_plot:
            plt.show()


def from_tune_datums(tune_data, north_is_highband, is_smurf: bool, imported_design_data: bool) -> OperateTuneData:
    """ Make a new instance of OperateTuneData from an iterable of TuneDatum() objects, usually a set or a list.

    Parameters
    ----------
    tune_data
        The fundamental data interable for OperateTuneData() that contains the TuneDatum() objects.
    north_is_highband : object : bool
        Since 2 SMURF systems must be used to read a single focal plane array, we use this variable to distinguish
        resonator data on the north side of the and the south side.
    is_smurf : bool
        We make additional data structures when we have SMuRF shaped data. This attribute determines if additional data
        structures can be made in the tune_data_organization_and_validation() method. It initializes as False and is
        made True when SMuRF data is read-in read_tunefile().
    imported_design_data : bool
        Once the design data is imported and mapped, we can make additional data structures in the
        tune_data_organization_and_validation() method. Initialized as False, it is turned to True in the
        map_design_data() method.

    Returns
    -------
    OperateTuneData
        An instance of the OperateTuneData data class for measurements of resonate frequencies on a single
        detector focal plane module.

    """
    new_operate_tune_data = OperateTuneData(north_is_highband=north_is_highband)
    # we make additional data structures when we have SMuRF data
    new_operate_tune_data.is_smurf = is_smurf
    # we can make additional data structures when we have the design data imported into measured tunes
    new_operate_tune_data.imported_design_data = imported_design_data
    # add in the tune data
    new_operate_tune_data.tune_data = set(tune_data)
    # run the data validation and make the data structures
    new_operate_tune_data.tune_data_organization_and_validation()
    return new_operate_tune_data
