import os
from operator import attrgetter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from single_tune import TuneDatum

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
    # for more info:
    # https://stackoverflow.com/questions/16476924/how-to-iterate-over-rows-in-a-dataframe-in-pandas#:~:text=DataFrame%20in%20Pandas%3F-,Answer%3A%20DON%27T*!,-Iteration%20in%20Pandas
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


def map_by_res_index(tune_data, design_attributes, design_data, mux_band_to_mux_pos_dict):
    # make a new set to hold the tune data that is updated with design data
    tune_data_new = set()
    # track TuneDatums that are mapped to a design record
    tune_data_with_design_data = set()
    # track TuneDatums that are *not* mapped to a design record
    tune_data_without_design_data = set()
    # is_highband has no meaning for design data
    design_tune_data_by_band_channel = design_data.tune_data_side_band_res_index[None]
    # loop overall the data
    for tune_datum in tune_data:
        # pull these properties out to access the design data
        is_north = tune_datum.is_north
        is_highband = tune_datum.is_highband
        smurf_band = tune_datum.smurf_band
        res_index = tune_datum.res_index
        # the high-band resonators were never fabricated, highband is really a positional designation for SMuRF
        if is_highband:
            design_band = smurf_band - 4
        else:
            design_band = smurf_band
        # see if there is design data available for this band-channel pair
        if design_band in design_tune_data_by_band_channel.keys() and \
                res_index in design_tune_data_by_band_channel[design_band].keys():
            design_datum = design_tune_data_by_band_channel[design_band][res_index]
            # we can extract specific parameters from the design_datum and build a dictionary
            design_dict = {design_key: design_datum.__getattribute__(design_key)
                           for design_key in design_attributes}
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

    return tune_data_new, tune_data_with_design_data, tune_data_without_design_data


def order_smurf_band_res_index(tune_data_band_index):
    smurf_bands = sorted(tune_data_band_index.keys())
    for smurf_band in smurf_bands:
        tune_data_this_band = tune_data_band_index[smurf_band]
        res_indexes = sorted(tune_data_this_band.keys())
        for res_index in res_indexes:
            tune_datum_this_band_and_channel = tune_data_this_band[res_index]
            if res_index == -1:
                # special handling for channel == -1
                freq_sorted_tune_data = sorted(tune_datum_this_band_and_channel, key=attrgetter('freq_mhz'))
                for tune_datum in freq_sorted_tune_data:
                    yield tune_datum
            else:
                # these are a required have a single TuneDatum for side-band-channel combinations
                yield tune_datum_this_band_and_channel


def smurf_band_res_index_to_freq(tune_data_band_index):
    tune_data_freq_ordered = sorted([tune_datum for tune_datum in order_smurf_band_res_index(tune_data_band_index)],
                                     key=attrgetter('freq_mhz'))
    freq_array = np.array([tune_datum.freq_mhz for tune_datum in tune_data_freq_ordered])
    tune_data_by_freq = {tune_datum.freq_mhz: tune_datum for tune_datum in tune_data_freq_ordered}
    return freq_array, tune_data_by_freq


def freq_rug_plot(freq_array_smurf, freq_array_design):
    fig = plt.figure(figsize=(16, 6))
    left = 0.05
    bottom = 0.08
    right = 0.99
    top = 0.85
    x_width = right - left
    y_height = top - bottom
    ax = fig.add_axes([left, bottom, x_width, y_height], frameon=False)
    # 'threads' of the rug plot
    ax.tick_params(axis="y", labelleft=False)
    for xdata, color, y_min, y_max in [(freq_array_smurf, 'dodgerblue', 0.0, 0.6),
                                       (freq_array_design, 'firebrick', 0.4, 1.0)]:
        for f_center in xdata:
            ax.plot((f_center, f_center), (y_min, y_max), ls='solid', linewidth=0.3, color=color, alpha=0.7)
    ax.set_ylim(bottom=0.0, top=1.0)
    ax.tick_params(axis='y',  # changes apply to the x-axis
                   which='both',  # both major and minor ticks are affected
                   left=False,  # ticks along the bottom edge are off
                   right=False,  # ticks along the top edge are off
                   labelleft=False)
    ax.tick_params(axis='x',  # changes apply to the x-axis
                   which='both',  # both major and minor ticks are affected
                   bottom=False,  # ticks along the bottom edge are off
                   top=True,  # ticks along the top edge are off
                   labelbottom=False,
                   labeltop=True)
    ax.xaxis.tick_top()
    ax.xaxis.tick_bottom()
    ax.set_xlabel(F"Frequency (MHz)")

    plt.show()


def map_by_freq(tune_data_side_band_res_index, design_attributes, design_data, mux_band_to_mux_pos_dict):
    # make a new set to hold the tune data that is updated with design data
    tune_data_new = set()
    # track TuneDatums that are mapped to a design record
    tune_data_with_design_data = set()
    # track TuneDatums that are *not* mapped to a design record
    tune_data_without_design_data = set()
    # one set of design frequencies is all that is needed for both sides of the UFM
    freq_array_design, tune_data_by_freq_design = \
        smurf_band_res_index_to_freq(tune_data_band_index=design_data.tune_data_side_band_res_index[None])
    # loop over both sides of the UFM
    for is_north in tune_data_side_band_res_index.keys():
        tune_data_this_side = tune_data_side_band_res_index[is_north]
        freq_array_smurf, tune_data_by_freq_smurf = \
            smurf_band_res_index_to_freq(tune_data_band_index=tune_data_this_side)
        freq_rug_plot(freq_array_smurf=freq_array_smurf, freq_array_design=freq_array_design)


    return tune_data_new, tune_data_with_design_data, tune_data_without_design_data
