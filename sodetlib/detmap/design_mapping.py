import os
from operator import attrgetter
from itertools import zip_longest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sodetlib.detmap.single_tune import TuneDatum

"""
File naming functions
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


def order_res_index(tune_data_by_res_index, ignore_neg_one=False):
    res_indexes = sorted(tune_data_by_res_index.keys())
    for res_index in res_indexes:
        tune_datum_this_band_and_channel = tune_data_by_res_index[res_index]
        if res_index == -1:
            if not ignore_neg_one:
                # special handling for channel == -1
                freq_sorted_tune_data = sorted(tune_datum_this_band_and_channel, key=attrgetter('freq_mhz'))
                for tune_datum in freq_sorted_tune_data:
                    yield tune_datum
        else:
            # these are a required have a single TuneDatum for side-band-channel combinations
            yield tune_datum_this_band_and_channel


def order_smurf_band_res_index(tune_data_band_index, ignore_neg_one=False):
    smurf_bands = sorted(tune_data_band_index.keys())
    for smurf_band in smurf_bands:
        tune_data_this_band = tune_data_band_index[smurf_band]
        for tune_datum in order_res_index(tune_data_by_res_index=tune_data_this_band, ignore_neg_one=ignore_neg_one):
            yield tune_datum


def smurf_band_res_index_to_freq(tune_data_band_index, ignore_neg_one=False):
    tune_data_freq_ordered = sorted([tune_datum for tune_datum in
                                     order_smurf_band_res_index(tune_data_band_index=tune_data_band_index,
                                                                ignore_neg_one=ignore_neg_one)],
                                     key=attrgetter('freq_mhz'))
    freq_array = np.array([tune_datum.freq_mhz for tune_datum in tune_data_freq_ordered])
    tune_data_by_freq = {tune_datum.freq_mhz: tune_datum for tune_datum in tune_data_freq_ordered}
    return freq_array, tune_data_by_freq


def f_array_and_tune_dict(tune_data_dict):
    tune_data_this_band_f_ordered = sorted([tune_data_dict[a_key] for a_key in tune_data_dict.keys()],
                                           key=attrgetter('freq_mhz'))
    freq_array = np.array([tune_datum.freq_mhz for tune_datum in tune_data_this_band_f_ordered])
    freq_to_datums = {freq_mhz: tune_datum for freq_mhz, tune_datum
                               in zip(freq_array, tune_data_this_band_f_ordered)}
    return freq_array, freq_to_datums


def freq_rug_plot(freq_array_smurf, freq_array_design, freq_array_smurf_shifted=None):
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
                                       (freq_array_design, 'firebrick', 0.4, 1.0),
                                       (freq_array_smurf_shifted, 'darkgoldenrod', 0.2, 0.8)]:
        if xdata is not None:
            if len(xdata) < 400:
                linewidth = 1.0
                alpha = 0.5
            else:
                linewidth = 0.3
                alpha = 0.7
            for f_center in xdata:
                ax.plot((f_center, f_center), (y_min, y_max), ls='solid', linewidth=linewidth, color=color,
                        alpha=alpha)
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


def find_nearest_index(array, value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or np.fabs(value - array[idx-1]) < np.fabs(value - array[idx])):
        return idx - 1
    else:
        return idx


def find_nearest(array, value):
    return array[find_nearest_index(array=array, value=value)]


def find_nearest_design_for_measured(design, measured):
    return np.array(find_nearest(array=design, value=meas) for meas in measured)


def nearest_diff(freq_array_smurf, freq_array_design):
    nearest_diff_values = np.fabs([find_nearest(array=freq_array_design, value=freq_smurf) - freq_smurf
                                   for freq_smurf in freq_array_smurf])
    return nearest_diff_values


def goal(freq_array_smurf, freq_array_design):
    nearest_diff_values = nearest_diff(freq_array_smurf=freq_array_smurf, freq_array_design=freq_array_design)
    sum_scaler = np.sum(nearest_diff_values ** 2.0)
    return sum_scaler


def transform(measure, b, m=1.0):
    return (m * measure) + b


def make_opt_func(freq_array_smurf, freq_array_design):
    def opt_func(tuple_like):
        b, m = tuple_like
        shifted_smurf = transform(measure=freq_array_smurf, b=b, m=m)
        sum_scaler = goal(freq_array_smurf=shifted_smurf, freq_array_design=freq_array_design)
        return sum_scaler
    return opt_func


def heal_mapping_right(design, measured):
    if len(measured) >= len(design):
        # There are more (or exactly equal number of) measured frequencies then available design frequencies
        design_to_measured_one_to_one = {des: meas for des, meas in zip(design, measured)}
        # if len(measured) == len(design) then this is an empty list
        measured_overrun = list(measured[len(design):])
        design_unmapped = []
    else:
        # There are less measured frequencies than available design frequencies, so some design frequencies are skipped.
        number_of_skips_available = len(design) - len(measured)
        # This is the mapping that would allow each meas to the closest design, but this is not one-to-one
        nearest_design_for_each_measured = find_nearest_design_for_measured(design=design, measured=measured)
        # pair each measured values with the nearest design value
        measured_to_nearest_design = [(meas, nearest_des) for meas, nearest_des in
                                      zip(measured, nearest_design_for_each_measured)]
        # initialize the data variables
        design_index_delta = 0
        design_to_measured_one_to_one = {}
        design_unmapped = []
        for meas_counter, (meas, nearest_des) in list(enumerate(measured_to_nearest_design)):
            design_to_possibly_skip = design = design[meas_counter + design_index_delta]

            if design_to_possibly_skip < nearest_des and number_of_skips_available != design_index_delta:
                # a skip is only allowed if the nearest_des is bigger then the design_to_possibly_skip
                # and the skips wer not already used.
                design_index_delta += 1
                design_unmapped.append(design_to_possibly_skip)
            else:
                design_to_measured_one_to_one[design_to_possibly_skip] = meas
        measured_overrun = []
    return design_to_measured_one_to_one, measured_overrun, design_unmapped


def heal_mapping_left(design, measured):
    design_right_transform = np.flip(design) * -1.0
    measured_right_transform = np.flip(measured) * -1.0
    design_to_measured_one_to_one_right_transform, measured_overrun_right_transform, design_unmapped_right_transform = \
        heal_mapping_right(design=design_right_transform, measured=measured_right_transform)
    design_to_measured_one_to_one = {design_right_key * -1.0:
                                         design_to_measured_one_to_one_right_transform[design_right_key] * -1.0
                                     for design_right_key in design_to_measured_one_to_one_right_transform.keys()}
    measured_overrun = [meas * -1.0 for meas in reversed(measured_overrun_right_transform)]
    design_unmapped = [des * -1.0 for des in reversed(design_unmapped_right_transform)]
    return design_to_measured_one_to_one, measured_overrun, design_unmapped


def map_by_freq(tune_data_side_band_res_index, design_attributes, design_data, mux_band_to_mux_pos_dict,
                ignore_smurf_neg_one=False, trim_at_mhz=10.0, show_plots=False):
    # make a new set to hold the tune data that is updated with design data
    tune_data_new = set()
    # track TuneDatums that are mapped to a design record
    tune_data_with_design_data = set()
    # track TuneDatums that are *not* mapped to a design record
    tune_data_without_design_data = set()
    # one set of design frequencies is all that is needed for both sides of the UFM
    design_tune_data_band_res_index = design_data.tune_data_side_band_res_index[None]
    freq_array_design, tune_data_by_freq_design = \
        smurf_band_res_index_to_freq(tune_data_band_index=design_tune_data_band_res_index)
    # loop over both sides of the UFM
    for is_north in tune_data_side_band_res_index.keys():
        # only consider the tune data this side of the UFM
        tune_data_this_side = tune_data_side_band_res_index[is_north]
        # get the tunes as a frequency array, and a dictionary with freq_mhz as a key to the tune_datums
        freq_array_smurf, tune_data_by_freq_smurf = \
            smurf_band_res_index_to_freq(tune_data_band_index=tune_data_this_side, ignore_neg_one=ignore_smurf_neg_one)
        # optimize the resonator to match across the entire side of a single UFM
        opt_func = make_opt_func(freq_array_smurf, freq_array_design)
        opt_result = minimize(fun=opt_func, x0=np.array([0.0, 1.0]), method='Nelder-Mead')
        if opt_result.success:
            b_opt, m_opt = opt_result.x
            shifted_smurf_optimal = transform(measure=freq_array_smurf, b=b_opt, m=m_opt)
            if show_plots:
                freq_rug_plot(freq_array_smurf=freq_array_smurf, freq_array_design=freq_array_design,
                              freq_array_smurf_shifted=shifted_smurf_optimal)
        else:
            if is_north:
                side_str = 'North Side UFM'
            elif is_north is None:
                side_str = ''
            else:
                side_str = 'South Side UFM'
            raise ValueError(f'No convergent Result in the {side_str} resonator mapping.')

        # With the resonators remapped to fit the design frequencies, now we will apply a mapping strategy
        # independently for each smurf band.
        for smurf_band in sorted(tune_data_this_side.keys()):
            tune_data_this_band = tune_data_this_side[smurf_band]
            # get the tune_datums ordered by frequency, built as a generator statement for speed, not clarity
            tune_data_this_band_f_ordered = sorted([tune_datum for tune_datum
                                                    in order_res_index(tune_data_by_res_index=tune_data_this_band,
                                                                       ignore_neg_one=ignore_smurf_neg_one)],
                                                   key=attrgetter('freq_mhz'))
            # get a frequency array
            freq_array_smurf = np.array([tune_datum.freq_mhz for tune_datum in tune_data_this_band_f_ordered])
            # apply the initial optimization
            freq_array_smurf_shifted = transform(measure=freq_array_smurf, b=b_opt, m=m_opt)
            # make a look-up dict for the shifted frequencies to the original tune datums
            shifted_to_tune_datums = {freq_shifted: tune_datum for freq_shifted, tune_datum
                                      in zip(freq_array_smurf_shifted, tune_data_this_band_f_ordered)}
            # get the design and look up dict
            freq_array_design, design_to_design_datums = \
                f_array_and_tune_dict(tune_data_dict=design_tune_data_band_res_index[smurf_band])

            if show_plots:
                freq_rug_plot(freq_array_smurf=freq_array_smurf, freq_array_design=freq_array_design,
                              freq_array_smurf_shifted=freq_array_smurf_shifted)
            # trim any outliers
            nearest_diff_values = nearest_diff(freq_array_smurf=freq_array_smurf_shifted,
                                               freq_array_design=freq_array_design)
            trimmed_tune_datums = {shifted_freq_mhz: shifted_to_tune_datums[shifted_freq_mhz]
                                   for shifted_freq_mhz, nearest_value_mhz
                                   in zip(freq_array_smurf_shifted, nearest_diff_values)
                                   if nearest_value_mhz < trim_at_mhz}
            # reset to the original frequencies use the trimmed data set
            freq_array_trimmed, trimmed_to_tune_datum = \
                f_array_and_tune_dict(tune_data_dict=trimmed_tune_datums)
            opt_func2 = make_opt_func(freq_array_smurf=freq_array_trimmed, freq_array_design=freq_array_design)
            opt_result = minimize(fun=opt_func2, x0=np.array([b_opt, m_opt]), method='Nelder-Mead')
            if opt_result.success:
                b_opt_this_band, m_opt_this_band = opt_result.x
            else:
                raise ValueError(f'No convergent Result for smurf band {smurf_band} resonator mapping.')
            # shift the values based on the optimization
            smurf_trimmed_and_shifted = transform(measure=freq_array_trimmed, b=b_opt_this_band, m=m_opt_this_band)
            if show_plots:
                freq_rug_plot(freq_array_smurf=freq_array_trimmed, freq_array_design=freq_array_design,
                              freq_array_smurf_shifted=smurf_trimmed_and_shifted)

            # map the shifted values to the original tune_datums
            trimmed_and_shifted_to_tune_datum = {freq_trim_shifted: trimmed_to_tune_datum[freq_trimmed]
                                                 for freq_trim_shifted, freq_trimmed
                                                 in zip(smurf_trimmed_and_shifted, freq_array_trimmed)}
            # find the closest indexes for the trimmed
            trimmed_and_shifted_to_design = {freq_trim_shifted: find_nearest_index(array=freq_array_design, value=freq_trim_shifted)
                                             for freq_trim_shifted in smurf_trimmed_and_shifted}

            closest_design_index = np.array(sorted(trimmed_and_shifted_to_design.keys()), dtype=int)
            #
            design_indexes_available = [an_index for an_index in range(len(freq_array_design))]
            if len(design_indexes_available) % 2 == 0:
                design_half_index = int(len(design_indexes_available) / 2)
            else:
                design_half_index = int((len(design_indexes_available) + 1) / 2)

            """
            Needs to be split by the frequencies to be mapped, to make dictionary's with frequencies keys.
            
            the mappings are monotonic, so left design indexes will be strictly greater then the right side
            """


            closest_design_indexes_left = closest_design_index[design_half_index < closest_design_index]
            smurf_trimmed_and_shifted_left = set(smurf_trimmed_and_shifted[design_half_index < closest_design_index])
            design_indexes_available_left = set(design_indexes_available[design_half_index:])
            mapping_per_index_counter = {}
            trimmed_and_shifted_one_to_one_design_left = {}
            for closest_design_index in closest_design_indexes_left:
                if closest_design_index in mapping_per_index_counter.keys():
                    mapping_per_index_counter[closest_design_index] += 1
                else:
                    mapping_per_index_counter[closest_design_index] = 1

            design_indexes_unmapped = design_indexes_available_left - set(mapping_per_index_counter.keys())
            freq_to_design_indexes_one_to_one = {design_index for design_index in mapping_per_index_counter.keys()
                                                 if mapping_per_index_counter[design_index] == 1}
            design_indexes_one_multi = {design_index for design_index in mapping_per_index_counter.keys()
                                        if mapping_per_index_counter[design_index] != 1}



            """
            start with this dict before entering the while loop that does the healing
            """
            trimmed_and_shifted_one_to_one_design_left = {freq: 'design_index_her' for freq in design_indexes_one_to_one}

            trimmed_and_shifted_one_to_one_design = {}
            while smurf_trimmed_and_shifted_left != set():
                """
                Needs to end when the multi are all gone
                """
                # keep doing this loop until if exits without reaching the break statement
                design_indexes_unmapped_array = np.array(sorted(design_indexes_unmapped))
                design_indexes_one_to_one_list = sorted(design_indexes_one_to_one)
                design_indexes_one_multi_list = sorted(design_indexes_one_multi)
                design_indexes_available_left_list = sorted(design_indexes_available_left)
                smurf_trimmed_and_shifted_left_list = sorted(smurf_trimmed_and_shifted_left)


                for design_index, smurf_freq in zip_longest(design_indexes_available_left_list,
                                                            smurf_trimmed_and_shifted_left_list):
                    """
                    This loop should only be over the remaining multi_desing indexes.
                    """
                    if design_index is None:
                        raise IndexError('write this part of the algorithm')
                    elif design_index in design_indexes_one_to_one:
                        trimmed_and_shifted_one_to_one_design[smurf_freq] = design_index
                        smurf_trimmed_and_shifted_left.remove(smurf_freq)
                        design_indexes_available_left.remove(design_index)
                        design_indexes_one_to_one.remove(design_index)
                    elif design_index in design_indexes_one_multi:
                        closest_unmapped = find_nearest(array=design_indexes_unmapped_array, value=design_index)
                        # here is the healing algorithm
                        if closest_unmapped < design_index:
                            # back the mapping up on value to fill in the unmapped index
                            for set_design_index in trimmed_and_shifted_one_to_one_design.keys():
                                if closest_unmapped < set_design_index < design_index:
                                    smurf_freq = trimmed_and_shifted_one_to_one_design[set_design_index]
                                    smurf_trimmed_and_shifted_left.add(smurf_freq)
                                    design_indexes_available_left.add(set_design_index)
                                    del trimmed_and_shifted_one_to_one_design[set_design_index]


                            print('test point 2')
                        else:
                            # move the mapping forward on index to fill in the unmapped index
                            design_indexes_unmapped.remove(closest_unmapped)
                        mapping_per_index_counter[design_index] = mapping_per_index_counter[design_index] - 1
                        if mapping_per_index_counter[design_index] == 1:
                            del mapping_per_index_counter[design_index]
                            design_indexes_one_multi.remove(design_index)

                            print('test point 3')


                else:
                    finished = True








            closest_design_indexes_right = closest_design_index[design_half_index >= closest_design_index]
            smurf_trimmed_and_shifted_right = smurf_trimmed_and_shifted[design_half_index >= closest_design_index]
            design_indexes_available_right = set(design_indexes_available[:design_half_index])



            print('temp test point')



















































    return tune_data_new, tune_data_with_design_data, tune_data_without_design_data
