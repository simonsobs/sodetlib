import os
from bisect import bisect
from typing import Union, NamedTuple
from operator import attrgetter


from random import randint
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


def add_design_data(meas_datum, des_datum, design_attributes, is_north):
    # we can extract specific parameters from the design_datum and build a dictionary
    design_dict = {design_key: des_datum.__getattribute__(design_key)
                   for design_key in design_attributes}
    # move the design frequency to the appropriate attribute
    design_dict['design_freq_mhz'] = des_datum.freq_mhz
    # get a mutable dictionary for the tune datum
    tune_dict = meas_datum.dict()
    # update the tune dict with the design parameters
    tune_dict.update(design_dict)
    # reassign these key-value pairs to the immutable TuneDatum
    tune_datum_with_design_data = TuneDatum(**tune_dict)
    return tune_datum_with_design_data


def map_by_res_index(tune_data, design_attributes, design_data):
    # make a new set to hold the tune data that is updated with design data
    tune_data_new = set()
    # track TuneDatums that are mapped to a design record
    tune_data_with_design_data = set()
    # track TuneDatums that are *not* mapped to a design record
    tune_data_without_design_data = set()
    # is_highband has no meaning for design data
    design_tune_data = design_data.tune_data_side_band_res_index
    # loop overall the data
    for tune_datum in tune_data:
        # pull these properties out to access the design data
        is_north = tune_datum.is_north
        is_highband = tune_datum.is_highband
        smurf_band = tune_datum.smurf_band
        res_index = tune_datum.res_index
        # see if there is design data available for this band-channel pair
        if is_north in design_tune_data.keys() and smurf_band in design_tune_data[is_north].keys() and \
                res_index in design_tune_data[is_north][smurf_band].keys():
            design_datum = design_tune_data[is_north][smurf_band][res_index]
            tune_datum_with_design_data = add_design_data(meas_datum=tune_datum, des_datum=design_datum,
                                                          design_attributes=design_attributes,
                                                          is_north=is_north)
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


def freq_rug_plot(freq_array_smurf, freq_array_design, f_array_adjusted=None, title=None):
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
    artists = []
    for xdata, color, y_min, y_max, label in [(freq_array_design, 'dodgerblue', 0.4, 1.0, '         Design'),
                                              (f_array_adjusted, 'darkgoldenrod', 0.2, 0.8, 'Output Measured'),
                                              (freq_array_smurf, 'firebrick', 0.0, 0.6, ' Input Measured')]:
        if xdata is not None:
            if len(xdata) < 400:
                linewidth = 2.0
                alpha = 0.5
            else:
                linewidth = 0.2
                alpha = 0.5
            for f_center in xdata:
                ax.plot((f_center, f_center), (y_min, y_max), ls='solid', linewidth=linewidth, color=color, alpha=alpha)
            artists.append(plt.Line2D(range(3), range(3), ls='solid', linewidth=3, color=color, alpha=alpha,
                                      label=label))
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
    ax.legend(handles=artists, loc=8)
    if title is not None:
        ax.set_title(title)

    plt.show(block=True)


def find_nearest_index(array, value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or np.fabs(value - array[idx-1]) < np.fabs(value - array[idx])):
        return idx - 1
    else:
        return idx


def find_nearest(array, value):
    return array[find_nearest_index(array=array, value=value)]


def find_nearest_design_for_measured(design, measured):
    nearest_design_for_measured = [find_nearest(array=design, value=meas) for meas in measured]
    return np.array(nearest_design_for_measured)


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


def pigeon_mapping_right(design, measured):
    # Two cases, an application of the pigeonhole principle
    # We only allow One measured resonator (pigeon) per designed resonator (a pigeonhole)
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
        # loop of the design frequencies, must map each measured.
        for meas_counter, (meas, nearest_des) in list(enumerate(measured_to_nearest_design)):
            design_to_possibly_skip = design[meas_counter + design_index_delta]
            # if skips are available, see if this design frequency should be skipped and make a better assignment
            while number_of_skips_available != design_index_delta:
                if design_to_possibly_skip < nearest_des:
                    # a skip is allowed if design_to_possibly_skip is less than the nearest_des for this meas
                    design_unmapped.append(design_to_possibly_skip)
                    design_index_delta += 1
                    design_to_possibly_skip = design[meas_counter + design_index_delta]
                else:
                    # if design_to_possibly_skip is equal to (or greater than) nearest_des, exit and assign
                    break
            design_to_measured_one_to_one[design_to_possibly_skip] = meas
        measured_overrun = []
    return design_to_measured_one_to_one, measured_overrun, design_unmapped


def pigeon_mapping_left(design, measured):
    diff = len(measured) - len(design)
    if diff >= 0:
        # There are more (or exactly equal number of) measured frequencies then available design frequencies
        design_to_measured_one_to_one = {des: meas for des, meas in zip(design, measured[diff:])}
        # if len(measured) == len(design) then this is an empty list
        measured_overrun = list(measured[:diff])
        design_unmapped = []
    else:
        design_right_transform = np.flip(design) * -1.0
        measured_right_transform = np.flip(measured) * -1.0
        design_to_measured_one_to_one_right_transform, measured_overrun_right_transform, design_unmapped_right_transform = \
            pigeon_mapping_right(design=design_right_transform, measured=measured_right_transform)
        design_to_measured_one_to_one = {design_right_key * -1.0:
                                         design_to_measured_one_to_one_right_transform[design_right_key] * -1.0
                                         for design_right_key in sorted(design_to_measured_one_to_one_right_transform.keys(), reverse=True)}
        measured_overrun = [meas * -1.0 for meas in reversed(measured_overrun_right_transform)]
        design_unmapped = [des * -1.0 for des in reversed(design_unmapped_right_transform)]

    return design_to_measured_one_to_one, measured_overrun, design_unmapped


class SimpleFreqDatum(NamedTuple):
    freq_mhz: float


minimize_kwargs = dict(method='Nelder-Mead', bounds=[(-50, 50), (0.95, 1.05)])


def res_refit_and_map(f_array_design: Union[list, tuple, np.array, dict],
                      adjusted_to_original: Union[list, tuple, np.array, dict],
                      trim_at_mhz: Union[None, float, int] = None,
                      show_plots: bool = False, verbose: bool = False, band_num: int = 0):
    if not isinstance(adjusted_to_original, dict):
        adjusted_to_original = {single_freq: single_freq for single_freq in adjusted_to_original}
    for adjusted_f in adjusted_to_original.keys():
        original_datum = adjusted_to_original[adjusted_f]
        if isinstance(original_datum, float):
            adjusted_to_original[adjusted_f] = SimpleFreqDatum(freq_mhz=adjusted_to_original[adjusted_f])

    # get the f_array from the sorted dictionary keys of the mapping of adjusted frequencies to the original frequencies
    f_array_adjusted = np.array(sorted(adjusted_to_original.keys()))

    """
    Trim the outlier data of a single smurf band and
    Shift the measured frequencies to be closer to the design frequencies.
    """
    if verbose:
        print(f'SMuRF-band:{band_num}  Making One-to-One refit and mapping')
    if trim_at_mhz is not None:
        # get distance in MHz to the nearest design frequency for each value of f_array_adjusted
        nearest_diff_values = nearest_diff(freq_array_smurf=f_array_adjusted, freq_array_design=f_array_design)
        # trim any outliers of f_array_adjusted that are more than trim_at_mhz from the nearest f_array_design value
        freq_array_trimmed = np.array([shifted_freq_mhz for shifted_freq_mhz, nearest_value_mhz
                                       in zip(f_array_adjusted, nearest_diff_values)
                                       if nearest_value_mhz <= trim_at_mhz])
        num_of_outliers = len(f_array_adjusted) - len(freq_array_trimmed)
        if num_of_outliers != 0:
            if verbose:
                print(f'  {num_of_outliers} outlier(s) are more than {trim_at_mhz} MHz from a design value.')
            # reset to the original frequencies use the trimmed data set
            opt_func2 = make_opt_func(freq_array_smurf=freq_array_trimmed, freq_array_design=f_array_design)
            opt_result = minimize(fun=opt_func2, x0=np.array([0.0, 1.0]), **minimize_kwargs)
            if opt_result.success:
                b_opt_this_band, m_opt_this_band = opt_result.x
            else:
                raise ValueError(f'No convergent Result for band {band_num} resonator mapping.')
            # shift the values based on the optimization, but use the array that still has the outliers
            smurf_processed = transform(measure=f_array_adjusted, b=b_opt_this_band, m=m_opt_this_band)
            if show_plots:
                freq_rug_plot(freq_array_smurf=freq_array_trimmed, freq_array_design=f_array_design,
                              f_array_adjusted=smurf_processed,
                              title=f"Trim Outlier Data and Reapply Minimization, Smurf-band {band_num}")
        else:
            if verbose:
                print(f'  No outliers that are more than {trim_at_mhz} MHz from a design value.')
            smurf_processed = f_array_adjusted
    else:
        smurf_processed = f_array_adjusted
    """
    Remove multiple measurements that map to a single design value 
    Starting from the center of the design array outward
    """
    # map the processed values (2 shifts with outliers add back in) to the original tune_datums
    smurf_processed_to_tune_datum = {freq_processed: adjusted_to_original[freq_shifted]
                                     for freq_processed, freq_shifted in
                                     zip(smurf_processed, f_array_adjusted)}
    # get a monotonic array of measured data
    process_measured_array = np.array(sorted([meas for meas in smurf_processed_to_tune_datum.keys()]))
    # Now find what are the nearest measured values is compared to the design array
    design_value_to_meas_index = {des: find_nearest_index(array=process_measured_array, value=des)
                                  for des in f_array_design}
    # get the half way index of the design to spit the data and to left and rights sides.
    if len(f_array_design) % 2 == 0:
        design_half_index = int(len(f_array_design) / 2)
    else:
        design_half_index = int((len(f_array_design) + 1) / 2)
    design_half_value = f_array_design[design_half_index]
    # the measurement index that corresponds to the design_half_index
    meas_half_index = design_value_to_meas_index[design_half_value]
    # split the arrays into left and right
    design_left = f_array_design[:design_half_index]
    meas_left = process_measured_array[:meas_half_index]
    design_right = f_array_design[design_half_index:]
    meas_right = process_measured_array[meas_half_index:]
    design_to_measured_one_to_one_left, measured_overrun_left, design_unmapped_left = \
        pigeon_mapping_left(design=design_left, measured=meas_left)
    design_to_measured_one_to_one_right, measured_overrun_right, design_unmapped_right = \
        pigeon_mapping_right(design=design_right, measured=meas_right)
    left_meas_sum = len(design_to_measured_one_to_one_left) + len(measured_overrun_left)
    right_meas_sum = len(design_to_measured_one_to_one_right) + len(measured_overrun_right)
    if left_meas_sum != len(meas_left):
        raise IndexError
    if right_meas_sum != len(meas_right):
        raise IndexError

    # put the one-to-one map together
    design_to_measured_one_to_one = {**design_to_measured_one_to_one_left,
                                     **design_to_measured_one_to_one_right}
    design_oto_array, measured_oto_array = zip(*[(des, design_to_measured_one_to_one[des]) for des
                                                 in design_to_measured_one_to_one.keys()])

    design_oto_list = list(design_oto_array)
    measured_oto_list = list(measured_oto_array)
    data_mapping_after_pigeon = design_oto_list + measured_overrun_left + measured_overrun_right
    if show_plots:
        freq_rug_plot(freq_array_smurf=smurf_processed, freq_array_design=f_array_design,
                      f_array_adjusted=data_mapping_after_pigeon,
                      title=f"One-to-One map from Pigeon Hole Mapping, Smurf-band {band_num}")
    """
    Heal unmapped data on the outside of the one-to-one mapped zones by
        filling the available un-mapped design points, if needed, and if possible.
    """
    meas_unmapped = measured_overrun_right + measured_overrun_left
    if meas_unmapped:
        if verbose:
            print(f'  right overrun: {measured_overrun_right}')
            print(f'   left overrun: {measured_overrun_left}')
            print('  Doing overrun healing...')

        # put the overrun measured data back into the solutions
        design_unmapped = design_unmapped_left + design_unmapped_right
        # start on the left side every other smurf band (for less biased algorithm behavior)
        do_left = bool(band_num % 2)
        while design_unmapped != [] and any([measured_overrun_right != [], measured_overrun_left != []]):
            if do_left:
                # check to see if there is still data on the left side
                if measured_overrun_left:
                    # get and remove left most un-mapped design point
                    left_design_unmapped = design_unmapped.pop(0)
                    # heal the left side overrun's right most point, get and remove it from measured_overrun_left
                    left_measure_overrun = measured_overrun_left.pop()
                    # consider this shifting the measured data by one to fill the first gap in design data.
                    # add the measured part to the beginning of the measured list, staying in monotonic order
                    measured_oto_list.insert(0, left_measure_overrun)
                    # insert the available design frequency into the one-to-one array to keep monotonic order
                    design_oto_list.insert(bisect(a=design_oto_list, x=left_design_unmapped), left_design_unmapped)
                # do the right side next loop
                do_left = False
            else:
                # check to see if there is data on the right side still
                if measured_overrun_right:
                    # get and remove left most un-mapped design point
                    right_design_unmapped = design_unmapped.pop()
                    # heal the right side overrun's left most point, get and remove it from measured_overrun_right
                    right_measure_overrun = measured_overrun_right.pop(0)
                    # effectively shift the measured array until the right most gap in the design data is filled
                    # add the measured part to the end of the measured list, staying in monotonic order
                    measured_oto_list.append(right_measure_overrun)
                    # insert the available design frequency into the one-to-one array to keep monotonic order
                    design_oto_list.insert(bisect(a=design_oto_list, x=right_design_unmapped),
                                           right_design_unmapped)
                # do the left side next loop
                do_left = True
        meas_unmapped = measured_overrun_right + measured_overrun_left
        if verbose:
            print('  Overrun healing complete.')
            if meas_unmapped:
                print(f' {len(meas_unmapped)} measured frequencies unmapped (more measured then design): ' +
                      f'{meas_unmapped}')
        if show_plots:
            freq_rug_plot(freq_array_smurf=data_mapping_after_pigeon, freq_array_design=f_array_design,
                          f_array_adjusted=design_oto_list + meas_unmapped,
                          title=f"One-to-One Final Healing to Map Overrun Edge Data, Smurf-band {band_num}")
    return measured_oto_list, design_oto_list, smurf_processed_to_tune_datum, meas_unmapped


def map_by_freq(tune_data_side_band_res_index, design_attributes, design_data,
                ignore_smurf_neg_one=False, trim_at_mhz: Union[None, float, int] = 10.0,
                show_plots=False, verbose=True,
                design_random_remove=False):
    # make a new set to hold the tune data that is updated with design data
    tune_data_new = set()
    # track TuneDatums that are mapped to a design record
    tune_data_with_design_data = set()
    # track TuneDatums that are *not* mapped to a design record
    tune_data_without_design_data = set()
    # one set of design frequencies is all that is needed for both sides of the UFM
    design_tune_data = design_data.tune_data_side_band_res_index
    design_tune_data_band_res_index = {}
    for is_north in design_tune_data.keys():
        design_this_side = {band_num: design_tune_data[is_north][band_num]
                            for band_num in design_tune_data[is_north].keys()}
        design_tune_data_band_res_index.update(design_this_side)

    freq_array_design, tune_data_by_freq_design = \
        smurf_band_res_index_to_freq(tune_data_band_index=design_tune_data_band_res_index)
    # loop over both sides of the UFM
    for is_north in tune_data_side_band_res_index.keys():
        """
        For entire Readout chain (all High or all Low SMuRF bands) and
        Shift the measured frequencies to be closer to the design frequencies.
        """
        # only consider the tune data this side of the UFM
        tune_data_this_side = tune_data_side_band_res_index[is_north]
        # get the tunes as a frequency array, and a dictionary with freq_mhz as a key to the tune_datums
        freq_array_smurf, tune_data_by_freq_smurf = \
            smurf_band_res_index_to_freq(tune_data_band_index=tune_data_this_side, ignore_neg_one=ignore_smurf_neg_one)
        # optimize the resonator to match across the entire side of a single UFM
        opt_func = make_opt_func(freq_array_smurf, freq_array_design)
        opt_result = minimize(fun=opt_func, x0=np.array(list((0.0, 1.0))), **minimize_kwargs)
        if opt_result.success:
            b_opt, m_opt = opt_result.x
            shifted_smurf_optimal = transform(measure=freq_array_smurf, b=b_opt, m=m_opt)
            if show_plots:
                freq_rug_plot(freq_array_smurf=freq_array_smurf, freq_array_design=freq_array_design,
                              f_array_adjusted=shifted_smurf_optimal,
                              title=f'Single SMuRF Measured Resonator Rescale to Design')
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
            """
            For a single smurf band and
            Shift the measured frequencies to be closer to the design frequencies.
            """
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
            shifted_to_original = {freq_shifted: tune_datum for freq_shifted, tune_datum
                                   in zip(freq_array_smurf_shifted, tune_data_this_band_f_ordered)}

            # get the design and look up dict
            freq_array_design_this_band, design_to_design_datums = \
                f_array_and_tune_dict(tune_data_dict=design_tune_data_band_res_index[smurf_band])

            # a test to force there to be over-mapped design frequencies
            if design_random_remove:
                print('Random missing design frequencies test - to generate over populated measurements arrays')
                # randomly remove some design data for a test
                freq_array_design_this_band_list = list(freq_array_design_this_band)
                for remove_loop_index in range(2):
                    freq_array_design_this_band_list.pop(randint(a=0, b=len(freq_array_design_this_band_list) - 1))

                freq_array_design_this_band = np.array(freq_array_design_this_band_list)
                print('Random missing design frequencies test - to generate over populated measurements arrays')

            if show_plots:
                freq_rug_plot(freq_array_smurf=freq_array_smurf, freq_array_design=freq_array_design_this_band,
                              f_array_adjusted=freq_array_smurf_shifted,
                              title=f'Previous All-SMuRF Minimization as Applied to smurf-band {smurf_band}')
            # Run the single-band mapping strategy
            measured_oto_list, design_oto_list, smurf_processed_to_tune_datum, meas_unmapped = \
                res_refit_and_map(f_array_design=freq_array_design_this_band,
                                  adjusted_to_original=shifted_to_original,
                                  trim_at_mhz=trim_at_mhz,
                                  show_plots=show_plots, verbose=verbose, band_num=smurf_band)

            """
            Record this solution in the standard format.
            """
            # Track the unmapped measured resonators
            for unmapped_meas_datum in [smurf_processed_to_tune_datum[meas] for meas in meas_unmapped]:
                # if there is no available design data then we pass the unchanged tune_datum to the new set
                tune_data_new.add(unmapped_meas_datum)
                # we also track what datums were not able to matched with design data.
                tune_data_without_design_data.add(unmapped_meas_datum)
            # Create new tune_datum from the design datum for the mapped resonators
            for meas, des in zip(measured_oto_list, design_oto_list):
                meas_datum = smurf_processed_to_tune_datum[meas]
                des_datum = design_to_design_datums[des]
                tune_datum_with_design_data = add_design_data(meas_datum=meas_datum, des_datum=des_datum,
                                                              design_attributes=design_attributes, is_north=is_north)
                # add this to the new data set
                tune_data_new.add(tune_datum_with_design_data)
                # track which datums have designs info
                tune_data_with_design_data.add(tune_datum_with_design_data)
    return tune_data_new, tune_data_with_design_data, tune_data_without_design_data
