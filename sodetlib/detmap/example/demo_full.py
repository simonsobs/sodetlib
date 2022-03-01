import os
import numpy as np
import matplotlib.pyplot as plt
from sodetlib.detmap.makemap import MapMaker
from sodetlib.detmap.simple_csv import read_csv
from sodetlib.detmap.meta_select import abs_path_detmap
from sodetlib.detmap.example.download_example_data import sample_data_init

"""
Data Analysis Demonstration
"""
dir_this_file = os.path.dirname(os.path.realpath(__file__))
# options for all record types.
output_parent_dir = os.path.join(dir_this_file, 'output')
do_csv_output = True
overwrite_csv_output = False
show_layout_plot = False
save_layout_plot = True
overwrite_plot = False
mapping_strategy = 'map_by_freq'
verbose = True

# options for psat plot (Mv6) example only.
psat_temp_k = 9.0
psat_show_plot = False
psat_save_plot = True

# where the sample data dir is located on the local machine
abs_path_sample_data = os.path.join(abs_path_detmap, 'sample_data')
sample_data_init(del_dir=False)

mv6_record = dict(north_is_highband=False, array_name='Mv6',
                  tunefile_path=os.path.join(abs_path_sample_data, '1632247315_tune.npy'),
                  dark_bias_lines=[4, 5, 6, 7, 8, 9, 10, 11],
                  vna_shift_mhz=10.0,
                  tune_data_vna_intermediate_filename=os.path.join(abs_path_sample_data, 'tune_data_vna.csv'),
                  path_north_side_vna=os.path.join(abs_path_sample_data, 'north_side_vna_farray.csv'),
                  path_south_side_vna=os.path.join(abs_path_sample_data, 'south_side_vna_farray.csv'),
                  cold_ramp_file=os.path.join(abs_path_sample_data, 'coldloadramp_example.csv'))

sv5_record = dict(north_is_highband=True, array_name='Sv5',
                  timestream_path=os.path.join(abs_path_sample_data, 'freq_map.npy'),
                  tunefile_path=os.path.join(abs_path_sample_data, '1619048156_tune.npy'))

cv4_record = dict(north_is_highband=False, array_name='cv4',
                  tunefile_path=os.path.join(abs_path_sample_data, '1628891659_tune.npy'))


mv6_map_maker = MapMaker(north_is_highband=mv6_record['north_is_highband'],
                         array_name=mv6_record['array_name'],
                         mapping_strategy=mapping_strategy, dark_bias_lines=mv6_record['dark_bias_lines'],
                         do_csv_output=do_csv_output, overwrite_csv_output=overwrite_csv_output,
                         show_layout_plot=show_layout_plot, save_layout_plot=save_layout_plot,
                         overwrite_plot=overwrite_plot,
                         output_parent_dir=output_parent_dir,
                         verbose=verbose)

mv6_smurf_map = mv6_map_maker.make_map_smurf(tunefile=mv6_record['tunefile_path'])
mv6_map_maker.psat_map(tune_data=mv6_smurf_map, cold_ramp_file=mv6_record['cold_ramp_file'],
                       temp_k=psat_temp_k, show_plot=psat_show_plot, save_plot=psat_save_plot)
mv6_vna_map = mv6_map_maker.make_map_vna(tune_data_vna_intermediate_filename=mv6_record['tune_data_vna_intermediate_filename'],
                                         path_north_side_vna=mv6_record['path_north_side_vna'],
                                         path_south_side_vna=mv6_record['path_south_side_vna'],
                                         shift_mhz=mv6_record['vna_shift_mhz'])


sv5_map_maker = MapMaker(north_is_highband=sv5_record['north_is_highband'],
                         array_name=sv5_record['array_name'],
                         mapping_strategy=mapping_strategy, dark_bias_lines=None,
                         do_csv_output=do_csv_output, overwrite_csv_output=overwrite_csv_output,
                         show_layout_plot=show_layout_plot, save_layout_plot=save_layout_plot,
                         overwrite_plot=overwrite_plot,
                         output_parent_dir=output_parent_dir,
                         verbose=verbose)
sv5_smurf_map = sv5_map_maker.make_map_smurf(tunefile=sv5_record['tunefile_path'])
sv5_g3_map = sv5_map_maker.make_map_g3_timestream(timestream=sv5_record['timestream_path'])


cv4_map_maker = MapMaker(north_is_highband=cv4_record['north_is_highband'],
                         array_name=cv4_record['array_name'],
                         mapping_strategy=mapping_strategy, dark_bias_lines=None,
                         do_csv_output=do_csv_output, overwrite_csv_output=overwrite_csv_output,
                         show_layout_plot=show_layout_plot, save_layout_plot=save_layout_plot,
                         overwrite_plot=overwrite_plot,
                         output_parent_dir=output_parent_dir,
                         verbose=verbose)
cv4_smurf_map = cv4_map_maker.make_map_smurf(tunefile=cv4_record['tunefile_path'])


"""
Data Interation from Analysis classes and from Output CSV files
"""
# # # if you like to work with rectangular data topologies, it is easy to cast the data into an iterable like a list
data_list = list(mv6_smurf_map)
# and then into a numpy array
data_array = np.array(data_list)

# # # Users can also skip the analysis and simply populate the OperateTuneData() class from an output file.
# use os.path.join() to create directory paths that also work on Windows.
dir_mv6_smurf_results = os.path.join(output_parent_dir, 'smurf_Mv6')
dir_mv6_smurf_plot = os.path.join(dir_mv6_smurf_results, 'plot')
filename_mv6_smurf_results = 'smurf_Mv6_map_by_freq.csv'
path_sv5_smurf_results = os.path.join(dir_mv6_smurf_results, filename_mv6_smurf_results)
# use the output data to plot a results
mv6_map_maker_from_output = MapMaker(output_parent_dir=output_parent_dir, array_name='mv6',
                                     from_output_csv=path_sv5_smurf_results)
operate_tune_data_mv6 = mv6_map_maker_from_output.load_from_output_csv()
operate_tune_data_mv6.plot_with_layout(plot_dir=dir_mv6_smurf_plot,
                                       overwrite_plot=True, show_plot=True, save_plot=False)

# # # If you do not care to use the method in the OperateTuneData() class, you can simply read in the output CSV
# to simply read-in the output data and it over it, use the csv reader inside this module
data_by_column, data_by_row = read_csv(path=path_sv5_smurf_results)

# # # is a simple example of plotting using the read-in output CSV
# use the iterated data to make your own filters
bandpass_selection = 150  # integer (GHz) either 90 or 150
polarization_selection = 'A'  # options are 'A', 'B', 'D' for 90 GHz 'A' or 'B' for 150 GHz

det_x_all_not_connect_pixels = []
det_y_all_not_connect_pixels = []
det_x_single_bandpass_and_pol = []
det_y_single_bandpass_and_pol = []
for single_data_row in data_by_row:
    if single_data_row['bandpass'] == bandpass_selection and single_data_row['pol'] == polarization_selection:
        det_x_single_bandpass_and_pol.append(single_data_row['det_x'])
        det_y_single_bandpass_and_pol.append(single_data_row['det_y'])
    if single_data_row['bandpass'] == 'NC':
        det_x_all_not_connect_pixels.append(single_data_row['det_x'])
        det_y_all_not_connect_pixels.append(single_data_row['det_y'])

# use the iterated data to make you own plot
# figure coordinates
left = 0.05
bottom = 0.08
right = 0.99
top = 0.95

# define the figure size in inches
fig = plt.figure(figsize=(6, 6))

# set the axes boundaries in terms of figure coordinates
coord_ax = [left, bottom, right-left, top - bottom]
# make a new axes for plotting in this figure
ax = fig.add_axes(coord_ax, frameon=False)

# plot the data
ax.plot(data_by_column['det_x'], data_by_column['det_y'], ls='None', marker='o', color='darkgoldenrod', label="All data", markersize=8)
ax.plot(det_x_all_not_connect_pixels, det_y_all_not_connect_pixels, ls='None', marker='x', color='firebrick', label="Not Connected",  markersize=9)
ax.plot(det_x_single_bandpass_and_pol, det_y_single_bandpass_and_pol, ls='None', marker='^', color='dodgerblue',  markersize=5,
        label=f"{bandpass_selection} GHz, pol: {polarization_selection}")
# add a title for the whole plot
fig.suptitle(f"Bandpass of {bandpass_selection} with polarization of '{polarization_selection}'")
# add a legend
ax.legend(loc='best', fontsize=11)
# show the figure
plt.show()
