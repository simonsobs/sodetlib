import os
import numpy as np
from sodetlib.detmap.makemap import MapMaker
from sodetlib.detmap.meta_select import abs_path_detmap
from sodetlib.detmap.example.download_example_data import sample_data_init


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


# if you like to work with rectangular data topologies, it is easy to cast the data into an iterable like a list
data_list = list(mv6_smurf_map)
# and then into a numpy array
data_array = np.array(data_list)
