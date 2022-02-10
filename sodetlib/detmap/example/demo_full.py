import os
import numpy as np
from sodetlib.detmap.makemap import make_map_smurf, make_map_vna, make_map_g3_timestream, psat_map
from sodetlib.detmap.detmap_config import get_config, abs_path_detmap


config = get_config(array_name='Mv6',
                    config_ymal_path=os.path.join(abs_path_detmap, 'example', 'example.yaml'),
                    output_data_dir=os.path.join(abs_path_detmap, 'output'))

# read the tunefile and initialize the data instance for SMuRF tunefile
tune_data_smurf = make_map_smurf(tunefile=config['tunefile'],
                                 north_is_highband=config['north_is_highband'],
                                 design_file=config['design_file_path'],
                                 waferfile=config['waferfile_path'],
                                 layout_position_path=config['mux_pos_num_to_mux_band_num_path'],
                                 dark_bias_lines=config['dark_bias_lines'],
                                 output_path_csv=config['output_filename_smurf'],
                                 layout_plot_path=config['layout_plot_filename_smurf'],
                                 do_csv_output=config['do_csv_output'],
                                 save_layout_plot=config['save_layout_plot'],
                                 show_layout_plot=config['show_layout_plot'],
                                 mapping_strategy=config['mapping_strategy'])

# read the tunefile and initialize the data instance for SMuRF tunefile
tune_data_vna = make_map_vna(tune_data_vna_output_filename=config['tune_data_vna_output_filename'],
                             north_is_highband=config['north_is_highband'],
                             path_north_side_vna=config['path_north_side_vna'],
                             path_south_side_vna=config['path_south_side_vna'],
                             shift_mhz=config['vna_shift_mhz'],
                             design_file=config['design_file_path'], waferfile=config['waferfile_path'],
                             layout_position_path=config['mux_pos_num_to_mux_band_num_path'],
                             dark_bias_lines=config['dark_bias_lines'],
                             output_path_csv=config['output_filename_vna'],
                             layout_plot_path=config['layout_plot_filename_vna'],
                             do_csv_output=config['do_csv_output'],
                             save_layout_plot=config['save_layout_plot'],
                             show_layout_plot=config['show_layout_plot'],
                             mapping_strategy=config['mapping_strategy'])

config = get_config(array_name='Sv5',
                    config_ymal_path=os.path.join(abs_path_detmap, 'example', 'example.yaml'),
                    output_data_dir=os.path.join(abs_path_detmap, 'output'))

tune_data_g3 = make_map_g3_timestream(timestream=config['timestream'],
                                      north_is_highband=config['north_is_highband'],
                                      design_file=config['design_file_path'], waferfile=config['waferfile_path'],
                                      layout_position_path=config['mux_pos_num_to_mux_band_num_path'],
                                      dark_bias_lines=config['dark_bias_lines'],
                                      output_path_csv=config['output_filename_g3'],
                                      layout_plot_path=config['layout_plot_filename_g3'],
                                      do_csv_output=config['do_csv_output'],
                                      save_layout_plot=config['save_layout_plot'],
                                      show_layout_plot=config['show_layout_plot'],
                                      mapping_strategy=config['mapping_strategy'])

# make the psat color maps
psat_map(tune_data=tune_data_smurf, cold_ramp_file=config['cold_ramp_file'],
         temp_k=config['psat_temp_k'], show_plot=config['psat_show_plot'], save_plot=config['psat_save_plot'])

# if you like to work with rectangular data topologies, it is easy to cast the data into an iterable like a list
data_list = list(tune_data_smurf)
# and then into a numpy array
data_array = np.array(data_list)
