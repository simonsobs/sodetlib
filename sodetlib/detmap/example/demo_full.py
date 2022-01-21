import os
import numpy as np
from sodetlib.detmap.makemap import make_map_smurf, make_map_vna, psat_map
from sodetlib.detmap.detmap_config import get_config, abs_path_detmap, abs_path_metadata_files_default


if __name__ == '__main__':
    config = get_config(config_ymal_path=os.path.join(abs_path_detmap, 'example', 'example.yaml'),
                        metadata_dir=abs_path_metadata_files_default,
                        metadata_waferfile="copper_map_corrected.csv",
                        metadata_designfile="umux_32_map.pkl",
                        metadata_mux_pos_to_mux_band_file='mux_pos_num_to_mux_band_num.csv')

    # read the tunefile and initialize the data instance for SMuRF tunefile
    tune_data_smurf = make_map_smurf(tunefile=config['tunefile'], north_is_highband=config['north_is_highband'],
                                     design_file=config['design_file_path'], waferfile=config['waferfile_path'],
                                     layout_position_path=config['mux_pos_num_to_mux_band_num_path'],
                                     dark_bias_lines=config['dark_bias_lines'],
                                     output_path_csv=config['output_filename_smurf'])
    # read the tunefile and initialize the data instance for SMuRF tunefile
    tune_data_vna = make_map_vna(tune_data_vna_output_filename=config['tune_data_vna_output_filename'],
                                 north_is_highband=config['north_is_highband'],
                                 path_north_side_vna=config['path_north_side_vna'],
                                 path_south_side_vna=config['path_south_side_vna'],
                                 shift_mhz=config['vna_shift_mhz'],
                                 design_file=config['design_file_path'], waferfile=config['waferfile_path'],
                                 layout_position_path=config['mux_pos_num_to_mux_band_num_path'],
                                 dark_bias_lines=config['dark_bias_lines'],
                                 output_path_csv=config['output_filename_vna'])

    # make the psat color maps
    psat_map(tune_data=tune_data_smurf, cold_ramp_file=config['cold_ramp_file'],
             temp_k=9.0, show_plot=False, save_plot=True)

    # if you like to work with rectangular data topologies, it is easy to cast the data into an iterable like a list
    data_list = list(tune_data_smurf)
    # and then into a numpy array
    data_array = np.array(data_list)
