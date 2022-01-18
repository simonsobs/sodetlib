import os
from sodetlib.detmap.makemap import psat_map
from sodetlib.detmap.detmap_conifg import get_config, abs_path_detmap, abs_path_metadata_files_default


if __name__ == '__main__':
    config = get_config(config_ymal_path=os.path.join(abs_path_detmap, 'example', 'example.yaml'),
                        metadata_dir=abs_path_metadata_files_default,
                        metadata_waferfile="copper_map_corrected.csv",
                        metadata_designfile="umux_32_map.pkl",
                        metadata_mux_pos_to_mux_band_file='mux_pos_num_to_mux_band_num.csv')

    #
    psat_map(tunefile=config['tunefile'], north_is_highband=config['north_is_highband'],
             output_filename_smurf=config['output_filename_smurf'],
             cold_ramp_file=config['cold_ramp_file'],
             design_file=config['design_file_path'], waferfile=config['waferfile_path'],
             mux_pos_num_to_mux_band_num=config['mux_pos_num_to_mux_band_num_path'],
             dark_bias_lines=config['dark_bias_lines'],
             psat_temp_k=config['psat_temp_k'], psat_show_plot=config['psat_show_plot'],
             psat_save_plot=config['psat_save_plot'])
