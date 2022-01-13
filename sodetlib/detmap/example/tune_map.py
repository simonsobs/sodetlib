from sodetlib.detmap.makemap import psat_map
from sodetlib.detmap.detmap_conifg import config


psat_map(tunefile=config['tunefile'], north_is_highband=config['north_is_highband'],
         output_filename_smurf=config['output_filename_smurf'],
         cold_ramp_file=config['cold_ramp_file'],
         design_file=config['design_file_path'], waferfile=config['waferfile_path'],
         mux_pos_num_to_mux_band_num=config['mux_pos_num_to_mux_band_num_path'],
         dark_bias_lines=config['dark_bias_lines'],
         psat_temp_k=config['psat_temp_k'], psat_show_plot=config['psat_show_plot'],
         psat_save_plot=config['psat_save_plot'])
