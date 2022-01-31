"""
A Developmental Environment for Testing New features.
"""
import os
from sodetlib.detmap.example.download_example_data import sample_data_init
from sodetlib.detmap.detmap_config import get_config, abs_path_sample_data_default, abs_path_metadata_files_default, \
    abs_path_detmap
from sodetlib.detmap.makemap import make_map_smurf, psat_map, make_map_g3_timestream


sample_data_init(del_dir=False)
config = get_config()
for_jack = True
abstracted = False


tune_data_smurf = make_map_smurf(tunefile=config['tunefile'], north_is_highband=config['north_is_highband'],
                                 mapping_strategy='map_by_freq')
psat_map(tune_data=tune_data_smurf, cold_ramp_file=config['cold_ramp_file'],
         temp_k=config['psat_temp_k'], show_plot=config['psat_show_plot'], save_plot=config['psat_save_plot'])
