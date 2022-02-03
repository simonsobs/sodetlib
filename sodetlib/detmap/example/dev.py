"""
A Developmental Environment for Testing New features.
"""
from sodetlib.detmap.example.download_example_data import sample_data_init
from sodetlib.detmap.detmap_config import get_config
from sodetlib.detmap.makemap import make_map_smurf, psat_map


sample_data_init(del_dir=False)
config = get_config()


tune_data_smurf = make_map_smurf(tunefile=config['tunefile'], north_is_highband=config['north_is_highband'],
                                 mapping_strategy='map_by_freq')
tune_data_smurf_res_index = make_map_smurf(tunefile=config['tunefile'], north_is_highband=config['north_is_highband'],
                                           mapping_strategy='map_by_res_index')
psat_map(tune_data=tune_data_smurf, cold_ramp_file=config['cold_ramp_file'],
         temp_k=config['psat_temp_k'], show_plot=config['psat_show_plot'], save_plot=config['psat_save_plot'])
