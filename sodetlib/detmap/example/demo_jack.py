import os

from sodetlib.detmap.detmap_config import abs_path_metadata_files_default, abs_path_detmap
from sodetlib.detmap.makemap import make_map_smurf
from sodetlib.detmap.channel_assignment import OperateTuneData
from sodetlib.detmap.layout_data import get_layout_data
from sodetlib.detmap.detmap_config import abs_path_detmap, get_config


abstracted = False
tunefile = os.path.join(abs_path_detmap, 'dev_data', '1619048156_tune.npy')


if abstracted:
    tune_data_smurf = make_map_smurf(tunefile=tunefile,
                                     north_is_highband=True)
else:
    config = get_config(output_data_dir=os.path.join(abs_path_detmap, 'output'))
    meta_dir = os.path.join(abs_path_detmap, 'meta')

    design_path = os.path.join(meta_dir, 'umux_32_map.pkl')
    wafer_path = os.path.join(meta_dir, 'UFM_Si_corrected.csv')
    layout_path = os.path.join(meta_dir, 'mux_pos_num_to_mux_band_num.csv')

    smurf_map = OperateTuneData(tune_path=tunefile, north_is_highband=True)
    design_map = OperateTuneData(design_file_path=design_path)
    layout_map = get_layout_data(wafer_path)

    smurf_map.map_design_data(design_map, layout_position_path=layout_path, mapping_strategy='map_by_freq')
    smurf_map.map_layout_data(layout_data=layout_map)
    smurf_map.write_csv(output_path_csv=config['output_filename_smurf'])
    smurf_map.plot_with_layout(plot_path=config['layout_plot_filename_smurf'],
                               show_plot=False, save_plot=True)
