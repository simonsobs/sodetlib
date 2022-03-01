import os

from sodetlib.detmap.makemap import MapMaker
from sodetlib.detmap.layout_data import get_layout_data
from sodetlib.detmap.channel_assignment import OperateTuneData
from sodetlib.detmap.example.download_example_data import sample_data_init
from sodetlib.detmap.meta_select import abs_path_detmap, get_metadata_files


abstracted = False
dir_this_file = os.path.dirname(os.path.realpath(__file__))
output_parent_dir = os.path.join(dir_this_file, 'output')
do_csv_output = True
overwrite_csv_output = False
show_layout_plot = False
save_layout_plot = True
overwrite_plot = False
mapping_strategy = 'map_by_freq'
verbose = True

# where the sample data dir is located on the local machine
abs_path_sample_data = os.path.join(abs_path_detmap, 'sample_data')
sample_data_init(del_dir=False)


sv5_record = dict(north_is_highband=True, array_name='Sv5',
                  timestream_path=os.path.join(abs_path_sample_data, 'freq_map.npy'),
                  tunefile_path=os.path.join(abs_path_sample_data, '1619048156_tune.npy'))

if abstracted:
    sv5_map_maker = MapMaker(north_is_highband=sv5_record['north_is_highband'],
                             array_name=sv5_record['array_name'],
                             mapping_strategy=mapping_strategy, dark_bias_lines=None,
                             do_csv_output=do_csv_output, overwrite_csv_output=overwrite_csv_output,
                             show_layout_plot=show_layout_plot, save_layout_plot=save_layout_plot,
                             overwrite_plot=overwrite_plot,
                             output_parent_dir=output_parent_dir,
                             verbose=verbose)
    sv5_smurf_map = sv5_map_maker.make_map_smurf(tunefile=sv5_record['tunefile_path'])
else:
    # Here we deal with the file naming structure which is handled above by the MapMaker() class
    output_dir_jack_data = os.path.join(output_parent_dir, 'jack_output')
    if not os.path.exists(output_dir_jack_data):
        os.mkdir(output_dir_jack_data)
    plot_dir = os.path.join(output_dir_jack_data, 'plot')
    if not os.path.exists(plot_dir):
        os.mkdir(plot_dir)
    output_path_csv = os.path.join(output_dir_jack_data, 'sv4_jack_output_pixel_mapping.csv')
    filename_pre_fix_for_layout_plot = 'sv4_layout_jack'
    # The deconstructed example for Jack to hack the code in the detmap submodule
    wafer_path, design_path, layout_path = get_metadata_files(array_name=sv5_record['array_name'], verbose=verbose)
    smurf_map = OperateTuneData(tune_path=sv5_record['tunefile_path'],
                                north_is_highband=sv5_record['north_is_highband'])
    design_map = OperateTuneData(design_file_path=design_path,  layout_position_path=layout_path,
                                 north_is_highband=sv5_record['north_is_highband'])
    layout_map = get_layout_data(wafer_path)

    smurf_map.map_design_data(design_map, mapping_strategy='map_by_freq')

    smurf_map.map_layout_data(layout_data=layout_map)
    smurf_map.write_csv(output_path_csv=output_path_csv)
    smurf_map.plot_with_layout(plot_dir=plot_dir, plot_filename=filename_pre_fix_for_layout_plot,
                               overwrite_plot=overwrite_plot,
                               show_plot=show_layout_plot, save_plot=save_layout_plot)
