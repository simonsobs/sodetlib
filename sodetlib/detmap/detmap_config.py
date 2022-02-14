"""
The Configuration file for the detector mapping algorithms.

Reads in a YAML file, see example/example.yaml for an example for both SMuRF and VNA data.

"""
import os
import sys
import yaml
from sodetlib.detmap.meta_select import get_metadata_files, abs_path_detmap, abs_path_metadata_files_default
from sodetlib.detmap.example.download_example_data import sample_data_init


# where the sample data dir is located on the local machine
abs_path_sample_data_default = os.path.join(abs_path_detmap, 'sample_data')
# keys that that hold filename data for a given configuration.
possible_input_filenames_in_config = {'tunefile', 'timestream', 'cold_ramp_file',
                                      'output_filename_smurf',
                                      'path_north_side_vna', 'path_south_side_vna'}
possible_output_filenames_in_config = {'tune_data_vna_output_filename', 'output_filename_vna',
                                       'output_filename_g3'}
possible_plot_filenames_in_config = {'layout_plot_filename_smurf', 'layout_plot_filename_vna',
                                     'layout_plot_filename_g3'}



def dir_name_join(path):
    # converts the '/' to '\\' in Windows, make code platform independent.
    folders_list = path.split('/')
    new_path = ''
    for single_folder in folders_list:
        if single_folder == '' and sys.platform != 'win32':
            new_path += '/'
        else:
            new_path = os.path.join(new_path, single_folder)
    return new_path


def get_config(array_name=None,
               config_ymal_path=os.path.join(abs_path_detmap, 'example', 'example.yaml'),
               metadata_dir=abs_path_metadata_files_default,
               output_data_dir=None):
    # read the ymal file with configuration data
    with open(config_ymal_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        waferfile_path, designfile_path, mux_pos_to_mux_band_file_path = \
            get_metadata_files(array_name=array_name, abs_path_metadata_files=metadata_dir, verbose=True)

    # the tunefile and psat data directory
    if config['data_dir'] is None or config['data_dir'].lower().strip() == 'none':
        data_dir = abs_path_sample_data_default
        # Check to see if the example data is available, if not it downloads it from a GoogleDrive host.
        sample_data_init()
    else:
        data_dir = dir_name_join(path=config['data_dir'])
    # the output data directory
    if output_data_dir is None:
        output_data_dir = data_dir
    output_data_dir = dir_name_join(path=output_data_dir)
    if not os.path.exists(output_data_dir):
        os.mkdir(output_data_dir)

    # set the full path for the file names in the configuration yaml file
    for file_name_key in set(config.keys()) & possible_input_filenames_in_config:
        config[file_name_key] = os.path.join(data_dir, config[file_name_key])
    for file_name_key in set(config.keys()) & possible_output_filenames_in_config:
        config[file_name_key] = os.path.join(output_data_dir, config[file_name_key])
    for file_name_key in set(config.keys()) & possible_plot_filenames_in_config:
        config[file_name_key] = os.path.join(output_data_dir, 'plots', config[file_name_key])
    # set the metadata full file paths.
    for key, var_name in [('waferfile_path', waferfile_path), ('design_file_path', designfile_path),
                          ('mux_pos_num_to_mux_band_num_path', mux_pos_to_mux_band_file_path)]:
        config[key] = var_name

    return config


if __name__ == '__main__':
    config = get_config()
