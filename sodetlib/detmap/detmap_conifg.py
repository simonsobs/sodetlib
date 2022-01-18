"""
The Configuration file for the detector mapping algorithms.

Reads in a YAML file, see example/example.yaml for an example for both SMuRF and VNA data.

"""
import os
import yaml
from sodetlib.detmap.example.download_example_data import sample_data_init


# get the absolute path fo the detector mapping code
real_path_this_file = os.path.dirname(os.path.realpath(__file__))
abs_path_sodetlib, _ = real_path_this_file.rsplit("sodetlib", 1)
abs_path_detmap = os.path.join(abs_path_sodetlib, "sodetlib", "detmap")

# get the absolute dir path for the configuration files
abs_path_metadata_files_default = os.path.join(abs_path_detmap, 'meta')
# where the sample data dir is located on the local machine
abs_path_sample_data_default = os.path.join(abs_path_detmap, 'sample_data')
# keys that that hold filename data for a given configuration.
possible_file_names_in_config = {'tunefile', 'cold_ramp_file', 'output_filename_smurf',
                                 'path_north_side_vna', 'path_south_side_vna',
                                 'tune_data_vna_output_filename', 'output_filename_vna'}

metadata_waferfile_default = "copper_map_corrected.csv"
metadata_designfile_default = "umux_32_map.pkl"
metadata_mux_pos_to_mux_band_file_default = 'mux_pos_num_to_mux_band_num.csv'

waferfile_default_path = os.path.join(abs_path_metadata_files_default, metadata_waferfile_default)
designfile_default_path = os.path.join(abs_path_metadata_files_default, metadata_designfile_default)
mux_pos_to_mux_band_file_default_path = os.path.join(abs_path_metadata_files_default,
                                                     metadata_mux_pos_to_mux_band_file_default)

output_csv_default_filename = 'pixel_freq_mapping.csv'


def get_config(config_ymal_path=os.path.join(abs_path_detmap, 'example', 'example.yaml'),
               metadata_dir=abs_path_metadata_files_default,
               metadata_waferfile=metadata_waferfile_default,
               metadata_designfile=metadata_designfile_default,
               metadata_mux_pos_to_mux_band_file=metadata_mux_pos_to_mux_band_file_default):
    # read the ymal file with configuration data
    with open(config_ymal_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # set the path for each metadata file
    waferfile_path = os.path.join(metadata_dir, metadata_waferfile)
    design_file_path = os.path.join(metadata_dir, metadata_designfile)
    mux_pos_num_to_mux_band_num_path = os.path.join(metadata_dir, metadata_mux_pos_to_mux_band_file)

    # the tunefile and psat data directory
    if config['data_dir'] is None or config['data_dir'].lower().strip() == 'none':
        data_dir = abs_path_sample_data_default
        # Check to see if the example data is available, if not it downloads it from a GoogleDrive host.
        sample_data_init()
    else:
        # converts the '/' to '\\' in Windows, make code platform independent.
        folders_list = config['data_dir'].split('/')
        data_dir = ''
        for single_folder in folders_list:
            data_dir = os.path.join(data_dir, single_folder)

    # set the full path for the file names in the configuration yaml file
    for file_name_key in set(config.keys()) & possible_file_names_in_config:
        config[file_name_key] = os.path.join(data_dir, config[file_name_key])
    # set the metadata full file paths.
    for key, var_name in [('waferfile_path', waferfile_path), ('design_file_path', design_file_path),
                          ('mux_pos_num_to_mux_band_num_path', mux_pos_num_to_mux_band_num_path)]:
        config[key] = var_name

    return config
