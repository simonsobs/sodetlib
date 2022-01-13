"""
The Configuration file for the detector mapping algorithms.

Reads in a YAML file, see example/example.yaml for an example for both SMuRF and VNA data.

"""
import os
import yaml


# use None to run use the example.yaml file and download the sample data set
config_ymal_path = None

# get the absolute path fo the detector mapping code
real_path_this_file = os.path.dirname(os.path.realpath(__file__))
abs_path_sodetlib, _ = real_path_this_file.rsplit("sodetlib", 1)
abs_path_detmap = os.path.join(abs_path_sodetlib, "sodetlib", "detmap")

# get the absolute dir path for the configuration files
abs_path_metadata_files = os.path.join(abs_path_detmap, 'meta')
# where the sample data dir is located on the local machine
abs_path_sample_data = os.path.join(abs_path_detmap, 'sample_data')

# get the absolute path for each metadata file
waferfile_path = os.path.join(abs_path_metadata_files, "copper_map_corrected.csv")
design_file_path = os.path.join(abs_path_metadata_files, "umux_32_map.pkl")
mux_pos_num_to_mux_band_num_path = os.path.join(abs_path_metadata_files, 'mux_pos_num_to_mux_band_num.csv')

if config_ymal_path is None:
    from sodetlib.detmap.example.download_example_data import sample_data_init
    # Check to see if the example data is available, if not it downloads it from a GoogleDrive host.
    sample_data_init()
    #  set the path to the example configuration yaml
    config_ymal_path = os.path.join(abs_path_detmap, 'example', 'example.yaml')

# read the ymal file
with open(config_ymal_path) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

possible_file_names_in_config = {'tunefile', 'cold_ramp_file', 'output_filename_smurf',
                                 'path_north_side_vna', 'path_south_side_vna',
                                 'tune_data_vna_output_filename',  'output_filename_vna'}
for file_name_key in set(config.keys()) & possible_file_names_in_config:
    config[file_name_key] = os.path.join(abs_path_sample_data, config[file_name_key])

for key, var_name in [('waferfile_path', waferfile_path), ('design_file_path', design_file_path),
                      ('mux_pos_num_to_mux_band_num_path', mux_pos_num_to_mux_band_num_path)]:
    config[key] = var_name
