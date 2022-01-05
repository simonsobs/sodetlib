"""
An example configuration file for the detector mapping algorithms.
"""


import os
import pathlib

from download_example_data import sample_data_init

# Check to see if the example data is available, if not it downloads it from a GoogleDrive host.
sample_data_init()

# get the absolute path fo the detector mapping code
abs_path_detmap = pathlib.Path(__file__).parent.resolve()
# set the absolute paths for the example and metadata directories.
abs_path_sample_data = os.path.join(abs_path_detmap, 'sample_data')
# get the absolute path for the configuration files
abs_path_config_files = os.path.join(abs_path_detmap, 'config_files')

"""
start the example configuration file, this is only a first try, most of this should be determined dynamically.
"""
north_is_highband = False  # use None for no highband, True for North as the highband, False for South as the highband
waferfile = os.path.join(abs_path_config_files, "copper_map_corrected.csv")
design_file = os.path.join(abs_path_config_files, "umux_32_map.pkl")
bands = range(8)
dark_bias_lines = [4, 5, 6, 7, 8, 9, 10, 11]  # If certain sides are covered
tunefile = os.path.join(abs_path_sample_data, '1632247315_tune.npy')

# mux position number (int in 0-27) to mux band number (int in 0-14) mapping file
mux_pos_num_to_mux_band_num_path = os.path.join(abs_path_config_files, 'mux_pos_num_to_mux_band_num.csv')

# cold ramp bath temperature sweep filename
cold_ramp_file = os.path.join(abs_path_sample_data, 'coldloadramp_example.csv')

# vna data
vna_shift_mhz = 10.0  # in MHz, applies a shift to the VNA data when determining the smurf band and resonator index
path_north_side_vna = os.path.join(abs_path_sample_data, 'north_side_vna_farray.csv')
path_south_side_vna = os.path.join(abs_path_sample_data, 'south_side_vna_farray.csv')
# below is an intermediate data product with assigned smurf bands
tune_data_vna_output_filename = os.path.join(abs_path_sample_data, 'tune_data_vna.csv')

# output smurf data filename
output_filename = os.path.join(abs_path_sample_data, "test_pixel_info.csv")

# output vna data filename
output_filename_vna = os.path.join(abs_path_sample_data, "test_pixel_info_vna.csv")
