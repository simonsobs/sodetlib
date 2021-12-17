"""
An example configuration file for the detector mapping algorithms.
"""


import os
import glob
import pathlib

from scratch.chw3k5.detmap.download_example_data import sample_data_init

# Check to see if the example data is available, if not it downloads it from a GoogleDrive host.
sample_data_init()
# get the absolute path for the configuration files
abs_path_config_files = pathlib.Path(__file__).parent.resolve()
# get the absolute path fo the detector mapping code
abs_path_detmap = abs_path_config_files.parent.resolve()
# set the absolute paths for the example and metadata directories.
abs_path_metadata = os.path.join(abs_path_detmap, 'metadata')
abs_path_sample_data = os.path.join(abs_path_detmap, 'sample_data')

"""
start the example configuration file, this is only a first try, most of this should be determined dynamically.
"""
north_is_highband = False  # use None for no highband, True for North as the highband, False for South as the highband
vna_shift_mhz = 10  # in MHz
waferfile = os.path.join(abs_path_metadata, "copper_map_corrected.csv")
design_file = os.path.join(abs_path_metadata, "umux_32_map.pkl")
bands = range(8)
# dict_thru = {"N": [7], "S": []}  # deprecated
dark_bias_lines = [4, 5, 6, 7, 8, 9, 10, 11]  # If certain sides are covered
tunefile = os.path.join(abs_path_sample_data, '1632247315_tune.npy')

# mux position number (int in 0-27) to mux band number (int in 0-14) mapping file
mux_pos_num_to_mux_band_num_path = os.path.join(abs_path_config_files, 'mux_pos_num_to_mux_band_num.csv')

# cold ramp bath temperature sweep filename
cold_ramp_file = os.path.join(abs_path_config_files, 'coldloadramp_example.csv')

# vna data paths
dir_N = os.path.join(abs_path_sample_data, 'north_vna')
dir_S = os.path.join(abs_path_sample_data, 'south_vna')
north_search_str = os.path.join(dir_N, '*.S2P')
N_band = sorted(glob.glob(north_search_str))
tune_data_vna_output_filename = os.path.join(abs_path_sample_data, 'tune_data_vna.csv')
redo_vna_tune = False  # when true the peak finder is rerun, even if the resulting output csv already exists.

south_search_str = os.path.join(dir_S, '*.S2P')
S_band = sorted(glob.glob(south_search_str))

# hard cord sorting needs to fixed
N_files = N_band[3:-1]
S_files = S_band[2:]

# output smurf data filename
output_filename = os.path.join(abs_path_sample_data, "test_pixel_info.csv")

# output vna data filename
output_filename_vna = os.path.join(abs_path_sample_data, "test_pixel_info_vna.csv")
