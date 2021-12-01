"""
An example configuration file for the detector mapping algorithms.
"""


import os
import glob

import numpy as np

from scratch.chw3k5.detmap.download_example_data import sample_data_init

# Check to see if the example data is available, if not it downloads it from a GoogleDrive host.
sample_data_init()


# start the example configuration file, this is only a first try, most of this should be determined dynamically.
highband = "S"
shift = 10
waferfile = os.path.join("../metadata", "copper_map_corrected.csv")
design_file = os.path.join("../metadata", "umux_32_map.pkl")
bands = np.arange(8)
dict_thru = {"N": [7], "S": []}
dark_bias_lines = [4, 5, 6, 7, 8, 9, 10, 11]  # If certain sides are covered
smurf_tune = os.path.join('../sample_data', 'sample_data/1632247315_tune.npy')

# mux position number (int in 0-27) to mux band number (int in 0-14) mapping file
mux_pos_num_to_mux_band_num_path = os.path.join('../sample_data', 'sample_data/mux_pos_num_to_mux_band_num.csv')

dir_N = os.path.join('../sample_data', 'sample_data/north_vna')
dir_S = os.path.join('../sample_data', 'sample_data/south_vna')
north_search_str = os.path.join(dir_N, '*.S2P')
N_band = sorted(glob.glob(north_search_str))

south_search_str = os.path.join(dir_S, '*.S2P')
S_band = sorted(glob.glob(south_search_str))

# hard cord sorting needs to fixed
N_files = N_band[3:-1]
S_files = S_band[2:]
