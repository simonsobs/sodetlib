"""
A basic demonstration of creating a Mapping CSV file.
"""
import os
import numpy as np
from sodetlib.detmap.example.download_example_data import sample_data_init
from sodetlib.detmap.detmap_config import abs_path_sample_data_default
from sodetlib.detmap.makemap import make_map_g3_timestream, make_map_smurf


sample_data_init(del_dir=False)

tune_data_smurf = make_map_smurf(tunefile=os.path.join(abs_path_sample_data_default, '1632247315_tune.npy'), north_is_highband=False)
tune_data_g3 = make_map_g3_timestream(timestream=os.path.join(abs_path_sample_data_default, 'freq_map.npy'), north_is_highband=False)

# if you like to work with rectangular data topologies, it is easy to cast the data into an iterable like a list
data_list = list(tune_data_smurf)
# and then into a numpy array
data_array = np.array(data_list)