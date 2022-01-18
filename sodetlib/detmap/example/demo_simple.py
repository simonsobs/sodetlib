"""
A basic demonstration of creating a Mapping CSV file.
"""
import os
from sodetlib.detmap.example.download_example_data import sample_data_init
from sodetlib.detmap.detmap_conifg import abs_path_sample_data_default
from sodetlib.detmap.makemap import make_map_g3_timestream, make_map_smurf


sample_data_init(del_dir=False)

make_map_smurf(tunefile=os.path.join(abs_path_sample_data_default, '1632247315_tune.npy'), north_is_highband=False)
make_map_g3_timestream(timestream=os.path.join(abs_path_sample_data_default, 'freq_map.npy'), north_is_highband=False)
