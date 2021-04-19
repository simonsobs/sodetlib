import numpy as np
import profile_band
import os


bands = np.array([2,3])
epics_root = 'smurf_server_s5'
config_file = '/usr/local/src/pysmurf/cfg_files/stanford/experiment_fp31_cc03-02_lbOnlyBay0.cfg'
shelf_manager = 'shm-smrf-sp01'
loopback = True

output_dir = '/data/smurf_data/20200406/'

print(f'Config file is {config_file}')

while True:
    # Loop over bands
    for band in bands:
        html_path = profile_band.run(band, epics_root, config_file,
            shelf_manager, True, loopback=loopback,
            no_find_freq=False,
                                     no_setup_notches=False, no_band_off=True, subband_low=52, subband_high=460)
        f = open(os.path.join(output_dir, f'profile_band{band}.txt'), 'ab')
        np.savetxt(f, [html_path], fmt='%s')
        f.close()

