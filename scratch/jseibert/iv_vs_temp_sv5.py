# updating IV_vs_temp to replicate the way things are done at Pton a little more 
# for ufm-sv5 measurements in sat-1
# Joe Seibert
# 7/20/21

import matplotlib
matplotlib.use('Agg')

import argparse
import numpy as np

from sodetlib.det_config  import DetConfig
import numpy as np

from sodetlib.smurf_funcs.det_ops import take_iv


if __name__=='__main__':
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--out-file')
    parser.add_argument('--temp',type=float)
    
    # Initialize pysmurf object
    cfg = DetConfig()
    cfg.load_config_files(slot=2)
    S = cfg.get_smurf_control()
    
    #hard-coding this for now
    S.load_tune('/data/smurf_data/tune/1626836367_tune.npy')
    S.R_sh = 0.0004

    # Parse args
    args = parser.parse_args()
    
    bands = np.arange(8)
    bgs = np.arange(12)

    temp = args.temp
    outfile = args.out_file

    if temp == 0.095:
        output_dict = {}
    else:
        output_dict = np.load(outfile, allow_pickle=True).item()
    output_dict[f'{temp} mK'] = {}
    
    # Take IVs on all bias groups
    iv_info = take_iv(S, cfg, 
                      bias_groups = bgs, 
                      wait_time=0.001, 
                      bias_high=16, 
                      bias_low=0, 
                      bias_step = 0.025, 
                      overbias_voltage=18, 
                      cool_wait=150, 
                      high_current_mode=False,
                      cool_voltage = 8)

    output_dict[f'{temp} mK']['all_bgs'] = iv_info
    np.save(outfile, output_dict)

    # Take IVs on one bias group at a time and save output
    for bg in bgs:
        iv_info = take_iv(S, cfg, 
                          bias_groups = bg, 
                          wait_time=0.001, 
                          bias_high=16, 
                         bias_low=0, 
                         bias_step = 0.025, 
                         overbias_voltage=18, 
                         cool_wait=150, 
                         high_current_mode=False,
                         cool_voltage = 8)

        output_dict[f'{temp} mK'][f'{bg}'] = iv_info     
        np.save(outfile, output_dict)      


    
