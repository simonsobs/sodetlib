import matplotlib
matplotlib.use('Agg')

import pysmurf
import argparse
import numpy as np
import os
import time




if __name__=='__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--setup', action='store_true')
    parser.add_argument('--config-file', required=True)
    parser.add_argument('--epics-root', default='smurf_server_s2')

    parser.add_argument('--out-file')

    parser.add_argument('--band', type=int, default=2)
    
    args = parser.parse_args()
    
    S = pysmurf.SmurfControl(
            epics_root = args.epics_root,
            cfg_file = args.config_file,
            setup = args.setup,make_logfile=False
    )

    print("Plots in director: ",S.plot_dir)

    #Initialize the band for our mux chip
    band = args.band

    
    #Start stream
    datafile = S.stream_data_on()
    time.sleep(10)
    #Bias step up
    S.set_tes_bias_bipolar(0,0.05)
    time.sleep(10)
    #Bias step down
    S.set_tes_bias_bipolar(0,0)
    time.sleep(10)
    #Stop stream
    S.stream_data_off()

    with open(args.out_file, 'a') as f:
        f.write(f'{datafile}\n')

    


