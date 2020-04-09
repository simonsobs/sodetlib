import matplotlib
matplotlib.use('Agg')

import pysmurf.client
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
    parser.add_argument('--flag')

    parser.add_argument('--band', type=int, default=2)
    parser.add_argument('--bias-group',type=int,default=1)
    parser.add_argument('--ob-volt',type=float,default=10)
    parser.add_argument('--step-volt',type=float,default=0.05)
    
    args = parser.parse_args()
    
    S = pysmurf.client.SmurfControl(
            epics_root = args.epics_root,
            cfg_file = args.config_file,
            setup = args.setup,make_logfile=False
    )

    print("Plots in director: ",S.plot_dir)

    #Initialize the band for our mux chip
    band = args.band
    flag = args.flag
    bias_group = args.bias_group
    ob_amp = args.ob_volt
    step_size = args.step_volt

    #serial_gradient algos for resonators moving w/ temp
    S.run_serial_gradient_descent(band)
    S.run_serial_eta_scan(band)
    
    #Start stream
    if flag == 'start':
        datafile = S.stream_data_on()
        S.set_tes_bias_high_current(bias_group=bias_group,write_log=True)
        S.set_tes_bias_bipolar(bias_group,ob_amp)

    time.sleep(10)
    #Bias step up
    S.set_tes_bias_bipolar(bias_group,ob_amp+step_size)
    time.sleep(10)
    #Bias step down
    S.set_tes_bias_bipolar(bias_group,ob_amp)
    time.sleep(10)
    
    #Stop stream
    if flag == 'stop':
        S.stream_data_off()
        with open(args.out_file, 'a') as f:
            f.write(f'{datafile}\n')

    


