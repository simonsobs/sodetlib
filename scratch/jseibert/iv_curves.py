import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pysmurf.client
import argparse
import numpy as np
import os
import time
import glob
from scipy import signal
import scipy.optimize as opt
pi = np.pi


if __name__=='__main__':    
    parser = argparse.ArgumentParser()

    # Arguments that are needed to create Pysmurf object
    parser.add_argument('--setup', action='store_true')
    parser.add_argument('--config-file', required=True)
    parser.add_argument('--epics-root', default='smurf_server_s2')

    # Custom arguments for this script
    parser.add_argument('-bg','--bias-groups', type=int, default=None,nargs = '+',
                        help='Desired bias group, defaults to whatever is specified in the config file')

    parser.add_argument('--wait-time',type = float, default = 0.1,
                        help = 'Wait time between steps. Default is 0.1')

    parser.add_argument('--bias-high',type = float, default = 19.9,
                        help = 'Maximum TES bias in volts. Default is 19.9')

    parser.add_argument('--bias-low',type = float, default = 0.0,
                        help = 'Maximum TES bias in volts. Default is 0.0')   

    parser.add_argument('--bias-step',type = float, default = 0.005,
                        help = 'Step size in volts. Default is 0.005')   

    parser.add_argument('--overbias-wait',type = float, default = 2.0,
                        help = 'Time to stay overbiased in seconds. Default is 2.0')

    parser.add_argument('--cool-wait',type = float, default = 30.0,
                        help = 'Time to stay in low current state after overbiasing in seconds. Default is 30.0')              

    parser.add_argument('--overbias',type = float, default = 19.9,
                        help = 'Overbias voltage, default is 19.9')

    parser.add_argument('-hcm','--high-current-mode',action='store_true', 
                        help='The current mode to take IVs in')

    parser.add_argument('--out-file')


    # Parse command line arguments
    args = parser.parse_args()


    bias_groups = args.bias_groups

    wait_time = args.wait_time

    bias_high = args.bias_high
    bias_low = args.bias_low
    bias_step = args.bias_step
    
    overbias_wait = args.overbias_wait
    cool_wait = args.cool_wait
    overbias_volt = args.overbias 

    high_current_mode = args.high_current_mode

    S = pysmurf.client.SmurfControl(
            epics_root = args.epics_root,
            cfg_file = args.config_file,
            setup = args.setup,make_logfile=False
    )

    iv_file = S.run_iv(bias_groups=bias_groups, wait_time=wait_time, bias=None,
               bias_high=bias_high, bias_low=bias_low, bias_step=bias_step,
               show_plot=False, overbias_wait=overbias_wait, cool_wait=cool_wait,
               make_plot=True, save_plot=True, plotname_append='',
               channels=None, band=None, high_current_mode=high_current_mode,
               overbias_voltage=overbias_volt, grid_on=True,
               phase_excursion_min=0.1, bias_line_resistance=None, do_analysis = True)
    '''
    with open(args.out_file, 'a') as fname:
        fname.write(f'plotdir : {S.plot_dir}, outdir : {S.output_dir}, Rsh : {S.R_sh}, bias_line_resistance : {S.bias_line_resistance}, high_low_ratio : {S.high_low_current_ratio}, pA_per_phi0 : {S.pA_per_phi0}, iv_file : {iv_file}\n')
    '''

    iv_info = {}
    iv_info['plot_dir'] = S.plot_dir
    iv_info['output_dir'] = S.output_dir
    iv_info['Rsh'] = S.R_sh
    iv_info['bias_line_resistance'] = S.bias_line_resistance
    iv_info['high_low_ratio'] = S.high_low_current_ratio
    iv_info['pA_per_phi0'] = S.pA_per_phi0
    iv_info['iv_file'] = iv_file
    fp = os.path.join(S.output_dir,f'{S.get_timestamp()}_iv_info.npy')
    S.log(f'Writing IV information to {fp}.')
    np.save(fp,iv_info)   
    S.pub.register_file(fp, 'iv_info',format='npy')
 
    # need to report bias_line_resistance and R_sh that were used when the IV was taken
    # and pA per phi0
    # and high_low_current_ratio

    # should have option to not run IV, and just analyze an IV instead

    # script logic: 
    # always be tuned up first! and know which bias groups you want
    # do you want to run IV curves? yes/no
    # is that all you wanted? yes/no
    # do you want to analyze your IVs? yes/no
    # do you want to set a bias point? yes/no
        # pick an average Rfrac bias point for the bias group
    # advanced: pick an optimal bias point, do this with the responsivity
        # need to think about this
        # are we doing this with R_op as defined in detector parameters doc?
