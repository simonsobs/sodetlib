import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pysmurf.client
import argparse
import numpy as np

from sodetlib.det_config import DetConfig
import sodetlib.smurf_funcs.health_check as hc
import sodetlib.smurf_funcs.smurf_ops as so
import sodetlib.smurf_funcs.optimize_params as op
from sodetlib.smurf_funcs import tickle
from sodetlib.analysis.tickle import analyze_tickle


if __name__=='__main__':
    cfg = DetConfig()
    parser = argparse.ArgumentParser()

    #Required argparse arguments
    parser.add_argument('--biasgroup', type = int, nargs = '+', required=True,
    			         help='bias group that you want to run tickles on')

    #Typically just use default values for argparse arguments
    parser.add_argument('--BW-target', '-m', type=float, default = 500,
                        help='Target readout bandwidth to optimize lms_gain')

    parser.add_argument('--wait-time', type=float,default=0.1,
                        help='Time to wait between flux steps in seconds.')

    parser.add_argument('--Npts',type=int,default = 3,
                        help='Number of points to average')

    parser.add_argument('--NPhi0s',type=int, default=4,
                        help='Number of periods in your squid curve.')

    parser.add_argument('--Nsteps',type=int,default=500,
                        help='Number of points in your squid curve.')

    parser.add_argument('--relock',action = 'store_true',
                        help='If specified will run relock.')

	parser.add_argument('--tickle-voltage', type=float, default = 0.1,
			help='Amplitude (not peak-peak) of your tickle in volts')

	parser.add_argument('--high-current', action = 'store_true')

	parser.add_argument('--over-bias',action = 'store_true')

	parser.add_argument('--channels', type=int, nargs = '+', default = None,
				help='Channels that you want to calculate the tickle response of')

	parser.add_argument('--make-channel-plots', action = 'store_true')

	parser.add_argument('--R-threshold',default = 100,
				help = 'Resistance threshold for determining detector channel')

    args = cfg.parse_args(parser)
    S = cfg.get_smurf_control(dump_configs=True)

    #This is dumb and potentially unneccessary, should figure out right way.
    if args.channels == 'None':
    	channels = None
    else:
    	channels = args.channels

    #Turns on amps and adjust/returns "optimal bias" and then does a few system
    #health checks.
    print('Running system health check.')
    hc.health_check(S,cfg)
    #Next find which bands and subbands have resonators attached
    print('Identifying active bands and subbands.')
    bands, subband_dict = so.find_subbands(S,cfg)
    #Now tune on those bands/find_subbands
    print('Tuning')
    num_chans_tune, tune_file = so.find_and_tune_freq(S,cfg,bands)
    #Now setup tracking
    optimize_dict = {}
    for band in bands:
        optimize_dict[band] = {}
        print(f'Optimizing tracking for band {band}')
        optimize_dict[band]['lms_freq_opt'], optimize_dict[band]['frac_pp'],
            optimize_dict[band]['bad_track_chans'],
            optimize_dict[band]['params'] = op.optimize_tracking(S, cfg, band)
        print(f'UC attenuator for band {band}')
        optimize_dict[band]['min_median'], optimize_dict[band]['min_atten'],
            optimize_dict[band]['drive']= op.optimize_power_per_band(S,cfg,band)
        print(f'Optimizing lms_gain for band {band}')
        optimize_dict[band]['opt_lms_gain'],optimize_dict[band]['lms_gain_dict'] =
            op.optimize_lms_gain(S,cfg,band,BW_target = args.BW_target)
        #Right now lms_gain doesn't set you to that after completion...we need
        #to add this
    print('Taking and analyzing optimized noise')
    datfile = S.take_stream_data(20)
    for band in bands:
        optimize_dict[band]['median_noise'], optimize_dict[band]['noise_dict'] =
            op.analyze_noise_psd(S,band,datfile)
    #take_squid_open_loop doesn't return the filepath to the data which you
    #need to run the fitting script, need ot update that.
    print('Taking DC SQUID Curves')
    raw_fr_data = so.take_squid_open_loop(S, cfg, bands,
                    wait_time = args.wait_time, Npts = args.Npts,
                    NPhi0s = args.NPhi0s, Nsteps = args. Nsteps,
                    relock = args.relock)
    #Right now not a good way to call the fitting stuff since its its own script
    #should we call this script to execute that one, not demo it here, or
    #convert the fitting script into importable functions?
    print('Identifying channels w/ detectors and calculating resistance.')
    #Need to add a function that identifies which biasgroups are connected so
    #that we don't need to pass a biasgroup argument.
    tickle_files = {}
    for band in bands:
        tickle_files[band] ,cur_dc = tickle.take_tickle(S, band = args.band,
                bias_group=args.biasgroup, tickle_voltage = args.tickle_voltage,
                high_current = args.high_current, over_bias = args.over_bias)
        optimize_dict[band]['tickle_dict'] = analyze_tickle(S, band = args.band, 
                    data_file = tickle_files[band], dc_level = cur_dc,
                    tickle_voltage = args.tickle_voltage,
                    high_current = args.high_current, channels = channels,
                    make_channel_plots = args.make_channel_plot,
                    R_threshold = args.R_threshold)
