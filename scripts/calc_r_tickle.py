import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pysmurf.client
import argparse
import numpy as np
import pickle as pkl
from scipy import signal
import os

from sodetlib.det_config import DetConfig
from sodetlib.smurf_funcs import take_tickle, analyze_tickle

if __name__=='__main__':
	cfg = DetConfig()
	parser = argparse.ArgumentParser()

	parser.add_argument('--band', type=int, required=True,
				help='band (must be in range [0,7])')

	parser.add_argument('--biasgroup', type = int, nargs = '+', required=True,
			help='bias group that you want to run tickles on')

	parser.add_argument('--tickle-voltage', type=float, default = 0.1,
			help='Amplitude (not peak-peak) of your tickle in volts')

	parser.add_argument('--high-current', action = 'store_true')

	parser.add_argument('--over-bias',action = 'store_true')

	parser.add_argument('--channels', type=int, nargs = '+', default = None,
				help='Channels that you want to calculate the tickle response of')

	parser.add_argument('--make-channel-plots', action = 'store_true')

	# Parse command line arguments
	args = cfg.parse_args(parser)

	S = cfg.get_smurf_control(dump_configs=True)

	#Put your script calls here
	if args.channels == 'None':
		channels = None
	else:
		channels = args.channels
	data_file, cur_dc = take_tickle(S, band = args.band, bias_group=args.biasgroup,
			tickle_voltage = args.tickle_voltage, high_current = args.high_current,
			over_bias = args.over_bias)
	analyze_tickle(S, band = args.band, data_file = data_file, dc_level = cur_dc, 
			tickle_voltage = args.tickle_voltage, high_current = args.high_current, 
			channels = channels, make_channel_plots = args.make_channel_plots)
