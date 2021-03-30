import numpy as np
import matplotlib
from pprint import pprint
import argparse

matplotlib.use('agg')
from sodetlib.det_config import DetConfig

parser = argparse.ArgumentParser(description='Stream Data for some amount of time')
parser.add_argument('--time', type=int, help='Number of seconds to stream data')

if __name__ == '__main__':
	args = parser.parse_args()
	if args.time is None:
		raise ValueError('You must supply a time to stream data')	
	
	cfg = DetConfig()
	cfg.load_config_files(slot=2)
	S = cfg.get_smurf_control(dump_configs=True, make_logfile=True)

	S.pA_per_phi0 = 9e6
	S.R_sh=400e-6
	S.bias_line_resistance=16400.0

	#band = 3
	#S.load_tune('/data/smurf_data/tune/1615914879_tune.npy')
	#S.relock(band)
	#for _ in range(3):
	 #   S.run_serial_gradient_descent(band=band)
	  #  S.run_serial_eta_scan(band=band)

	S.take_stream_data(args.time, return_data=True, register_file=True)

