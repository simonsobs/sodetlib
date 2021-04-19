import numpy as np
import matplotlib
from pprint import pprint
import argparse

matplotlib.use('agg')
from sodetlib.det_config import DetConfig

parser = argparse.ArgumentParser(description='Stream Data for some amount of time')
parser.add_argument('state', choices=['on', 'off'],
            help='Toggle Data Streaming')
parser.add_argument('--time', type=int, help='Number of seconds to stream data')

if __name__ == '__main__':
    args = parser.parse_args()
    
    stream_finite = False
    '''
    if args.on is not None:
        start_stream = True
    elif args.off is not None:
        stop_stream = true
    elif args.time is None:
        raise ValueError('You must supply a time to stream data if not using --on or --off')
    else:
        stream_finite = True
    '''	
    cfg = DetConfig()
    cfg.load_config_files(slot=2)
    S = cfg.get_smurf_control(dump_configs=True)

    if args.state == 'off':
        S.stream_data_off()
    else:
        S.pA_per_phi0 = 9e6
        S.R_sh=400e-6
        S.bias_line_resistance=16400.0

        if args.time is not None:
            S.take_stream_data(args.time, return_data=True, register_file=True)
        else:
            S.stream_data_on()
