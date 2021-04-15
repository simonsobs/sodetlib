"""
This is a utility script for the OCS pysmurf controller which starts/stops
data streaming, or takes data for some duration.
"""

import matplotlib
matplotlib.use('Agg')
import argparse

from sodetlib.det_config import DetConfig

if __name__ == '__main__':
    cfg = DetConfig()

    parser = argparse.ArgumentParser()
    parser.add_argument('state', choices=['on', 'off'])
    parser.add_argument('--duration', '-d', type=float)
    args = cfg.parse_args(parser)

    S = cfg.get_smurf_control(dump_configs=True)

    if args.state == 'on':
        if args.duration is not None:
            datfile = S.take_stream_data(args.duration)
        else:
            datfile = S.stream_data_on()
    else:
        S.stream_data_off()



