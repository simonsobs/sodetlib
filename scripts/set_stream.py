"""
This is a utility script for the OCS pysmurf controller which starts/stops
data streaming, or takes data for some duration.
"""

import matplotlib
matplotlib.use('Agg')
import argparse
import sodetlib as sdl

from sodetlib.det_config import DetConfig

if __name__ == '__main__':
    cfg = DetConfig()

    parser = argparse.ArgumentParser()
    parser.add_argument('state', choices=['on', 'off'])
    parser.add_argument('--duration', '-d', type=float)
    parser.add_argument('--emulator', action='store_true')
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--epics-root', type=str, default=None)
    parser.add_argument('--skip-freq-mask', action='store_true')
    parser.add_argument('--apply-dev-cfg', action='store_true')
    args = cfg.parse_args(parser)

    S = cfg.get_smurf_control(
        dump_configs=args.dump_configs, epics_root=args.epics_root,
        apply_dev_configs=args.apply_dev_cfg
    )

    if args.state == 'on':
        stream_kw = {
            'emulator': args.emulator,
            'tag': args.tag,
            'make_freq_mask': not args.skip_freq_mask,
        }
        if args.duration is not None:
            sid = sdl.take_g3_data(S, args.duration, **stream_kw)
        else:
            sid = sdl.stream_g3_on(S, **stream_kw)
    else:
        sdl.stream_g3_off(S, emulator=args.emulator)
