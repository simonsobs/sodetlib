import argparse
from sodetlib.det_config import DetConfig
from sodetlib.legacy.smurf_funcs import det_ops
from sodetlib.legacy.analysis import det_analysis


def main(S, cfg, biasgroups, voltage, freq):
    kwargs = {}
    if voltage is not None:
        kwargs['tickle_voltage'] = voltage
    if freq is not None:
        kwargs['tickle_freq'] = freq
    run_file = det_ops.take_tickle(S, cfg, biasgroups, **kwargs)
    det_analysis.analyze_tickle_data(S, run_file)


if __name__ == '__main__':
    cfg = DetConfig()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--biasgroups', '--bgs', nargs='+', type=int, default=list(range(12)),
        help="Bias groups to run tickle and analysis")
    parser.add_argument('--tickle-voltage', '--tv', type=float,
                        help='Tickle voltage')
    parser.add_argument('--tickle-freq', '--tf', type=float,
                        help='Tickle frequency (Hz)')
    args = cfg.parse_args(parser)
    S = cfg.get_smurf_control(dump_configs=True)
    main(S, cfg, args.biasgroups, args.tickle_voltage, args.tickle_freq)

