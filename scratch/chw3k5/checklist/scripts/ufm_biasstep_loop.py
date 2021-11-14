"""
The core code is from Princeton, created by Yuhan Wang and Daniel Dutcher Oct/Nov 2021.

The code was refactored by Caleb Wheeler Nov 2021 to support argparse.
The argparse implementation (found at the bottom of the file) and provides a minimal
documentation framework.

Use:

python3 uxm_bath_iv_noise.py -h

to see the available options and required formatting.
"""
import csv
import time

import numpy as np
import matplotlib

matplotlib.use('Agg')


def verbose_print(msg, verbose=True):
    if verbose:
        verbose_print(f"{prefix_str} {msg}", verbose)
    return


def ufm_biasstep_loop(S, bias_high, bath_temp, out_fn, stream_time=120, verbose=True):
    fieldnames = ['bath_temp', 'bias_v', 'band', 'data_path', 'step_size']
    with open(out_fn, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

    over_bias_sleep_time = 300
    verbose_print(f'overbias_tes_all and sleeping for {over_bias_sleep_time} seconds', verbose)
    S.overbias_tes_all(bias_groups=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], overbias_wait=1, tes_bias=12, cool_wait=3,
                       high_current_mode=False, overbias_voltage=12)
    time.sleep(over_bias_sleep_time)

    step_array = np.arange(bias_high, 0, -0.2)
    step_size = 0.01
    for bias_voltage_step in step_array:

        bias_voltage = bias_voltage_step
        S.set_tes_bias_bipolar_array(
            [bias_voltage, bias_voltage, bias_voltage, bias_voltage, bias_voltage, bias_voltage, bias_voltage,
             bias_voltage, bias_voltage, bias_voltage, bias_voltage, bias_voltage, 0., 0., 0.])
        verbose_print(f'Streaming data for {stream_time} seconds, for bias_voltage_step {bias_voltage_step}', verbose)
        time.sleep(stream_time)

        dat_path = S.stream_data_on()
        for k in [0, 1, 2, 3, 4]:
            bias_voltage = bias_voltage_step
            S.set_tes_bias_bipolar_array(
                [bias_voltage, bias_voltage, bias_voltage, bias_voltage, bias_voltage, bias_voltage, bias_voltage,
                 bias_voltage, bias_voltage, bias_voltage, bias_voltage, bias_voltage, 0., 0., 0.])
            time.sleep(2)
            bias_voltage = bias_voltage_step - step_size
            S.set_tes_bias_bipolar_array(
                [bias_voltage, bias_voltage, bias_voltage, bias_voltage, bias_voltage, bias_voltage, bias_voltage,
                 bias_voltage, bias_voltage, bias_voltage, bias_voltage, bias_voltage, 0., 0., 0.])
            time.sleep(2)
        S.stream_data_off()
        row = {}
        row['bath_temp'] = bath_temp
        row['bias_v'] = bias_voltage_step
        row['band'] = 'all'
        row['data_path'] = dat_path
        row['step_size'] = 0.01

        with open(out_fn, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow(row)
    return


prefix_str = f'\n From {ufm_biasstep_loop.__name__} '

if __name__ == '__main__':
    import argparse
    from sodetlib.det_config import DetConfig

    """
    The code below will only run if the file is run directly, but not if elements from this file are imported.
    For example:
        python3 ufm_biasstep_loop.py -args_for_argparse
    will have __name__ == '__main__' as True, and the code below will run locally.
    """
    # Seed a new parse for this file with the parser from the SMuRF controller class
    cfg = DetConfig()

    # set up the parser for this Script
    parser = argparse.ArgumentParser(description='Parser for noise_stack_by_band_new.py script.')
    """
    Cannot find confirmation about how these variables are going being set. 
    """
    parser.add_argument('bias-high', type=float, default='None',
                        help='The High Bias Voltage.')

    # optional arguments
    parser.add_argument('--bath-temp', dest='bath_temp', type=float, default=100.0,
                        help='The Bath Temperature in mK')
    parser.add_argument('--output-file', dest='output_file', type=str, default='None',
                        help='The full path for the output file name.')

    """
    ^^^^^^^^^^^^^^
    Cannot find confirmation about how these variables are going being set. 
    """

    parser.add_argument('--stream-time', dest='stream_time', type=float, default=120.0,
                        help="float, optional, default is 120.0. The amount of time to sleep in seconds while " +
                             "streaming SMuRF data for analysis.")
    parser.add_argument('--verbose', dest='verbose', action='store_true',
                        help="Turns on printed output from the script. The default is --verbose." +
                             "--no-verbose has minimal (or no) print statements.")
    parser.add_argument('--no-verbose', dest='verbose', action='store_false', default=True,
                        help="Turns off printed output from the script. The default is --verbose." +
                             "--no-verbose has minimal (or no) print statements.")

    # parse the args for this script
    args = cfg.parse_args(parser)
    S = cfg.get_smurf_control(dump_configs=True)

    # run the def in this file
    ufm_biasstep_loop(S=S, bias_high=args.bias_high, bath_temp=args.bath_temp, out_fn=args.output_file,
                            stream_time=args.stream_time, verbose=args.verbose)
