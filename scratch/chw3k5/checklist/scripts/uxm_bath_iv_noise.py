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
        print(f"{prefix_str} {msg}")
    return


def uxm_bath_iv_noise(S, cfg, bands, bath_temp, out_fn, stream_time=120.0, verbose=False):
    for band in bands:
        S.run_serial_gradient_descent(band)
        S.run_serial_eta_scan(band)
        S.set_feedback_enable(band, 1)
        S.tracking_setup(band, reset_rate_khz=cfg.dev.bands[band]['flux_ramp_rate_khz'],
                         fraction_full_scale=cfg.dev.bands[band]['frac_pp'], make_plot=False,
                         save_plot=False, show_plot=False, channel=S.which_on(band),
                         nsamp=2 ** 18, lms_freq_hz=None, meas_lms_freq=True,
                         feedback_start_frac=cfg.dev.bands[band]['feedback_start_frac'],
                         feedback_end_frac=cfg.dev.bands[band]['feedback_end_frac'],
                         lms_gain=cfg.dev.bands[band]['lms_gain'])

    bias_groups = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    S.set_filter_disable(0)
    S.set_rtm_arb_waveform_enable(0)
    S.set_downsample_factor(20)
    for bias_index, bias_g in enumerate(bias_groups):
        S.set_tes_bias_low_current(bias_g)

    bias_v = 0
    bias_array = np.zeros(S._n_bias_groups)
    for bg in bias_groups:
        bias_array[bg] = bias_v
    S.set_tes_bias_bipolar_array(bias_array)
    time.sleep(stream_time)
    datafile_self = S.stream_data_on()
    time.sleep(stream_time)
    S.stream_data_off()

    fieldnames = ['bath_temp', 'bias_voltage', 'bias_line', 'band', 'data_path', 'type']
    row = {}
    row['data_path'] = datafile_self
    row['bias_voltage'] = bias_v
    row['type'] = 'noise'
    row['bias_line'] = 'all'
    row['band'] = 'all'
    row['bath_temp'] = bath_temp
    with open(out_fn, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow(row)

    for bias_gp in bias_groups:
        row = {}
        row['bath_temp'] = bath_temp
        row['bias_line'] = bias_gp
        row['band'] = 'all'
        row['bias_voltage'] = 'IV 20 to 0'
        row['type'] = 'IV'
        verbose_print(f'Taking IV on bias line {bias_gp}, all band', verbose)

        row['data_path'] = S.run_iv(
            bias_groups=[bias_gp],
            wait_time=0.001,
            bias_high=20,
            bias_low=0,
            bias_step=0.025,
            overbias_voltage=18,
            cool_wait=30,
            high_current_mode=False,
            make_plot=False,
            save_plot=True,
            #        cool_voltage=18.
        )

        with open(out_fn, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow(row)

        time.sleep(6)


prefix_str = f'\n From {uxm_bath_iv_noise.__name__} '

if __name__ == '__main__':
    import argparse
    from sodetlib.det_config import DetConfig

    """
    The code below will only run if the file is run directly, but not if elements from this file are imported.
    For example:
        python3 uxm_bath_iv_noise.py -args_for_argparse
    will have __name__ == '__main__' as True, and the code below will run locally.
    """
    # Seed a new parse for this file with the parser from the SMuRF controller class
    cfg = DetConfig()

    # set up the parser for this Script
    parser = argparse.ArgumentParser(description='Parser for noise_stack_by_band_new.py script.')
    parser.add_argument('bands', type=int, metavar='bands', nargs='+', action='append',
                        help='The SMuRF bands to target.')

    # optional arguments
    """
    Cannot find confirmation about how these variables are going being set. 
    """
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
    uxm_bath_iv_noise(S=S, cfg=cfg, bands=args.bands[0], bath_temp=args.bath_temp, out_fn=args.output_file,
                      stream_time=args.steam_time, verbose=args.verbose)
