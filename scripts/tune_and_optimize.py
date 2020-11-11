from sodetlib.analysis.tickle import analyze_tickle
from sodetlib.smurf_funcs import tickle
import sodetlib.smurf_funcs.optimize_params as op
import sodetlib.smurf_funcs.smurf_ops as so
import sodetlib.smurf_funcs.health_check as hc
from sodetlib.det_config import DetConfig
from sodetlib.util import cprint, TermColors

import time
import os
import numpy as np
import argparse
import pysmurf.client
import matplotlib.pyplot as plt
import matplotlib
import pickle as pkl
matplotlib.use('Agg')


def clear_cfg(cfg, dump=True):
    cfg.dev.exp['tunefile'] = None
    for band in range(8):
        cfg.dev.update_band(band, {
            'optimized_tracking': False,
            'active_subbands': []
        })
    if dump:
        cfg_path = os.path.abspath(os.path.expandvars(cfg.dev_file))
        cfg.dev.dump(cfg_path, clobber=True)
    return


if __name__ == '__main__':
    cfg = DetConfig()
    parser = argparse.ArgumentParser()

    # Required argparse arguments
    parser.add_argument('--biasgroup', type=int, nargs='+', required=True,
                        help='bias group that you want to run tickles on')

    # Typically just use default values for argparse arguments
    parser.add_argument('--BW-target', '-m', type=float, default=500,
                        help='Target readout bandwidth to optimize lms_gain')

    parser.add_argument('--wait-time', type=float, default=0.1,
                        help='Time to wait between flux steps in seconds.')

    parser.add_argument('--Npts', type=int, default=3,
                        help='Number of points to average')

    parser.add_argument('--NPhi0s', type=int, default=4,
                        help='Number of periods in your squid curve.')

    parser.add_argument('--Nsteps', type=int, default=500,
                        help='Number of points in your squid curve.')

    parser.add_argument('--relock', action='store_true',
                        help='If specified will run relock.')

    parser.add_argument('--tickle-voltage', type=float, default=0.1,
                        help='Amplitude (not peak-peak) of your tickle in '
                             'volts')

    parser.add_argument('--high-current', action='store_true')

    parser.add_argument('--over-bias', action='store_true')

    parser.add_argument('--channels', type=int, nargs='+', default=None,
                        help='Channels that you want to calculate the tickle response of')
    parser.add_argument('--make-channel-plots', action='store_true')

    parser.add_argument('--R-threshold', default=100,
                        help='Resistance threshold for determining detector channel')
    parser.add_argument('--all', '-a', action='store_true',
                        help="If set will run all possible optimizations.")

    args = cfg.parse_args(parser)
    S = cfg.get_smurf_control(dump_configs=True)

    channels = args.channels

    cfg_path = os.path.abspath(os.path.expandvars(cfg.dev_file))

    if args.all:
        clear_cfg(cfg, dump=True)

    # Turns on amps and adjust/returns "optimal bias" and then does a few
    # system health checks.
    cprint('Running system health check.', style=TermColors.HEADER)
    ctime_prev = time.time()
    health_check_success = hc(S, cfg)
    ctime_now = time.time()
    health_check_time = ctime_now - ctime_prev
    cprint(f'Health check took {health_check_time} sec')
    cfg.dev.dump(cfg_path, clobber=True)

    if not health_check_success:
        raise Exception("Health Check Failed!")

    # Next find which bands and subbands have resonators attached
    cprint('Identifying active bands and subbands', style=TermColors.HEADER)
    ctime_prev = ctime_now
    bands, subband_dict = so.find_subbands(S, cfg)
    ctime_now = time.time()
    find_subband_time = ctime_now - ctime_prev
    cprint(f'Find subbands took {find_subband_time} sec')
    cfg.dev.dump(cfg_path, clobber=True)

    # Now tune on those bands/find_subbands
    tunefile = cfg.dev.exp['tunefile']
    if tunefile is None:
        relock_flag = False
        cprint('Tuning', style=TermColors.HEADER)
        start = time.time()
        num_chans_tune, tunefile = so.find_and_tune_freq(S, cfg, bands)
        stop = time.time()
        tune_time = stop - start
        cprint(f'Find and tune freq took {tune_time} sec')
        cfg.dev.dump(cfg_path, clobber=True)
    else:
        relock_flag = True
        cprint(f"Loading Tunefile {tunefile} from device config",
               style=TermColors.HEADER)
        start = time.time()
        S.load_tune(tunefile)
        stop = time.time()
        tune_time = stop - start

    # Now setup tracking
    ctime_prev = ctime_now
    optimize_dict = {}

    active_bands = []
    for band in bands:
        if 'resonances' in S.freq_resp[band]:
            active_bands.append(band)
        else:
            cprint(f"No resonators in band {band}", False)

    for band in active_bands:
        cprint("Relocking")
        if relock_flag:
            S.relock(band)
        for _ in range(2):
            S.run_serial_gradient_descent(band)
            S.run_serial_eta_scan(band)

        optimize_dict[band] = {}

        band_cfg = cfg.dev.bands[band]
        if band_cfg.get('optimized_tracking', False):
            cprint(f"Tracking for band {band} has already been optimized",
                   True)
        else:
            cprint(f'Optimizing tracking for band {band}',
                   style=TermColors.HEADER)
            start = time.time()
            lms_freq, frac_pp, tracking_dict = op.optimize_tracking(
                S, cfg, band, relock=False)
            stop = time.time()
            cprint(f"Optimize tracking band {band} took {stop - start} sec")
            cfg.dev.dump(cfg_path, clobber=True)
            optimize_dict.update({
                'lms_freq': lms_freq,
                'frac_pp': frac_pp,
                'tracking_dict': tracking_dict,
                'tracking_time': stop - start
            })

        if not band_cfg.get('optimized_drive', False):
            cprint(f'UC attenuator for band {band}', style=TermColors.HEADER)
            start = time.time()
            noise, atten, drive = op.optimize_power_per_band(S, cfg, band, meas_time=1)
            stop = time.time()
            cprint(f"Optimize input power for band {band} took {stop - start} sec")
            cfg.dev.dump(cfg_path, clobber=True)
            optimize_dict.update({
                'atten': atten, 'drive': drive,
                'input_power_time': stop - start
            })
        else:
            cprint(f"Drive and atten for band {band} has already been "
                    "optimized", True)

        if not band_cfg.get('optimized_lms_gain', False):
            cprint(f'Optimizing lms_gain for band {band}',
                   style=TermColors.HEADER)
            start = time.time()
            lms_gain, lms_gain_dict = op.optimize_lms_gain(S, cfg, band,
                                                           BW_target=args.BW_target)
            stop = time.time()
            cprint(f"Optimize lms_gain for band {band} took {stop - start} sec")
            cfg.dev.dump(cfg_path, clobber=True)
            optimize_dict.update({
                'lms_gain': lms_gain,
                'lms_gain_dict': lms_gain_dict,
                'lms_gain_time': stop - start
            })
        else:

            cprint(f"LMS Gain for band {band} has already been optimized",
                   True)
            print("Running tracking kwargs with optimal args.")
            tracking_kwargs = {
                'reset_rate_khz': 4, 'lms_freq_hz': band_cfg['lms_freq_hz'],
                'fraction_full_scale': band_cfg['frac_pp'], 'make_plot': True, 'save_plot': True, 'show_plot': False,
                'channel': [], 'nsamp': 2**18,
                'feedback_start_frac': 0.02, 'feedback_end_frac': 0.94,
                'lms_gain': band_cfg['lms_gain'], 'return_data': True
            }
            S.tracking_setup(band, **tracking_kwargs)

    # Right now lms_gain doesn't set you to that after completion...we need
    # to add this
    cfg.dev.dump(cfg_path, clobber=True)

    cprint('Taking and analyzing optimized noise', style=TermColors.HEADER)
    datafile = S.take_stream_data(20)
    fig, ax = plt.subplots()
    for band in active_bands:
        start = time.time()
        median_noise, noise_dict = op.analyze_noise_psd(S, band, datafile,
                                                        fit_curve=False)
        stop = time.time()
        cprint(f"Averaging noise for band {band} took {stop - start} sec")
        wls = np.array([c['white noise'] for c in noise_dict.values()])
        print(wls)
        optimize_dict[band].update({
            'wls': wls,
            'median_noise': median_noise,
            'noise_dict': noise_dict,
            'noise_time': stop - start
        })
        ax.hist(wls, bins=40, range=(0, 200), alpha=0.5, label=f'Band {band}')
    ax.set(xlabel="White Noise (pA/rt(Hz))", ylabel="Number of channels")
    ax.legend()
    path = os.path.join(S.plot_dir, f'{S.get_timestamp()}_wl_histogram.png')
    print(f"Saving figure to {path}")
    fig.savefig(path)
    S.pub.register_file(path, 'wl_hist', plot=True)

    cfg.dev.dump(cfg_path, clobber=True)

    # Need to add a function that identifies which biasgroups are connected so
    # that we don't need to pass a biasgroup argument.
    cprint('Identifying channels w/ detectors and calculating resistance.',
           style=TermColors.HEADER)
    ctime_prev = ctime_now
    tickle_files = {}
    for band in active_bands:
        tickle_files[band], cur_dc = tickle.take_tickle(
            S, band=band, bias_group=args.biasgroup,
            tickle_voltage=args.tickle_voltage, high_current=args.high_current,
            over_bias=args.over_bias)
        optimize_dict[band]['tickle_dict'] = analyze_tickle(
            S, band=band, data_file=tickle_files[band], dc_level=cur_dc,
            tickle_voltage=args.tickle_voltage, high_current=args.high_current,
            channels=channels, make_channel_plots=True,
            R_threshold=args.R_threshold)

    ctime_now = time.time()
    cprint(f'Tickle took {ctime_now - ctime_prev} sec')
    cfg.dev.dump(cfg_path, clobber=True)
    pkl.dump(optimize_dict, open('/sodetlib/tests/demo_script.pkl', 'wb'))

    summary = '-'*40 + '\n'
    summary += "Optimization summary\n"
    summary += '-'*40 + '\n'
    summary += f"Health Check: {health_check_success} ({health_check_time} sec)\n"
    summary += f"Find Subbands: ({find_subband_time} sec)"

    cprint(summary, True)




