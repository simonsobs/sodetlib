from sodetlib.analysis.tickle import analyze_tickle
from sodetlib.smurf_funcs import tickle
import sodetlib.smurf_funcs.optimize_params as op
import sodetlib.smurf_funcs.smurf_ops as so
from sodetlib.smurf_funcs.health_check import health_check
from sodetlib.det_config import DetConfig
from sodetlib.util import cprint, TermColors, SectionTimer

import time
import os
import numpy as np
import argparse
import pysmurf.client
from pysmurf.client.util.pub import set_action
import matplotlib.pyplot as plt
import matplotlib
import pickle as pkl
matplotlib.use('Agg')


@set_action()
def full_optimize(S, cfg, args):
    timer = SectionTimer()
    cfg_path = os.path.abspath(os.path.expandvars(cfg.dev_file))

    ##############################################################
    # Health Check
    ##############################################################
    timer.start_section('Health Check')
    cprint("HealthCheck", style=TermColors.HEADER)

    if not health_check(S, cfg):
        raise Exception("Health Check Failed!")

    cfg.dev.dump(cfg_path, clobber=True)

    ##############################################################
    # Find bands and subbands
    ##############################################################
    timer.start_section("Find subbands")
    cprint("Find Subbands", style=TermColors.HEADER)
    bands, subband_dict = so.find_subbands(S, cfg)
    cfg.dev.dump(cfg_path, clobber=True)

    ##############################################################
    # Tuning
    ##############################################################
    timer.start_section("Tuning")
    cprint("Tuning", style=TermColors.HEADER)

    tunefile = cfg.dev.exp['tunefile']
    if tunefile is None:
        relock_flag = False
        num_chans_tune, tunefile = so.find_and_tune_freq(S, cfg, bands)
        cfg.dev.dump(cfg_path, clobber=True)
    else:
        relock_flag = True
        S.load_tune(tunefile)

    # Find which bands have resonators
    active_bands = []
    if args.bands is not None:
        active_bands = args.bands
    else:
        active_bands = []
        for k, resp in S.freq_resp.items():
            if 'resonances' in resp:
                active_bands.append(int(k))
            else:
                cprint(f"No resonators in band {k}", False)

    optimize_dict = {}
    # Start band-specific operations
    for band in active_bands:
        timer.start_section(f"Relock (b{band})")
        cprint(f"Relock (b{band})", style=TermColors.HEADER)

        out = {}
        band_cfg = cfg.dev.bands[band]
        if relock_flag:
            S.relock(band)
        for _ in range(2):
            S.run_serial_gradient_descent(band)
            S.run_serial_eta_scan(band)
        # Running tracking setup with existing dev_config args
        S.tracking_setup(
            band, reset_rate_khz=4, make_plot=True, save_plot=True,
            show_plot=False, channel=[], nsamp=2**18, feedback_start_frac=0.02,
            feedback_end_frac=0.94, return_data=False,
            lms_freq_hz=band_cfg['lms_freq_hz'],
            fraction_full_scale=band_cfg['frac_pp'],
            lms_gain=band_cfg['lms_gain']
        )

        ##############################################################
        # Tracking optimization
        ##############################################################
        timer.start_section(f"Optimize Tracking (b{band})")
        cprint(f"Optimize Tracking (b{band})", style=TermColors.HEADER)

        if band_cfg.get('optimized_tracking', False):
            cprint(f"Tracking for band {band} has already been optimized",
                   True)
            out.update({
                'lms_freq_hz': band_cfg['lms_freq_hz'],
                'frac_pp': band_cfg['frac_pp'],
                'optimize_tracking_out': None
            })
        else:
            lms_freq, frac_pp, tracking_out = op.optimize_tracking(
                S, cfg, band, relock=False
            )
            out.update({
                'lms_freq_hz': lms_freq, 'frac_pp': frac_pp,
                'optimize_tracking_out': tracking_out
            })
            cfg.dev.dump(cfg_path, clobber=True)

        ##############################################################
        # Input power optimization
        ##############################################################
        timer.start_section(f"Optimize Input Power (b{band})")
        cprint(f"Optimize Input Power (b{band})", style=TermColors.HEADER)

        if band_cfg.get('optimized_drive', False):
            cprint(f"Drive and atten for band {band} has already been "
                   "optimized", True)
            out.update({
                'uc_att': band_cfg['uc_att'], 'drive': band_cfg['drive'],
            })
        else:
            noise, uc_atten, drive = op.optimize_power_per_band(
                S, cfg, band, meas_time=1
            )
            cfg.dev.dump(cfg_path, clobber=True)
            out.update({
                'uc_att': uc_atten, 'drive': drive
            })

        ##############################################################
        # LMS Gain optimization
        ##############################################################
        timer.start_section(f"Optimize LMS Gain (b{band})")
        cprint(f"Optimize LMS Gain (b{band})", style=TermColors.HEADER)
        if band_cfg.get('optimized_lms_gain', False):
            cprint(f"LMS Gain for band {band} has already been optimized",
                   True)
            out.update({'lms_gain': band_cfg['lms_gain'],
                        'optimize_lms_gain_out': None})
        else:
            lms_gain, lms_gain_out = op.optimize_lms_gain(
                S, cfg, band, BW_target=args.BW_target
            )
            out.update({
                'lms_gain': lms_gain, 'optimize_lms_gain_out': lms_gain_out
            })
            cfg.dev.dump(cfg_path, clobber=True)

        optimize_dict[band] = out

        # Rerun tracking setup with optimal params
        timer.start_section(f"Tracking setup (b{band})")
        cprint(f"Rerunning tracking setup for band {band}",
               style=TermColors.HEADER)

        S.tracking_setup(
            band, reset_rate_khz=band_cfg['flux_ramp_rate_khz'],
            make_plot=True, save_plot=True, show_plot=False, channel=[],
            nsamp=2**18, feedback_start_frac=0.02, feedback_end_frac=0.94,
            return_data=False, lms_freq_hz=out['lms_freq_hz'],
            fraction_full_scale=out['frac_pp'], lms_gain=out['lms_gain']
        )

    ##############################################################
    # Optimized Noise Calculation
    ##############################################################
    timer.start_section("Optimized Noise Analysis")
    cprint("Optimized Noise Analysis", style=TermColors.HEADER)

    datafile = S.take_stream_data(20, register_file=False)

    cols, rows = len(active_bands) % 4, 1 + len(active_bands) // 4
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows), squeeze=False)
    for i, band in enumerate(active_bands):
        median_noise, noise_dict = op.analyze_noise_psd(S, band, datafile,
                                                        fit_curve=False)
        wls = np.array([chan['white noise'] for chan in noise_dict.values()])
        optimize_dict[band].update({
            'wls': wls,
            'median_noise': median_noise,
            'noise_dict': noise_dict,
        })

        cprint(f"Median noise for band {band}: {median_noise}", True)
        ax = axes[i // 4, i % 4]
        ax.hist(wls, bins=40, range=(0, median_noise*3), alpha=0.8,
                label=f'Band {band}', edgecolor='black')
        ax.axvline(median_noise, color='C1')
        ax.set(title=f'Band {band}', xlabel="White Noise (pA/rt(Hz))",
               ylabel="Number of channels")
    path = os.path.join(S.plot_dir, f'{S.get_timestamp()}_wl_histogram.png')
    fig.savefig(path)
    S.pub.register_file(path, 'wl_hist', plot=True)

    ##############################################################
    # Tickle Calculation
    ##############################################################
    timer.start_section("Tickle Calculation")
    cprint("Identifying detector channels and calculating resistance",
           style=TermColors.HEADER)
    tickle_files = {}
    for band in active_bands:
        tickle_files[band], cur_dc = tickle.take_tickle(
            S, band=band, bias_group=args.biasgroup,
            tickle_voltage=args.tickle_voltage, high_current=args.high_current,
            over_bias=args.over_bias)
        tick_dict = analyze_tickle(
            S, band=band, data_file=tickle_files[band], dc_level=cur_dc,
            tickle_voltage=args.tickle_voltage, high_current=args.high_current,
            channels=args.channels, make_channel_plots=True,
            R_threshold=args.R_threshold)

        det_chans = [
            int(c) for c, d in tick_dict.items() if d['detector_channel']
        ]

        cprint(f"Detectors found: {det_chans}", True)
        optimize_dict[band]['tickle_dict'] = tick_dict
        optimize_dict[band]['detector_chans'] = det_chans
        cfg.dev.update_band(band, {'detectors': det_chans})
        cfg.dev.dump(cfg_path, clobber=True)

    pkl.dump(optimize_dict, open('/sodetlib/tests/demo_script.pkl', 'wb'))

    ##############################################################
    # Summary
    ##############################################################
    summary = ''
    for b in active_bands:
        out = optimize_dict[b]
        summary += "-"*60 + "\n" + f"Band {b} Summary\n" + "-"*60 + "\n"
        for key in ['lms_freq_hz', 'frac_pp', 'drive', 'uc_att',
                    'lms_gain', 'median_noise', 'detector_chans']:
            summary += f'{key:20s}:\t{out[key]}\n'
    cprint(summary, True)

    timer.stop()
    time_file = os.path.join(
        S.output_dir, f"{S.get_timestamp()}_optimize_time_summary.txt"
    )
    cprint(timer.summary(), True)
    with open(time_file, 'w') as f:
        f.write(timer.summary())


def clear_cfg(cfg, dump=True):
    cfg.dev.exp['tunefile'] = None
    for band in range(8):
        cfg.dev.update_band(band, {
            'optimized_tracking': False,
            'optimized_drive': False,
            'optimized_lms_gain': False,
            'active_subbands': []
        })
    if dump:
        cfg_path = os.path.abspath(os.path.expandvars(cfg.dev_file))
        cfg.dev.dump(cfg_path, clobber=True)
    return


def make_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()

    parser.add_argument('--all', '-a', action='store_true',
                        help="If set will run all possible optimizations.")
    parser.add_argument('--bands', '-b', default=None, nargs='+', type=int,
                        help="Bands to optimize")
    parser.add_argument('--BW-target', '--bw', type=float, default=500,
                        help='Target readout bandwidth to optimize lms_gain')

    tickle_group = parser.add_argument_group('tickle', "Tickle Arguments")
    tickle_group.add_argument('--biasgroup', '--bg', type=int, nargs='+',
                              help='bias group that you want to run tickles '
                                   'on')
    tickle_group.add_argument('--tickle-voltage', type=float, default=0.1,
                              help='Amplitude (not peak-peak) of your tickle '
                                   'in volts')
    tickle_group.add_argument('--high-current', action='store_true',
                              help="Whether to run tickle in high current "
                                   "mode")
    tickle_group.add_argument('--over-bias', action='store_true',
                              help='Whether or not to bias in high current '
                                   'mode before taking tickle')
    tickle_group.add_argument('--channels', type=int, nargs='+', default=None,
                              help='Channels that you want to calculate the '
                                   'tickle response of')
    tickle_group.add_argument('--R-threshold', default=100,
                              help='Resistance threshold for determining '
                                   'detector channel')
    return parser


if __name__ == '__main__':
    cfg = DetConfig()
    parser = make_parser()
    args = cfg.parse_args(parser)
    S = cfg.get_smurf_control(dump_configs=True)

    cfg_path = os.path.abspath(os.path.expandvars(cfg.dev_file))
    if args.all:
        clear_cfg(cfg, dump=True)

    full_optimize(S, cfg, args)
