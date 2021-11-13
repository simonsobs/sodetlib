"""
The core code is from Princeton, created by Yuhan Wang and Daniel Dutcher Oct/Nov 2021.

The code was refactored by Caleb Wheeler Nov 2021 to support argparse.
The argparse implementation (found at the bottom of the file) and provides a minimal
documentation framework.

Use:

python3 ufm_optimize_quick.py -h

to see the available options and required formatting.
"""
import time

import numpy as np
import scipy.signal as signal
import matplotlib

from uc_tuner import uc_rough_tune, uc_fine_tune


matplotlib.use('Agg')


def verbose_print(msg, verbose=True):
    if verbose:
        print(f"{prefix_str} {msg}")
    return


def ufm_optimize(S, cfg, opt_band=0, stream_time=20.0, fmin=5, fmax=50, fs=200, nperseg=2 ** 16,
                 detrend='constant', verbose=False):
    if verbose:
        print(f"{prefix_str} plotting directory is:\n{S.plot_dir}")

    S.all_off()
    S.set_rtm_arb_waveform_enable(0)
    S.set_filter_disable(0)
    S.set_downsample_factor(20)
    S.set_mode_dc()

    if verbose:
        print(f"{prefix_str} Setting up band {opt_band}, the initialization band")

    S.set_att_dc(opt_band, cfg.dev.bands[opt_band]["dc_att"])
    if verbose:
        print(f"{prefix_str} band {opt_band} dc_att {S.get_att_dc(opt_band)}")

    S.set_att_uc(opt_band, cfg.dev.bands[opt_band]["uc_att"])
    if verbose:
        print(f"{prefix_str} band {opt_band} uc_att {S.get_att_uc(opt_band)}")

    S.amplitude_scale[opt_band] = cfg.dev.bands[opt_band]["drive"]
    if verbose:
        print(f"{prefix_str} band {opt_band} tone power {S.amplitude_scale[opt_band]}")

        print(f"{prefix_str} estimating phase delay")

    S.estimate_phase_delay(opt_band)
    if verbose:
        print(f"{prefix_str} setting synthesis scale")
    # hard coding it for the current fw
    S.set_synthesis_scale(opt_band, 1)
    if verbose:
        print(f"{prefix_str} running find freq")
    S.find_freq(opt_band, tone_power=cfg.dev.bands[opt_band]["drive"], make_plot=True)
    if verbose:
        print(f"{prefix_str} running setup notches")
    S.setup_notches(opt_band, tone_power=cfg.dev.bands[opt_band]["drive"], new_master_assignment=True)
    if verbose:
        print(f"{prefix_str} running serial gradient descent and eta scan")
    S.run_serial_gradient_descent(opt_band)
    S.run_serial_eta_scan(opt_band)
    if verbose:
        print(f"{prefix_str} running tracking setup")
    S.set_feedback_enable(opt_band, 1)
    S.tracking_setup(
        opt_band,
        reset_rate_khz=cfg.dev.bands[opt_band]["flux_ramp_rate_khz"],
        fraction_full_scale=cfg.dev.bands[opt_band]["frac_pp"],
        make_plot=False,
        save_plot=False,
        show_plot=False,
        channel=S.which_on(opt_band),
        nsamp=2 ** 18,
        lms_freq_hz=cfg.dev.bands[opt_band]["lms_freq_hz"],
        meas_lms_freq=cfg.dev.bands[opt_band]["meas_lms_freq"],
        feedback_start_frac=cfg.dev.bands[opt_band]["feedback_start_frac"],
        feedback_end_frac=cfg.dev.bands[opt_band]["feedback_end_frac"],
        lms_gain=cfg.dev.bands[opt_band]["lms_gain"],
    )
    if verbose:
        print(f"{prefix_str} checking tracking")
    S.check_lock(
        opt_band,
        reset_rate_khz=cfg.dev.bands[opt_band]["flux_ramp_rate_khz"],
        fraction_full_scale=cfg.dev.bands[opt_band]["frac_pp"],
        lms_freq_hz=cfg.dev.bands[opt_band]["lms_freq_hz"],
        feedback_start_frac=cfg.dev.bands[opt_band]["feedback_start_frac"],
        feedback_end_frac=cfg.dev.bands[opt_band]["feedback_end_frac"],
        lms_gain=cfg.dev.bands[opt_band]["lms_gain"],
    )

    if verbose:
        print(f"{prefix_str} taking {stream_time}s timestream")

    # non blocking statement to start time stream and return the dat filename
    dat_path = S.stream_data_on()
    # collect stream data
    time.sleep(stream_time)
    # end the time stream
    S.stream_data_off()

    wl_list_temp = []
    timestamp, phase, mask, tes_bias = S.read_stream_data(dat_path, return_tes_bias=True)

    bands, channels = np.where(mask != -1)
    phase *= S.pA_per_phi0 / (2.0 * np.pi)  # uA

    for c, (b, ch) in enumerate(zip(bands, channels)):
        if ch < 0:
            continue
        ch_idx = mask[b, ch]
        f, Pxx = signal.welch(phase[ch_idx], nperseg=nperseg,
            fs=fs, detrend=detrend)
        Pxx = np.sqrt(Pxx)
        fmask = (fmin < f) & (f < fmax)

        wl = np.median(Pxx[fmask])
        wl_list_temp.append(wl)

    noise_param = wl_list_temp

    wl_median = np.median(noise_param)
    wl_length = len(noise_param)
    channel_length = len(noise_param)

    if wl_median > 250:
        print(f"{prefix_str} WL: {wl_median} with {wl_length} channels out of {channel_length}")
        raise IndexError(f"wl_median={wl_median} is to high. " +
                         "Something might be wrong, power level might be really off, please investigate")

    if wl_median < 120:
        if verbose:
            print(f"{prefix_str} WL: {wl_median} with {wl_length} channels out of {channel_length}\ncan be fine tuned")

        current_uc_att = S.get_att_uc(opt_band)
        current_tune_power = S.amplitude_scale[opt_band]

        estimate_att, current_tune_power, lowest_wl_index, wl_median = \
            uc_fine_tune(S=S, cfg=cfg, band=opt_band, current_uc_att=current_uc_att,
                         current_tune_power=current_tune_power,
                         stream_time=stream_time, fmin=fmin, fmax=fmax, fs=fs, nperseg=nperseg,
                         detrend=detrend, prefix_str=prefix_str, verbose=verbose)
        if verbose:
            print(f"{prefix_str} achieved at uc att {estimate_att} drive {current_tune_power}")

    if 120 < wl_median < 150:
        if verbose:
            print(f"{prefix_str} WL: {wl_median} with {wl_length} channels out of {channel_length}\ncan be fine tuned")

        current_uc_att = S.get_att_uc(opt_band)
        current_tune_power = S.amplitude_scale[opt_band]

        estimate_att, current_tune_power, lowest_wl_index, wl_median = \
            uc_rough_tune(S=S, cfg=cfg, band=opt_band, current_uc_att=current_uc_att,
                          current_tune_power=current_tune_power,
                          stream_time=stream_time, fmin=fmin, fmax=fmax, fs=fs, nperseg=nperseg,
                          detrend=detrend, prefix_str=prefix_str, verbose=verbose)

        if estimate_att < 16:
            if verbose:
                print(f"{prefix_str} adjusting tune power and uc att")
            new_tune_power = current_tune_power + 2
            adjusted_uc_att = current_uc_att + 11
            S.set_att_uc(opt_band, adjusted_uc_att)
            S.find_freq(opt_band, tone_power=new_tune_power, make_plot=True)
            S.setup_notches(opt_band, tone_power=new_tune_power, new_master_assignment=True)
            S.run_serial_gradient_descent(opt_band)
            S.run_serial_eta_scan(opt_band)
            current_uc_att = adjusted_uc_att
            current_tune_power = new_tune_power

        if estimate_att > 26:
            if verbose:
                print(f"{prefix_str} adjusting tune power and uc att")
            new_tune_power = current_tune_power - 2
            adjusted_uc_att = current_uc_att - 11
            S.set_att_uc(opt_band, adjusted_uc_att)
            S.find_freq(opt_band, tone_power=new_tune_power, make_plot=True)
            S.setup_notches(opt_band, tone_power=new_tune_power, new_master_assignment=True)
            S.run_serial_gradient_descent(opt_band)
            S.run_serial_eta_scan(opt_band)
            current_uc_att = adjusted_uc_att
            current_tune_power = new_tune_power

        estimate_att, current_tune_power, lowest_wl_index, wl_median = \
            uc_fine_tune(S=S, cfg=cfg, band=opt_band, current_uc_att=current_uc_att,
                         current_tune_power=current_tune_power,
                         stream_time=stream_time, fmin=fmin, fmax=fmax, fs=fs, nperseg=nperseg,
                         detrend=detrend, prefix_str=prefix_str, verbose=verbose)
        if verbose:
            print(f"{prefix_str} achieved at uc att {estimate_att} drive {current_tune_power}")
        step2_index = lowest_wl_index

        if step2_index == 0:
            if verbose:
                print(f"{prefix_str} can be fine tuned")
            estimate_att, current_tune_power, lowest_wl_index, wl_median = \
                uc_fine_tune(S=S, cfg=cfg, band=opt_band, current_uc_att=current_uc_att,
                             current_tune_power=current_tune_power,
                             stream_time=stream_time, fmin=fmin, fmax=fmax, fs=fs, nperseg=nperseg,
                             detrend=detrend, prefix_str=prefix_str, verbose=verbose)
        if verbose:
            print(f"{prefix_str} achieved at uc att {estimate_att} drive {current_tune_power}")

    if wl_median > 150 and wl_median < 250:
        if verbose:
            print(f"{prefix_str} WL: {wl_median} with {wl_length} channels out of {channel_length}\ncan be fine tuned")

        current_uc_att = S.get_att_uc(opt_band)
        current_tune_power = S.amplitude_scale[opt_band]

        estimate_att, current_tune_power, lowest_wl_index, wl_median = \
            uc_rough_tune(S=S, cfg=cfg, band=opt_band, current_uc_att=current_uc_att,
                          current_tune_power=current_tune_power,
                          stream_time=stream_time, fmin=fmin, fmax=fmax, fs=fs, nperseg=nperseg,
                          detrend=detrend, prefix_str=prefix_str, verbose=verbose)

        if wl_median < 120:
            print(f"{prefix_str} WL: {wl_median} with {wl_length} channels out of {channel_length}\ncan be fine tuned")

            current_uc_att = S.get_att_uc(opt_band)
            current_tune_power = S.amplitude_scale[opt_band]

            estimate_att, current_tune_power, lowest_wl_index, wl_median = \
                uc_fine_tune(S=S, cfg=cfg, band=opt_band, current_uc_att=current_uc_att,
                             current_tune_power=current_tune_power,
                             stream_time=stream_time, fmin=fmin, fmax=fmax, fs=fs, nperseg=nperseg,
                             detrend=detrend, prefix_str=prefix_str, verbose=verbose)
            if verbose:
                print(f"{prefix_str} achieved at uc att {estimate_att} drive {current_tune_power}")

        if wl_median > 120:
            if verbose:
                print(f"{prefix_str} WL: {wl_median} with {wl_length} channels out of {channel_length}\ncan be fine tuned")

            current_uc_att = S.get_att_uc(opt_band)
            current_tune_power = S.amplitude_scale[opt_band]

            estimate_att, current_tune_power, lowest_wl_index, wl_median = \
                uc_rough_tune(S=S, cfg=cfg, band=opt_band, current_uc_att=current_uc_att,
                              current_tune_power=current_tune_power,
                              stream_time=stream_time, fmin=fmin, fmax=fmax, fs=fs, nperseg=nperseg,
                              detrend=detrend, prefix_str=prefix_str, verbose=verbose)
            step1_index = lowest_wl_index

            if estimate_att < 16:
                if verbose:
                    print("adjusting tune power and uc att")
                new_tune_power = current_tune_power + 2
                adjusted_uc_att = current_uc_att + 12
                S.set_att_uc(opt_band, adjusted_uc_att)
                S.find_freq(opt_band, tone_power=new_tune_power, make_plot=True)
                S.setup_notches(opt_band, tone_power=new_tune_power, new_master_assignment=True)
                S.run_serial_gradient_descent(opt_band)
                S.run_serial_eta_scan(opt_band)
                current_uc_att = adjusted_uc_att
                current_tune_power = new_tune_power

            if estimate_att > 26:
                if verbose:
                    print("adjusting tune power and uc att")
                new_tune_power = current_tune_power - 2
                adjusted_uc_att = current_uc_att - 11
                S.set_att_uc(opt_band, adjusted_uc_att)
                S.find_freq(opt_band, tone_power=new_tune_power, make_plot=True)
                S.setup_notches(opt_band, tone_power=new_tune_power, new_master_assignment=True)
                S.run_serial_gradient_descent(opt_band)
                S.run_serial_eta_scan(opt_band)
                current_uc_att = adjusted_uc_att
                current_tune_power = new_tune_power

            estimate_att, current_tune_power, lowest_wl_index, wl_median = \
                uc_fine_tune(S=S, cfg=cfg, band=opt_band, current_uc_att=current_uc_att,
                             current_tune_power=current_tune_power,
                             stream_time=stream_time, fmin=fmin, fmax=fmax, fs=fs, nperseg=nperseg,
                             detrend=detrend, prefix_str=prefix_str, verbose=verbose)
            if verbose:
                print(f"{prefix_str} achieved at uc att {estimate_att} drive {current_tune_power}")
            step2_index = lowest_wl_index

            if step2_index == 0 and step1_index == 0:
                if verbose:
                    print(f"{prefix_str} can be fine tuned")
                estimate_att, current_tune_power, lowest_wl_index, wl_median = \
                    uc_fine_tune(S=S, cfg=cfg, band=opt_band, current_uc_att=current_uc_att,
                                 current_tune_power=current_tune_power,
                                 stream_time=stream_time, fmin=fmin, fmax=fmax, fs=fs, nperseg=nperseg,
                                 detrend=detrend, prefix_str=prefix_str, verbose=verbose)

            if step2_index == 4 and step1_index == 4:
                if verbose:
                    print(f"{prefix_str} can be fine tuned")
                estimate_att, current_tune_power, lowest_wl_index, wl_median = \
                    uc_fine_tune(S=S, cfg=cfg, band=opt_band, current_uc_att=current_uc_att,
                                 current_tune_power=current_tune_power,
                                 stream_time=stream_time, fmin=fmin, fmax=fmax, fs=fs, nperseg=nperseg,
                                 detrend=detrend, prefix_str=prefix_str, verbose=verbose)
            if verbose:
                print(f"{prefix_str} achieved at uc att {estimate_att} drive {current_tune_power}")

    try:
        if verbose:
            print(f"{prefix_str} WL: {wl_median} with {wl_length} channels out of {channel_length}")
            print(f"{prefix_str} achieved at uc att {estimate_att} drive {current_tune_power}")
    except:
        print(f"{prefix_str} WL={wl_median} is off, please investigate")
    if verbose:
        print(f"{prefix_str} plotting directory is:\n{S.plot_dir}")
    return


prefix_str = f'\n From {ufm_optimize.__name__} '

if __name__ == '__main__':
    import argparse
    from sodetlib.det_config import DetConfig

    """
    The code below will only run if the file is run directly, but not if elements from this file are imported.
    For example:
        python3 ufm_optimize_quick.py -args_for_argparse
    will have __name__ == '__main__' as True, and the code below will run locally.
    """
    # Seed a new parse for this file with the parser from the SMuRF controller class
    cfg = DetConfig()

    # set up the parser for this Script
    parser = argparse.ArgumentParser(description='Parser for ufm_optimize_quick.py script.')
    parser.add_argument('band', type=int, metavar='band',
                        help='The SMuRF band number to optimize on.')

    # optional arguments
    parser.add_argument('--stream-time', dest='stream_time', type=float, default=20.0,
                        help="float, optional, default is 20.0. The amount of time to sleep in seconds while " +
                             "streaming SMuRF data for analysis.")
    parser.add_argument('--fmin', dest='fmin', type=float, default=float('-inf'),
                        help="float, optional, default is float('-inf'). The lower frequency (Hz) bound used " +
                             "when creating a mask of white noise levels Suggested value of 5.0")
    parser.add_argument('--fmax', dest='fmax', type=float, default=float('inf'),
                        help="float, optional, default is float('inf'). The upper frequency (Hz) bound used " +
                             "when creating a mask of white noise levels Suggested value of 50.0")
    parser.add_argument('--fs', dest='fs', type=float, default=None,
                        help="float, optional, default is None. Passed to scipy.signal.welch. The sample rate.")
    parser.add_argument('--nperseg', dest='nperseg', type=int, default=2 ** 18,
                        help="int, optional, default is 2**18. The number of samples used in the PSD estimator. " +
                             "See scipy.signal.welch.")
    parser.add_argument('--detrend', dest='detrend', default='constant',
                        help="str, optional, default is 'constant'. Passed to scipy.signal.welch.")
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
    ufm_optimize(S=S, cfg=cfg, opt_band=args.band, stream_time=args.stream_time,
                 fmin=args.fmin, fmax=args.fmax, fs=args.fs, nperseg=args.nperseg,
                 detrend=args.detrend, verbose=args.verbose)


