"""
The core code is from Princeton, created by Yuhan Wang and Daniel Dutcher Oct/Nov 2021.

The code was refactored by Caleb Wheeler Nov 2021 to support argparse.
The argparse implementation (found at the bottom of the file) and provides a minimal
documentation framework.

Use:

python3 ufm_optimize_quick_normal.py -h

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


def ufm_optimize_normal(S, cfg, opt_band=0, stream_time=20.0, fmin=5, fmax=50, fs=200, nperseg=2 ** 16,
                        detrend='constant', verbose=False):
    verbose_print("plotting directory is:", verbose)
    verbose_print(S.plot_dir)

    S.all_off()
    S.set_rtm_arb_waveform_enable(0)
    S.set_filter_disable(0)
    S.set_downsample_factor(20)
    S.set_mode_dc()

    verbose_print("setting up band {}".format(opt_band))

    S.set_att_dc(opt_band, cfg.dev.bands[opt_band]["dc_att"])
    verbose_print("band {} dc_att {}".format(opt_band, S.get_att_dc(opt_band)))

    S.set_att_uc(opt_band, cfg.dev.bands[opt_band]["uc_att"])
    verbose_print("band {} uc_att {}".format(opt_band, S.get_att_uc(opt_band)))

    S.amplitude_scale[opt_band] = cfg.dev.bands[opt_band]["drive"]
    verbose_print("band {} tone power {}".format(opt_band, S.amplitude_scale[opt_band]))

    verbose_print("estimating phase delay")
    S.estimate_phase_delay(opt_band)
    verbose_print("setting synthesis scale")
    # hard coding it for the current fw
    S.verbose_print(opt_band, 1)
    verbose_print("running find freq")
    S.find_freq(opt_band, tone_power=cfg.dev.bands[opt_band]["drive"], make_plot=True)
    verbose_print("running setup notches")
    S.setup_notches(
        opt_band, tone_power=cfg.dev.bands[opt_band]["drive"], new_master_assignment=True
    )
    verbose_print("running serial gradient descent and eta scan")
    S.run_serial_gradient_descent(opt_band)
    S.run_serial_eta_scan(opt_band)
    verbose_print("running tracking setup")
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
    verbose_print("checking tracking")
    S.check_lock(
        opt_band,
        reset_rate_khz=cfg.dev.bands[opt_band]["flux_ramp_rate_khz"],
        fraction_full_scale=cfg.dev.bands[opt_band]["frac_pp"],
        lms_freq_hz=cfg.dev.bands[opt_band]["lms_freq_hz"],
        feedback_start_frac=cfg.dev.bands[opt_band]["feedback_start_frac"],
        feedback_end_frac=cfg.dev.bands[opt_band]["feedback_end_frac"],
        lms_gain=cfg.dev.bands[opt_band]["lms_gain"],
    )
    S.overbias_tes_all(bias_groups=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], overbias_wait=1, tes_bias=12, cool_wait=3,
                       high_current_mode=False, overbias_voltage=12)
    verbose_print("waiting for thermal environment get stabilized")
    time.sleep(120)

    verbose_print(f"taking {stream_time}s timestream")

    # non blocking statement to start time stream and return the dat filename
    dat_path = S.stream_data_on()
    # collect stream data
    time.sleep(stream_time)
    # end the time stream
    S.stream_data_off()

    wl_list_temp = []
    timestamp, phase, mask, tes_bias = S.read_stream_data(dat_path,
                                                          return_tes_bias=True)

    bands, channels = np.where(mask != -1)
    phase *= S.pA_per_phi0 / (2.0 * np.pi)  # uA

    for c, (b, ch) in enumerate(zip(bands, channels)):
        if ch < 0:
            continue
        ch_idx = mask[b, ch]
        sampleNums = np.arange(len(phase[ch_idx]))
        t_array = sampleNums / fs
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
    verbose_print('wl_median')

    if wl_median > 150:
        verbose_print(f"WL: {wl_median} with {wl_length} channels out of {channel_length}", True)
        verbose_print("something might be wrong, power level might be really off, please investigate", True)

    if wl_median < 60:
        verbose_print(f"WL: {wl_median} with {wl_length} channels out of {channel_length}", verbose)
        verbose_print("can be fine tuned", verbose)

        current_uc_att = S.get_att_uc(opt_band)
        current_tune_power = S.amplitude_scale[opt_band]

        estimate_att, current_tune_power, lowest_wl_index, wl_median = \
            uc_fine_tune(S=S, cfg=cfg, band=opt_band, current_uc_att=current_uc_att,
                         current_tune_power=current_tune_power,
                         stream_time=stream_time, fmin=fmin, fmax=fmax, fs=fs, nperseg=nperseg,
                         detrend=detrend, prefix_str=prefix_str, verbose=verbose)

        verbose_print(f"achieved at uc att {estimate_att} drive {current_tune_power}", verbose)

    if wl_median > 60 and wl_median < 120:
        verbose_print(f"WL: {wl_median} with {wl_length} channels out of {channel_length}", verbose)
        verbose_print("can be fine tuned", verbose)

        current_uc_att = S.get_att_uc(opt_band)
        current_tune_power = S.amplitude_scale[opt_band]

        estimate_att, current_tune_power, lowest_wl_index, wl_median = \
            uc_rough_tune(S=S, cfg=cfg, band=opt_band, current_uc_att=current_uc_att,
                          current_tune_power=current_tune_power,
                          stream_time=stream_time, fmin=fmin, fmax=fmax, fs=fs, nperseg=nperseg,
                          detrend=detrend, prefix_str=prefix_str, verbose=verbose)

        if estimate_att < 16:
            verbose_print("adjusting tune power and uc att", verbose)
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
            verbose_print("adjusting tune power and uc att", verbose)
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
        verbose_print(f"WL: {wl_median} with {wl_length} channels out of {channel_length}", verbose)
        
        step2_index = lowest_wl_index

        if step2_index == 0:
            verbose_print("can be fine tuned", verbose)
            estimate_att, current_tune_power, lowest_wl_index, wl_median = \
                uc_fine_tune(S=S, cfg=cfg, band=opt_band, current_uc_att=current_uc_att,
                             current_tune_power=current_tune_power,
                             stream_time=stream_time, fmin=fmin, fmax=fmax, fs=fs, nperseg=nperseg,
                             detrend=detrend, prefix_str=prefix_str, verbose=verbose)

        verbose_print(f"achieved at uc att {estimate_att} drive {current_tune_power}", verbose)

    if wl_median > 120 and wl_median < 150:
        verbose_print(f"WL: {wl_median} with {wl_length} channels out of {channel_length}", verbose)
        verbose_print("can be fine tuned", verbose)

        current_uc_att = S.get_att_uc(opt_band)
        current_tune_power = S.amplitude_scale[opt_band]

        estimate_att, current_tune_power, lowest_wl_index, wl_median = \
            uc_rough_tune(S=S, cfg=cfg, band=opt_band, current_uc_att=current_uc_att,
                          current_tune_power=current_tune_power,
                          stream_time=stream_time, fmin=fmin, fmax=fmax, fs=fs, nperseg=nperseg,
                          detrend=detrend, prefix_str=prefix_str, verbose=verbose)

        if wl_median < 60:
            verbose_print(f"WL: {wl_median} with {wl_length} channels out of {channel_length}", verbose)
            verbose_print("can be fine tuned", verbose)

            current_uc_att = S.get_att_uc(opt_band)
            current_tune_power = S.amplitude_scale[opt_band]

            estimate_att, current_tune_power, lowest_wl_index, wl_median = \
                uc_fine_tune(S=S, cfg=cfg, band=opt_band, current_uc_att=current_uc_att,
                             current_tune_power=current_tune_power,
                             stream_time=stream_time, fmin=fmin, fmax=fmax, fs=fs, nperseg=nperseg,
                             detrend=detrend, prefix_str=prefix_str, verbose=verbose)

            verbose_print(f"achieved at uc att {estimate_att} drive {current_tune_power}", verbose)

        if wl_median > 60:
            verbose_print(f"WL: {wl_median} with {wl_length} channels out of {channel_length}", verbose)
            verbose_print("can be fine tuned", verbose)

            current_uc_att = S.get_att_uc(opt_band)
            current_tune_power = S.amplitude_scale[opt_band]

            if estimate_att < 16:
                verbose_print("adjusting tune power and uc att", verbose)
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
                verbose_print("adjusting tune power and uc att", verbose)
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
                uc_rough_tune(S=S, cfg=cfg, band=opt_band, current_uc_att=current_uc_att,
                              current_tune_power=current_tune_power,
                              stream_time=stream_time, fmin=fmin, fmax=fmax, fs=fs, nperseg=nperseg,
                              detrend=detrend, prefix_str=prefix_str, verbose=verbose)
            step1_index = lowest_wl_index

            # if estimate_att < 16:
            #     verbose_print("adjusting tune power and uc att")
            #     new_tune_power = current_tune_power + 2
            #     adjusted_uc_att = current_uc_att + 12
            #     S.set_att_uc(band, adjusted_uc_att)
            #     S.find_freq(band, tone_power=new_tune_power, make_plot=True)
            #     S.setup_notches(band, tone_power=new_tune_power, new_master_assignment=True)
            #     S.run_serial_gradient_descent(band)
            #     S.run_serial_eta_scan(band)
            #     current_uc_att = adjusted_uc_att
            #     current_tune_power = new_tune_power

            # if estimate_att > 26:
            #     verbose_print("adjusting tune power and uc att")
            #     new_tune_power = current_tune_power + 2
            #     adjusted_uc_att = current_uc_att - 11
            #     S.set_att_uc(band, adjusted_uc_att)
            #     S.find_freq(band, tone_power=new_tune_power, make_plot=True)
            #     S.setup_notches(band, tone_power=new_tune_power, new_master_assignment=True)
            #     S.run_serial_gradient_descent(band)
            #     S.run_serial_eta_scan(band)
            #     current_uc_att = adjusted_uc_att
            #     current_tune_power = new_tune_power

            estimate_att, current_tune_power, lowest_wl_index, wl_median = \
                uc_fine_tune(S=S, cfg=cfg, band=opt_band, current_uc_att=current_uc_att,
                             current_tune_power=current_tune_power,
                             stream_time=stream_time, fmin=fmin, fmax=fmax, fs=fs, nperseg=nperseg,
                             detrend=detrend, prefix_str=prefix_str, verbose=verbose)
            verbose_print(f"WL: {wl_median} with {wl_length} channels out of {channel_length}", verbose)
            verbose_print("can be fine tuned", verbose)
            step2_index = lowest_wl_index

            if step2_index == 0 and step1_index == 0:
                verbose_print("can be fine tuned", verbose)
                estimate_att, current_tune_power, lowest_wl_index, wl_median = \
                    uc_fine_tune(S=S, cfg=cfg, band=opt_band, current_uc_att=current_uc_att,
                                 current_tune_power=current_tune_power,
                                 stream_time=stream_time, fmin=fmin, fmax=fmax, fs=fs, nperseg=nperseg,
                                 detrend=detrend, prefix_str=prefix_str, verbose=verbose)

            if step2_index == 4 and step1_index == 4:
                verbose_print("can be fine tuned", verbose)
                estimate_att, current_tune_power, lowest_wl_index, wl_median = \
                    uc_fine_tune(S=S, cfg=cfg, band=opt_band, current_uc_att=current_uc_att,
                                 current_tune_power=current_tune_power,
                                 stream_time=stream_time, fmin=fmin, fmax=fmax, fs=fs, nperseg=nperseg,
                                 detrend=detrend, prefix_str=prefix_str, verbose=verbose)

            verbose_print(f"achieved at uc att {estimate_att} drive {current_tune_power}", verbose)

    try:
        verbose_print(f"WL: {wl_median} with {wl_length} channels out of {channel_length}", verbose)
        verbose_print(f"achieved at uc att {estimate_att} drive {current_tune_power}", verbose)
    except:
        verbose_print('WL is off, please investigate', verbose)
    verbose_print(f"plotting directory is:\n{S.plot_dir}", verbose)
    return


prefix_str = f'\n From {ufm_optimize_normal.__name__} '

if __name__ == '__main__':
    import argparse
    from sodetlib.det_config import DetConfig

    """
    The code below will only run if the file is run directly, but not if elements from this file are imported.
    For example:
        python3 ufm_optimize_quick_normal.py -args_for_argparse
    will have __name__ == '__main__' as True, and the code below will run locally.
    """
    # Seed a new parse for this file with the parser from the SMuRF controller class
    cfg = DetConfig()

    # set up the parser for this Script
    parser = argparse.ArgumentParser(description='Parser for time_stream.py script.')
    parser.add_argument('bands', type=int, metavar='bands', nargs='+', action='append',
                        help='The SMuRF bands (ints) to optimize. This is expected to be a sequence of N integers.')

    # optional arguments
    parser.add_argument('--n-scan-per-band', dest='n_scan_per_band', type=int, default=1,
                        help="int, optional, default is 1.  See n_scan argument in PySmuRF Docs for "
                             "full_band_resp() -> (int, optional, default 1) – The number of scans to take " +
                             "and average.")
    parser.add_argument('--wait-bwt-bands-sec', dest='wait_btw_bands_sec', type=float, default=5,
                        help="float, optional, default is 5. While looping over bands, wait this amount of time in " +
                             "seconds between bands.")
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
    ufm_optimize_normal(S=S, cfg=cfg, opt_band=args.band, stream_time=20.0, fmin=5, fmax=50, fs=200, nperseg=2 ** 16,
                        detrend='constant', verbose=args.verbose)
