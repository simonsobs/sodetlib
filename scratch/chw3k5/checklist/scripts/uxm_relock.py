"""
The core code is from Princeton, created by Yuhan Wang and Daniel Dutcher Oct/Nov 2021.

The code was refactored by Caleb Wheeler Nov 2021 to support argparse.
The argparse implementation (found at the bottom of the file) and provides a minimal
documentation framework.

Use:

python3 uxm_relock.py -h

to see the available options and required formatting.
"""
import os
import time

import numpy as np
from scipy import signal
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def verbose_print(msg, verbose=True):
    if verbose:
        print(f"{prefix_str} {msg}")
    return


def uxm_relock(S, cfg, fav_tune_files=None, bands=None, stream_time=20.0, fmin=5, fmax=50, fs=200, nperseg=2 ** 16,
               detrend='constant', verbose=True):
    if verbose:
        verbose_print('plotting directory is:\n{S.plot_dir}', verbose)
    if bands is None:
        bands = S.config.get('init').get('bands')
    if fav_tune_files is None:
        tune_dir = '/data/smurf_data/tune/'
        # this orders the files in the tune directory, most recent time stamps first.
        all_tune_files = sorted(os.listdir(tune_dir), reverse=True)
        for file_in_tune_dir in all_tune_files:
            if '_tune.npy' in file_in_tune_dir:
                fav_tune_files = os.path.join(tune_dir, file_in_tune_dir)
                break
        else:
            # only happened when the break statement is never triggered.
            raise FileNotFoundError(f"No tune files were found in {tune_dir}")

    S.all_off()
    S.set_rtm_arb_waveform_enable(0)
    S.set_filter_disable(0)
    S.set_downsample_factor(20)
    S.set_mode_dc()

    S.load_tune(fav_tune_files)

    for band in bands:
        verbose_print('setting up band {}'.format(band), verbose)

        S.set_att_dc(band, cfg.dev.bands[band]['dc_att'])
        verbose_print('band {} dc_att {}'.format(band, S.get_att_dc(band)), verbose)

        S.set_att_uc(band, cfg.dev.bands[band]['uc_att'])
        verbose_print('band {} uc_att {}'.format(band, S.get_att_uc(band)), verbose)

        S.amplitude_scale[band] = cfg.dev.bands[band]['drive']
        verbose_print('band {} tone power {}'.format(band, S.amplitude_scale[band]), verbose)

        verbose_print('setting synthesis scale', verbose)
        # hard coding it for the current fw
        S.set_synthesis_scale(band, 1)

        verbose_print('running relock', verbose)
        S.relock(band, tone_power=cfg.dev.bands[band]['drive'])

        S.run_serial_gradient_descent(band);
        S.run_serial_eta_scan(band);

        verbose_print('running tracking setup', verbose)
        S.set_feedback_enable(band, 1)
        S.tracking_setup(band, reset_rate_khz=cfg.dev.bands[band]['flux_ramp_rate_khz'],
                         fraction_full_scale=cfg.dev.bands[band]['frac_pp'], make_plot=False, save_plot=False,
                         show_plot=False, channel=S.which_on(band), nsamp=2 ** 18,
                         lms_freq_hz=cfg.dev.bands[band]["lms_freq_hz"],
                         meas_lms_freq=cfg.dev.bands[band]["meas_lms_freq"],
                         feedback_start_frac=cfg.dev.bands[band]['feedback_start_frac'],
                         feedback_end_frac=cfg.dev.bands[band]['feedback_end_frac'],
                         lms_gain=cfg.dev.bands[band]['lms_gain'])
    #	print('checking tracking')
    #	S.check_lock(band,reset_rate_khz=cfg.dev.bands[band]['flux_ramp_rate_khz'],fraction_full_scale=cfg.dev.bands[band]['frac_pp'], lms_freq_hz=cfg.dev.bands[band]["lms_freq_hz"], feedback_start_frac=cfg.dev.bands[band]['feedback_start_frac'],feedback_end_frac=cfg.dev.bands[band]['feedback_end_frac'],lms_gain=cfg.dev.bands[band]['lms_gain'])

    verbose_print(f'taking {stream_time}s timestream', verbose)

    # non blocking statement to start time stream and return the dat filename
    dat_path = S.stream_data_on()
    # collect stream data
    time.sleep(stream_time)
    # end the time stream
    S.stream_data_off()

    start_time = dat_path[-14:-4]

    timestamp, phase, mask, tes_bias = S.read_stream_data(dat_path, return_tes_bias=True)
    verbose_print(f'loaded the .dat file at: {dat_path}', verbose)

    # hard coded variables
    bands, channels = np.where(mask != -1)
    phase *= S.pA_per_phi0 / (2.0 * np.pi)  # uA
    sample_nums = np.arange(len(phase[0]))
    t_array = sample_nums / fs

    # reorganize the data by band then channel
    stream_by_band_by_channel = {}
    for band, channel in zip(bands, channels):
        if band not in stream_by_band_by_channel.keys():
            stream_by_band_by_channel[band] = {}
        ch_idx = mask[band, channel]
        stream_by_band_by_channel[band][channel] = phase[ch_idx]

    # plot the band channel data
    fig, axs = plt.subplots(4, 2, figsize=(12, 24), gridspec_kw={'width_ratios': [2, 2]}, dpi=50)
    for band in sorted(stream_by_band_by_channel.keys()):
        wl_list_temp = []
        stream_single_band = stream_by_band_by_channel[band]
        ax_this_band = axs[band // 2, band % 2]
        for channel in sorted(stream_single_band.keys()):
            stream_single_channel = stream_single_band[channel]

            f, Pxx = signal.welch(stream_single_channel, fs=fs, detrend=detrend, nperseg=nperseg)
            Pxx = np.sqrt(Pxx)
            fmask = (fmin < f) & (f < fmax)
            wl = np.median(Pxx[fmask])
            wl_list_temp.append(wl)
            stream_single_channel_norm = stream_single_channel - np.mean(stream_single_channel)
            ax_this_band.plot(t_array, stream_single_channel_norm, color='C0', alpha=0.002)
        wl_median = np.median(wl_list_temp)
        band_yield = len(stream_single_band)
        ax_this_band.set_xlabel('time [s]')
        ax_this_band.set_ylabel('Phase [pA]')
        ax_this_band.grid()
        ax_this_band.set_title(f'band {band} yield {band_yield} median noise {wl_median:.2f}')
        ax_this_band.set_ylim([-10000, 10000])

    save_name = f'{start_time}_band_noise_stack.png'
    verbose_print(f'Saving plot to {os.path.join(S.plot_dir, save_name)}', verbose)
    plt.savefig(os.path.join(S.plot_dir, save_name))

    fig, axs = plt.subplots(4, 4, figsize=(24, 24), gridspec_kw={'width_ratios': [2, 2, 2, 2]}, dpi=50)
    for band in sorted(stream_by_band_by_channel.keys()):
        wl_list_temp = []
        stream_single_band = stream_by_band_by_channel[band]
        ax_this_band = axs[band // 2, band % 2 * 2]
        for channel in sorted(stream_single_band.keys()):
            stream_single_channel = stream_single_band[channel]
            f, Pxx = signal.welch(stream_single_channel,
                                  fs=fs, detrend=detrend, nperseg=nperseg)
            Pxx = np.sqrt(Pxx)
            fmask = (fmin < f) & (f < fmax)
            wl = np.median(Pxx[fmask])
            wl_list_temp.append(wl)
            ax_this_band.loglog(f, Pxx, color='C0', alpha=0.2)

        wl_median = np.median(wl_list_temp)

        band_yield = len(stream_single_band)
        ax_this_band.set_xlabel('Frequency [Hz]')
        ax_this_band.set_ylabel('Amp [pA/rtHz]')
        ax_this_band.grid()
        ax_this_band.axvline(1.4, linestyle='--', alpha=0.6, label='1.4 Hz', color='C1')
        ax_this_band.axvline(60, linestyle='--', alpha=0.6, label='60 Hz', color='C2')
        ax_this_band.set_title(f'band {band} yield {band_yield}')
        ax_this_band.set_ylim([1, 5e3])

        ax_this_band_2 = axs[band // 2, band % 2 * 2 + 1]
        ax_this_band_2.set_xlabel('Amp [pA/rtHz]')
        ax_this_band_2.set_ylabel('count')
        ax_this_band_2.hist(wl_list_temp, range=(0, 300), bins=60)
        ax_this_band_2.axvline(wl_median, linestyle='--', color='gray')
        ax_this_band_2.grid()
        ax_this_band_2.set_title(f'band {band} yield {band_yield} median noise {wl_median:.2f}')
        ax_this_band_2.set_xlim([0, 300])

    save_name = f'{start_time}_band_psd_stack.png'
    verbose_print(f'Saving plot to {os.path.join(S.plot_dir, save_name)}', verbose)
    plt.savefig(os.path.join(S.plot_dir, save_name))

    # S.save_tune()

    verbose_print(f'plotting directory is:\n{S.plot_dir}', verbose)
    return


prefix_str = f'\n From {uxm_relock.__name__}'

if __name__ == "__main__":
    import argparse
    from sodetlib.det_config import DetConfig

    """
    The code below will only run if the file is run directly, but not if elements from this file are imported.
    For example:
        python3 uxm_relock.py -args_for_argparse
    will have __name__ == '__main__' as True, and the code below will run locally.
    """
    # Seed a new parse for this file with the parser from the SMuRF controller class
    cfg = DetConfig()

    # set up the parser for this Script
    # python3 /sodetlib/scratch/chw3k5/checklist/scripts/uxm_relock.py 0 1 2 3 4 5 6 7 --tune-file-path latest --slot 3
    # --stream-time 20.0 --fmin 5.0 --fmax 50.0 --fs 200.0 --nperseg 65536 --detrend constant
    parser = argparse.ArgumentParser(description='Parser for ufm_optimize_quick.py script.')
    parser.add_argument('bands', type=int, metavar='bands', nargs='+', action='append',
                        help='Required, N additional positional arguments, int, separated by a space.' +
                             'SMuRF bands to target.')

    # optional arguments
    parser.add_argument('--tune-file-path', metavar='path', dest='tune_file_path', default='latest',
                        help='Optional, str, default is latest. The full path to the tune file for relock.' +
                             "Use the keyword 'latest' get the most recent tune file.")
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
    if args.tune_file_path == 'latest':
        fav_tune_files = None
    else:
        fav_tune_files = args.tune_file_path
    uxm_relock(S=S, cfg=cfg, fav_tune_files=fav_tune_files, bands=args.bands[0], stream_time=args.stream_time,
               fmin=args.fmin, fmax=args.fmax, fs=args.fs, nperseg=args.nperseg,
               detrend=args.detrend, verbose=args.verbose)
