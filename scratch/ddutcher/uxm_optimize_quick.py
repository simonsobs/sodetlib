# uxm_optimize_quick.py
# Use:
# python3 uxm_optimize_quick.py -h
# to see the available options and required formatting.

import time
import os, sys
import numpy as np
import scipy.signal as signal
import sodetlib.smurf_funcs.optimize_params as op
from uc_tuner import uc_rough_tune, uc_fine_tune
import logging

sys.path.append("/sodetlib/scratch/ddutcher")
from optimize_fracpp import optimize_fracpp

logger = logging.getLogger(__name__)


def uxm_optimize(
    S,
    cfg,
    bands=None,
    low_noise_thresh=120,
    med_noise_thresh=150,
    high_noise_thresh=250,
    stream_time=20.0,
    fmin=5,
    fmax=50,
    detrend="constant",
):
    """
    Optimize tone power and uc atten for the specified bands.

    Parameters
    ----------
    S :
        Pysmurf control instance
    cfg :
        DetConfig instance
    bands : array-like
        List of SMuRF bands to optimize
    low_noise_thresh : float
        If white noise below this, do one fine adjustment to uc atten then stop.
    med_noise_thresh : float
        If white noise below this, do one rough adjustment to uc atten.
        If white noise above this, do up to two rough adjustments to uc atten.
    high_noise_thresh : float
        If white noise above this, raise ValueError. Inspect manually.

    Other Parameters
    ----------------
    stream_time, fmin, fmax, detrend:
        Used for noise taking and white noise calculation.

    See also
    --------
    uc_tuner.uc_rough_tune, uc_tuner.uc_fine_tune
    """
    logger.info(f"plotting directory is:\n{S.plot_dir}")

    if bands is None:
        bands = S.config.get("init").get("bands")

    bands = np.atleast_1d(bands)
    for opt_band in bands:
        S.all_off()
        S.set_rtm_arb_waveform_enable(0)
        S.set_filter_disable(0)
        S.set_downsample_factor(20)
        S.set_mode_dc()

        logger.info(f"Setting up band {opt_band}, the initialization band")

        S.set_att_dc(opt_band, cfg.dev.bands[opt_band]["dc_att"])
        logger.info(f"band {opt_band} dc_att {S.get_att_dc(opt_band)}")

        S.set_att_uc(opt_band, cfg.dev.bands[opt_band]["uc_att"])
        logger.info(f"band {opt_band} uc_att {S.get_att_uc(opt_band)}")

        S.amplitude_scale[opt_band] = cfg.dev.bands[opt_band]["drive"]
        logger.info(f"band {opt_band} tone power {S.amplitude_scale[opt_band]}")
        logger.info(f"estimating phase delay")
        try:
            S.estimate_phase_delay(opt_band)
        except Exception:
            logger.warning("Estimate phase delay failed due to PV timeout.")
        logger.info(f"setting synthesis scale")
        # hard coding it for the current fw
        S.set_synthesis_scale(opt_band, 1)
        logger.info(f"running find freq")
        S.find_freq(
            opt_band, tone_power=cfg.dev.bands[opt_band]["drive"], make_plot=True
        )
        logger.info(f"running setup notches")
        S.setup_notches(
            opt_band,
            tone_power=cfg.dev.bands[opt_band]["drive"],
            new_master_assignment=True,
        )
        logger.info(f"running serial gradient descent and eta scan")
        S.run_serial_gradient_descent(opt_band)
        S.run_serial_eta_scan(opt_band)
        logger.info(f"running tracking setup")
        optimize_fracpp(S, cfg, bands=opt_band, update_cfg=True)
        logger.info(f"checking tracking")
        S.check_lock(
            opt_band,
            reset_rate_khz=cfg.dev.bands[opt_band]["flux_ramp_rate_khz"],
            fraction_full_scale=cfg.dev.bands[opt_band]["frac_pp"],
            lms_freq_hz=cfg.dev.bands[opt_band]["lms_freq_hz"],
            feedback_start_frac=cfg.dev.bands[opt_band]["feedback_start_frac"],
            feedback_end_frac=cfg.dev.bands[opt_band]["feedback_end_frac"],
            lms_gain=cfg.dev.bands[opt_band]["lms_gain"],
        )

        logger.info(f"taking {stream_time}s timestream")

        # non blocking statement to start time stream and return the dat filename
        dat_path = S.stream_data_on()
        # collect stream data
        time.sleep(stream_time)
        # end the time stream
        S.stream_data_off()
        fs = S.get_sample_frequency()
        wl_list_temp = []
        timestamp, phase, mask, tes_bias = S.read_stream_data(
            dat_path, return_tes_bias=True
        )

        bands, channels = np.where(mask != -1)
        phase *= S.pA_per_phi0 / (2.0 * np.pi)  # uA

        for c, (b, ch) in enumerate(zip(bands, channels)):
            if ch < 0:
                continue
            ch_idx = mask[b, ch]
            nsamps = len(phase[ch_idx])
            f, Pxx = signal.welch(phase[ch_idx], nperseg=nsamps, fs=fs, detrend=detrend)
            Pxx = np.sqrt(Pxx)
            fmask = (fmin < f) & (f < fmax)

            wl = np.median(Pxx[fmask])
            wl_list_temp.append(wl)

        noise_param = wl_list_temp

        wl_median = np.median(noise_param)
        wl_length = len(noise_param)
        channel_length = len(noise_param)

        if wl_median > high_noise_thresh:
            logger.info(
                f"WL: {wl_median} with {wl_length} channels out of {channel_length}",
            )
            raise ValueError(
                f"wl_median={wl_median} is to high. "
                + "Something might be wrong, power level might be really off, please investigate"
            )

        elif wl_median < low_noise_thresh:
            # Do one fine tune and stop.
            logger.info(
                f"WL: {wl_median} with {wl_length} channels out of {channel_length}\ncan be fine tuned",
            )

            current_uc_att = S.get_att_uc(opt_band)
            current_tone_power = S.amplitude_scale[opt_band]

            estimate_att, current_tone_power, lowest_wl_index, wl_median = uc_fine_tune(
                S=S,
                cfg=cfg,
                band=opt_band,
                current_uc_att=current_uc_att,
                current_tone_power=current_tone_power,
                stream_time=stream_time,
                fmin=fmin,
                fmax=fmax,
                detrend=detrend,
            )

        elif low_noise_thresh < wl_median < med_noise_thresh:
            # Do a rough tune followed by a fine tune.
            logger.info(
                f"WL: {wl_median} with {wl_length} channels out of {channel_length}\ncan be rough-tuned",
            )

            current_uc_att = S.get_att_uc(opt_band)
            current_tone_power = S.amplitude_scale[opt_band]

            (
                estimate_att,
                current_tone_power,
                lowest_wl_index,
                wl_median,
            ) = uc_rough_tune(
                S=S,
                cfg=cfg,
                band=opt_band,
                current_uc_att=current_uc_att,
                current_tone_power=current_tone_power,
                stream_time=stream_time,
                fmin=fmin,
                fmax=fmax,
                detrend=detrend,
            )

            # Adjust the tone powers so the attenuations can have some dynamic range.
            if estimate_att < 16:
                logger.info(f"adjusting tone power and uc att")
                new_tone_power = current_tone_power + 2
                adjusted_uc_att = np.min([current_uc_att + 11, 30])
                S.set_att_uc(opt_band, adjusted_uc_att)
                S.find_freq(opt_band, tone_power=new_tone_power, make_plot=True)
                S.setup_notches(
                    opt_band, tone_power=new_tone_power, new_master_assignment=True
                )
                S.run_serial_gradient_descent(opt_band)
                S.run_serial_eta_scan(opt_band)
                current_uc_att = adjusted_uc_att
                current_tone_power = new_tone_power

            if estimate_att > 26:
                logger.info(f"adjusting tone power and uc att")
                new_tone_power = current_tone_power - 2
                adjusted_uc_att = np.max([current_uc_att - 11, 0])
                S.set_att_uc(opt_band, adjusted_uc_att)
                S.find_freq(opt_band, tone_power=new_tone_power, make_plot=True)
                S.setup_notches(
                    opt_band, tone_power=new_tone_power, new_master_assignment=True
                )
                S.run_serial_gradient_descent(opt_band)
                S.run_serial_eta_scan(opt_band)
                current_uc_att = adjusted_uc_att
                current_tone_power = new_tone_power

            estimate_att, current_tone_power, lowest_wl_index, wl_median = uc_fine_tune(
                S=S,
                cfg=cfg,
                band=opt_band,
                current_uc_att=current_uc_att,
                current_tone_power=current_tone_power,
                stream_time=stream_time,
                fmin=fmin,
                fmax=fmax,
                detrend=detrend,
            )
            logger.info(f"achieved at uc att {estimate_att} drive {current_tone_power}")

            step2_index = lowest_wl_index
            if step2_index == 0 or step2_index == -1:
                # Best noise was found at the edge of uc_att range explored; re-center and repeat.
                logger.info(f"can be fine tuned")
                (
                    estimate_att,
                    current_tone_power,
                    lowest_wl_index,
                    wl_median,
                ) = uc_fine_tune(
                    S=S,
                    cfg=cfg,
                    band=opt_band,
                    current_uc_att=estimate_att,
                    current_tone_power=current_tone_power,
                    stream_time=stream_time,
                    fmin=fmin,
                    fmax=fmax,
                    detrend=detrend,
                )
            logger.info(f"achieved at uc att {estimate_att} drive {current_tone_power}")

        elif med_noise_thresh < wl_median < high_noise_thresh:
            # Do up to two rough tunes followed by one or more fine tunes.
            logger.info(
                f"WL: {wl_median} with {wl_length} channels out of {channel_length}\ncan be rough tuned",
            )

            current_uc_att = S.get_att_uc(opt_band)
            current_tone_power = S.amplitude_scale[opt_band]

            (
                estimate_att,
                current_tone_power,
                lowest_wl_index,
                wl_median,
            ) = uc_rough_tune(
                S=S,
                cfg=cfg,
                band=opt_band,
                current_uc_att=current_uc_att,
                current_tone_power=current_tone_power,
                stream_time=stream_time,
                fmin=fmin,
                fmax=fmax,
                detrend=detrend,
            )

            if wl_median < low_noise_thresh:
                # Do one fine tune and stop.
                logger.info(
                    f"WL: {wl_median} with {wl_length} channels out of {channel_length}\ncan be fine tuned",
                )

                current_uc_att = S.get_att_uc(opt_band)
                current_tone_power = S.amplitude_scale[opt_band]

                (
                    estimate_att,
                    current_tone_power,
                    lowest_wl_index,
                    wl_median,
                ) = uc_fine_tune(
                    S=S,
                    cfg=cfg,
                    band=opt_band,
                    current_uc_att=current_uc_att,
                    current_tone_power=current_tone_power,
                    stream_time=stream_time,
                    fmin=fmin,
                    fmax=fmax,
                    detrend=detrend,
                )

            else:
                # Do another rough tune.
                logger.info(
                    f"WL: {wl_median} with {wl_length} channels out of {channel_length}\ncan be rough tuned",
                )

                current_uc_att = S.get_att_uc(opt_band)
                current_tone_power = S.amplitude_scale[opt_band]

                (
                    estimate_att,
                    current_tone_power,
                    lowest_wl_index,
                    wl_median,
                ) = uc_rough_tune(
                    S=S,
                    cfg=cfg,
                    band=opt_band,
                    current_uc_att=current_uc_att,
                    current_tone_power=current_tone_power,
                    stream_time=stream_time,
                    fmin=fmin,
                    fmax=fmax,
                    detrend=detrend,
                )
                step1_index = lowest_wl_index

                # Adjust the tone powers so the attenuations can have some dynamic range.
                if estimate_att < 16:
                    logger.info("adjusting tone power and uc att")
                    new_tone_power = current_tone_power + 2
                    adjusted_uc_att = np.min([current_uc_att + 12, 30])
                    S.set_att_uc(opt_band, adjusted_uc_att)
                    S.find_freq(opt_band, tone_power=new_tone_power, make_plot=True)
                    S.setup_notches(
                        opt_band, tone_power=new_tone_power, new_master_assignment=True
                    )
                    S.run_serial_gradient_descent(opt_band)
                    S.run_serial_eta_scan(opt_band)
                    current_uc_att = adjusted_uc_att
                    current_tone_power = new_tone_power

                if estimate_att > 26:
                    logger.info("adjusting tone power and uc att")
                    new_tone_power = current_tone_power - 2
                    adjusted_uc_att = np.max([current_uc_att - 11, 0])
                    S.set_att_uc(opt_band, adjusted_uc_att)
                    S.find_freq(opt_band, tone_power=new_tone_power, make_plot=True)
                    S.setup_notches(
                        opt_band, tone_power=new_tone_power, new_master_assignment=True
                    )
                    S.run_serial_gradient_descent(opt_band)
                    S.run_serial_eta_scan(opt_band)
                    current_uc_att = adjusted_uc_att
                    current_tone_power = new_tone_power

                (
                    estimate_att,
                    current_tone_power,
                    lowest_wl_index,
                    wl_median,
                ) = uc_fine_tune(
                    S=S,
                    cfg=cfg,
                    band=opt_band,
                    current_uc_att=current_uc_att,
                    current_tone_power=current_tone_power,
                    stream_time=stream_time,
                    fmin=fmin,
                    fmax=fmax,
                    detrend=detrend,
                )
                logger.info(
                    f"achieved at uc att {estimate_att} drive {current_tone_power}",
                )

                step2_index = lowest_wl_index
                if step2_index == 0 or step2_index == -1:
                    # Best noise was found at the edge of uc_att range explored; re-center and repeat.
                    logger.info(f"can be fine tuned")
                    (
                        estimate_att,
                        current_tone_power,
                        lowest_wl_index,
                        wl_median,
                    ) = uc_fine_tune(
                        S=S,
                        cfg=cfg,
                        band=opt_band,
                        current_uc_att=estimate_att,
                        current_tone_power=current_tone_power,
                        stream_time=stream_time,
                        fmin=fmin,
                        fmax=fmax,
                        detrend=detrend,
                    )

        else:
            # wl_median above high_noise_thresh
            raise ValueError(f"WL={wl_median} is off, please investigate")
            
        logger.info(
            f"WL: {wl_median} with {wl_length} channels out of {channel_length}",
        )
        logger.info(
            f"achieved at uc att {estimate_att} drive {current_tone_power}",
        )
        logger.info(f"plotting directory is:\n{S.plot_dir}")

        cfg.dev.update_band(
            opt_band,
            {"uc_att": estimate_att, "drive": current_tone_power},
            update_file=True,
        )


if __name__ == "__main__":
    import argparse
    from sodetlib.det_config import DetConfig
    # Seed a new parse for this file with the parser from the SMuRF controller class
    cfg = DetConfig()

    # set up the parser for this Script
    parser = argparse.ArgumentParser(
        description="Parser for uxm_optimize_quick.py script."
    )
    parser.add_argument(
        "assem_type",
        type=str,
        choices=["ufm", "umm"],
        default="ufm",
        help="Assembly type, ufm or umm. Determines the relevant noise thresholds.",
    )
    parser.add_argument(
        "--bands",
        type=int,
        nargs="+",
        default=None,
        help="The SMuRF bands to optimize on.",
    )

    # optional arguments
    parser.add_argument(
        "--stream-time",
        dest="stream_time",
        type=float,
        default=20.0,
        help="float, optional, default is 20.0. The amount of time to sleep in seconds while "
        + "streaming SMuRF data for analysis.",
    )
    parser.add_argument(
        "--fmin",
        dest="fmin",
        type=float,
        default=5.0,
        help="float, optional, default is 5.0. The lower frequency (Hz) bound used "
        + "when creating a mask of white noise levels. Suggested value of 5.0",
    )
    parser.add_argument(
        "--fmax",
        dest="fmax",
        type=float,
        default=50.0,
        help="float, optional, default is 50.0. The upper frequency (Hz) bound used "
        + "when creating a mask of white noise levels Suggested value of 50.0",
    )
    parser.add_argument(
        "--detrend",
        dest="detrend",
        default="constant",
        help="str, optional, default is 'constant'. Passed to scipy.signal.welch.",
    )
    parser.add_argument(
        "--loglevel",
        type=str.upper,
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the log level for printed messages. The default is pulled from "
        + "$LOGLEVEL, defaulting to INFO if not set.",
    )

    # parse the args for this script
    args = cfg.parse_args(parser)

    if args.loglevel is None:
        args.loglevel = os.environ.get("LOGLEVEL", "INFO")
    numeric_level = getattr(logging, args.loglevel)
    logging.basicConfig(
        format="%(levelname)s: %(funcName)s: %(message)s", level=numeric_level
    )

    S = cfg.get_smurf_control(dump_configs=True, make_logfile=(numeric_level != 10))

    if args.assem_type == "ufm":
        high_noise_thresh = 250
        med_noise_thresh = 150
        low_noise_thresh = 120
    elif args.assem_type == "umm":
        high_noise_thresh = 250
        med_noise_thresh = 65
        low_noise_thresh = 45
    else:
        raise ValueError("Assembly must be either 'ufm' or 'umm'.")

    # power amplifiers
    op.cryo_amp_check(S, cfg)

    # run the def in this file
    uxm_optimize(
        S=S,
        cfg=cfg,
        bands=args.bands,
        stream_time=args.stream_time,
        fmin=args.fmin,
        fmax=args.fmax,
        low_noise_thresh=low_noise_thresh,
        med_noise_thresh=med_noise_thresh,
        high_noise_thresh=high_noise_thresh,
        detrend=args.detrend,
    )
