# uxm_optimize_quick.py
# Use:
# python3 uxm_optimize_quick.py -h
# to see the available options and required formatting.

import time
import os, sys
import numpy as np
import scipy.signal as signal
import sodetlib.smurf_funcs.optimize_params as op
from uc_tuner import UCTuner
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

        logger.info(f"\nSetting up band {opt_band}.")

        S.set_att_dc(opt_band, cfg.dev.bands[opt_band]["dc_att"])
        logger.info(f"band {opt_band} dc_att {S.get_att_dc(opt_band)}")

        S.set_att_uc(opt_band, cfg.dev.bands[opt_band]["uc_att"])
        logger.info(f"band {opt_band} uc_att {S.get_att_uc(opt_band)}")

        S.amplitude_scale[opt_band] = cfg.dev.bands[opt_band]["drive"]
        logger.info(f"band {opt_band} tone power {S.amplitude_scale[opt_band]}")
        logger.info(f"estimating phase delay")
        try:
            S.estimate_phase_delay(opt_band)
        except ValueError:
            # Intended to catch ADC saturation but not PV timeout
            raise
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

        uctuner = UCTuner(S, cfg, band=opt_band, stream_time=stream_time,
                          fmin=fmin, fmax=fmax, detrend=detrend)
        logger.info(f"taking {stream_time}s timestream")

        uctuner.uc_tune(uc_attens=uctuner.current_uc_att, initial_attempt=True)

        if uctuner.wl_median > high_noise_thresh:
            logger.info(uctuner.status)
            raise ValueError(
                f"wl_median={uctuner.wl_median} is to high. "
                + "Something might be wrong, power level might be really off, please investigate"
            )

        elif uctuner.wl_median < low_noise_thresh:
            # Do one fine tune and stop.
            logger.info(uctuner.status)
            logger.info("Can be fine-tuned")
            uctuner.fine_tune()

        elif low_noise_thresh < uctuner.wl_median < med_noise_thresh:
            # Do a rough tune followed by a fine tune.
            logger.info(uctuner.status)
            logger.info("Can be rough-tuned")
            uctuner.rough_tune()

            # If needed, adjust the tone powers so the attenuations can have
            # some dynamic range.
            if uctuner.estimate_att < 16:
                logger.info(f"adjusting tone power and uc att")
                uctuner.increase_tone_power()

            if uctuner.estimate_att > 26:
                logger.info(f"adjusting tone power and uc att")
                uctuner.decrease_tone_power()

            uctuner.fine_tune()

            if uctuner.lowest_wl_index == 0 or uctuner.lowest_wl_index == -1:
                # Best noise was found at the edge of uc_att range explored;
                # re-center and repeat.
                logger.info(uctuner.status)
                logger.info(f"Can be fine-tuned")
                uctuner.fine_tune()

        elif med_noise_thresh < uctuner.wl_median < high_noise_thresh:
            # Do up to two rough tunes followed by one or more fine tunes.
            logger.info(uctuner.status)
            logger.info("Can be rough-tuned")
            uctuner.rough_tune()

            if uctuner.wl_median < low_noise_thresh:
                # Do one fine tune and stop.
                logger.info(uctuner.status)
                logger.info("Can be fine-tuned")
                uctuner.fine_tune()

            else:
                # Do another rough tune.
                logger.info(uctuner.status)
                logger.info("Can be rough-tuned")
                uctuner.rough_tune()

                # If needed, adjust the tone powers so the attenuations can have
                # some dynamic range.
                if uctuner.estimate_att < 16:
                    logger.info(f"adjusting tone power and uc att")
                    uctuner.increase_tone_power()

                if uctuner.estimate_att > 26:
                    logger.info(f"adjusting tone power and uc att")
                    uctuner.decrease_tone_power()

                uctuner.fine_tune()

                if uctuner.lowest_wl_index == 0 or uctuner.lowest_wl_index == -1:
                    # Best noise was found at the edge of uc_att range explored;
                    # re-center and repeat.
                    logger.info(uctuner.status)
                    logger.info(f"Can be fine-tuned")
                    uctuner.fine_tune()

        else:
            # wl_median above high_noise_thresh
            raise ValueError(f"WL={uctuner.wl_median:.1f} is off, please investigate")
            
        logger.info(uctuner.status)
        logger.info(f"achieved at uc att {uctuner.estimate_att} drive"
                    + f" {uctuner.current_tone_power}.")
        logger.info(f"plotting directory is:\n{S.plot_dir}")

        cfg.dev.update_band(
            opt_band,
            {"uc_att": uctuner.estimate_att, "drive": uctuner.current_tone_power},
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
