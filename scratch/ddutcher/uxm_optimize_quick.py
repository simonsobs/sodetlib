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
import matplotlib
matplotlib.use('Agg')

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

    See also
    --------
    uc_tuner.UCTuner
    """
    logger.info(f"plotting directory is:\n{S.plot_dir}")

    if bands is None:
        bands = S.config.get("init").get("bands")

    bands = np.atleast_1d(bands)
    for opt_band in bands:
        # Do initial setup steps.
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

        uctuner = UCTuner(S, cfg, band=opt_band)

        uctuner.uc_tune(uc_attens=uctuner.current_uc_att)
        logger.info(f"UC attens: {uctuner.uc_attens}")
        logger.info(f"Noise levels: {uctuner.wl_list}")
        logger.info(f"Number of active channels: {uctuner.wl_length}")

        if uctuner.wl_median > high_noise_thresh:
            raise ValueError(
                f"wl_median={uctuner.wl_median} is to high. "
                + "Something might be wrong, power level might be really off, please investigate"
            )

        elif uctuner.wl_median < low_noise_thresh:
            # Do one fine tune and stop.
            status_tune(logger, uctuner, "fine")

        elif low_noise_thresh < uctuner.wl_median < med_noise_thresh:
            # Do a rough tune followed by a fine tune.
            status_tune(logger, uctuner, "rough")

            # If needed, adjust the tone powers so the attenuations can have
            # some dynamic range.
            if uctuner.estimate_att < 4:
                uctuner.increase_tone_power()
                logger.info(f"Adjusting tone power to {uctuner.current_tone_power}"
                            + f" and uc_att to {uctuner.current_uc_att}"
                )

            if uctuner.estimate_att > 26:
                uctuner.decrease_tone_power()
                logger.info(f"Adjusting tone power to {uctuner.current_tone_power}"
                            + f" and uc_att to {uctuner.current_uc_att}"
                )

            status_tune(logger, uctuner, "fine")

            if uctuner.lowest_wl_index == 0 or uctuner.lowest_wl_index == -1:
                # Best noise was found at the edge of uc_att range explored;
                # re-center and repeat.
                uctuner.current_uc_att = uctuner.estimate_att
                status_tune(logger, uctuner, "fine")

        elif med_noise_thresh < uctuner.wl_median < high_noise_thresh:
            # Do up to two rough tunes followed by one or more fine tunes.
            status_tune(logger, uctuner, "rough")

            if uctuner.wl_median < low_noise_thresh:
                # Do one fine tune and stop.
                status_tune(logger, uctuner, "fine")

            else:
                # Do another rough tune.
                status_tune(logger, uctuner, "rough")

                # If needed, adjust the tone powers so the attenuations can have
                # some dynamic range.
                if uctuner.estimate_att < 4:
                    uctuner.increase_tone_power()
                    logger.info(f"Adjusting tone power to {uctuner.current_tone_power}"
                                + f" and uc_att to {uctuner.current_uc_att}"
                    )

                if uctuner.estimate_att > 26:
                    uctuner.decrease_tone_power()
                    logger.info(f"Adjusting tone power to {uctuner.current_tone_power}"
                                + f" and uc_att to {uctuner.current_uc_att}"
                    )

                status_tune(logger, uctuner, "fine")

                if uctuner.lowest_wl_index == 0 or uctuner.lowest_wl_index == -1:
                    # Best noise was found at the edge of uc_att range explored;
                    # re-center and repeat.
                    uctuner.current_uc_att = uctuner.estimate_att
                    status_tune(logger, uctuner, "fine")

        else:
            # wl_median above high_noise_thresh
            raise ValueError(f"WL={uctuner.wl_median:.1f} is off, please investigate")
            
        logger.info(uctuner.status)
        logger.info(f"Best noise {uctuner.best_wl:.1f} pA/rtHz achieved at"
                    + f" uc att {uctuner.best_att} drive {uctuner.best_tone}."
        )
        logger.info(f"plotting directory is:\n{S.plot_dir}")

        cfg.dev.update_band(
            opt_band,
            {"uc_att": uctuner.best_att, "drive": uctuner.best_tone},
            update_file=True,
        )


def status_tune(logger, uctuner, rof):
    ''' Declares rough/fine tunability and uctuner status, then tunes.
    rof = string, either "rough" or "fine", indicating tuning type.'''
    logger.info(f"Can be {rof}-tuned")
    if rof == "fine":
        uctuner.fine_tune()
    elif rof == "rough":
        uctuner.rough_tune()
    else:
        raise ArgumentError(f'tune type must be "rough" or "fine"; was given "{rof}"')
    logger.info(f"UC attens: {uctuner.uc_attens}")
    logger.info(f"Noise levels: {uctuner.wl_list}")
    logger.info(f"Number of active channels: {uctuner.wl_length}")


if __name__ == "__main__":
    import argparse
    from sodetlib.det_config import DetConfig

    cfg = DetConfig()

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
    success = op.cryo_amp_check(S, cfg)
    if not success:
        raise OSError("Health check failed")

    # run the def in this file
    uxm_optimize(
        S=S,
        cfg=cfg,
        bands=args.bands,
        low_noise_thresh=low_noise_thresh,
        med_noise_thresh=med_noise_thresh,
        high_noise_thresh=high_noise_thresh,
    )
