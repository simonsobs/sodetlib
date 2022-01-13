# uxm_setup.py
#
# Use:
# python3 uxm_setup.py -h
# to see the available options and required formatting.

import os, sys
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import sodetlib.smurf_funcs.optimize_params as op
import logging

sys.path.append('/sodetlib/scratch/ddutcher')
from noise_stack_by_band import noise_stack_by_band

logger = logging.getLogger(__name__)

def uxm_setup(S, cfg, bands=None):
    """
    Use values in cfg to setup UXM for use.
    """
    logger.info(f"plotting directory is:\n{S.plot_dir}")
    if bands is None:
        bands = S.config.get("init").get("bands")

    S.all_off()
    S.set_rtm_arb_waveform_enable(0)
    S.set_filter_disable(0)
    S.set_downsample_factor(20)
    S.set_mode_dc()

    for band in bands:
        logger.info(f"setting up band {band}")

        S.set_att_dc(band, cfg.dev.bands[band]["dc_att"])
        logger.info("band {} dc_att {}".format(band, S.get_att_dc(band)))

        S.set_att_uc(band, cfg.dev.bands[band]["uc_att"])
        logger.info("band {} uc_att {}".format(band, S.get_att_uc(band)))

        S.amplitude_scale[band] = cfg.dev.bands[band]["drive"]
        logger.info(
            "band {} tone power {}".format(band, S.amplitude_scale[band])
        )

        # logger.info("estimating phase delay")
        # try:
        #     S.estimate_phase_delay(band)
        # except Exception:
        #     logger.warning('Estimate phase delay failed due to PV timeout.')
        logger.info("setting synthesis scale")
        # hard coding it for the current fw
        S.set_synthesis_scale(band, 1)
        logger.info("running find freq")
        S.find_freq(band, tone_power=cfg.dev.bands[band]["drive"], make_plot=True)
        logger.info("running setup notches")
        S.setup_notches(
            band, tone_power=cfg.dev.bands[band]["drive"], new_master_assignment=True
        )
        logger.info("running serial gradient descent and eta scan")
        S.run_serial_gradient_descent(band)
        S.run_serial_eta_scan(band)
        logger.info("running tracking setup")
        S.set_feedback_enable(band, 1)
        S.tracking_setup(
            band,
            reset_rate_khz=cfg.dev.bands[band]["flux_ramp_rate_khz"],
            fraction_full_scale=cfg.dev.bands[band]["frac_pp"],
            make_plot=True,
            save_plot=True,
            show_plot=False,
            channel=S.which_on(band)[::10],
            nsamp=2 ** 18,
            lms_freq_hz=cfg.dev.bands[band]["lms_freq_hz"],
            meas_lms_freq=cfg.dev.bands[band]["meas_lms_freq"],
            feedback_start_frac=cfg.dev.bands[band]["feedback_start_frac"],
            feedback_end_frac=cfg.dev.bands[band]["feedback_end_frac"],
            lms_gain=cfg.dev.bands[band]["lms_gain"],
        )

    S.save_tune()
    cfg.dev.update_experiment({'tunefile': S.tune_file}, update_file=True)


if __name__ == "__main__":
    import argparse
    from sodetlib.det_config import DetConfig
    
    cfg = DetConfig()

    # set up the parser for this Script
    parser = argparse.ArgumentParser(
        description="Parser for uxm_setup.py script."
    )
    parser.add_argument(
        "--bands",
        type=int,
        default=None,
        nargs="+",
        help="The SMuRF bands to target. Will default to the bands "
        + "listed in the pysmurf configuration file."
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
        + "when creating a mask of white noise levels Suggested value of 5.0",
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
        choices=['DEBUG','INFO','WARNING','ERROR','CRITICAL'],
        help="Set the log level for printed messages. The default is pulled from "
        +"$LOGLEVEL, defaulting to INFO if not set.",
    )

    # parse the args for this script
    args = cfg.parse_args(parser)
    if args.loglevel is None:
        args.loglevel = os.environ.get("LOGLEVEL","INFO")
    numeric_level = getattr(logging, args.loglevel)
    logging.basicConfig(
        format="%(levelname)s: %(funcName)s: %(message)s", level=numeric_level
    )
    
    S = cfg.get_smurf_control(dump_configs=True, make_logfile=(numeric_level != 10))

    # power amplifiers
    op.cryo_amp_check(S, cfg)
    # run the defs in this file
    uxm_setup(S=S, cfg=cfg, bands=args.bands)
    # plot noise histograms
    noise_stack_by_band(
        S,
        stream_time=args.stream_time,
        fmin=args.fmin,
        fmax=args.fmax,
        detrend=args.detrend,
    )
