import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sodetlib.smurf_funcs.optimize_params as op
from sodetlib import noise


def uxm_relock(S, cfg, bands=None, setup_notches=False, estimate_phase_delay=False):
    S.all_off()
    S.set_rtm_arb_waveform_enable(0)
    S.set_filter_disable(0)
    S.set_downsample_factor(20)
    S.set_mode_dc()

    if bands is None:
        bands = S.config.get("init").get("bands")

    for band in bands:
        print('setting up band {}'.format(band))

        S.set_att_dc(band, cfg.dev.bands[band]['dc_att'])
        print('band {} dc_att {}'.format(band,S.get_att_dc(band)))

        S.set_att_uc(band, cfg.dev.bands[band]['uc_att'])
        print('band {} uc_att {}'.format(band,S.get_att_uc(band)))

        S.amplitude_scale[band] = cfg.dev.bands[band]['drive']
        print('band {} tone power {}'.format(band, S.amplitude_scale[band] ))

        if estimate_phase_delay:
            print('estimating phase delay')
            try:
                S.estimate_phase_delay(band)
            except ValueError:
                # Intended to catch ADC saturation but not PV timeout
                raise
            except Exception:
                logger.warning("Estimate phase delay failed due to PV timeout.")
        # load tune now so freq resp not overwritten by estimate_phase_delay
        # load all bands so that S.tune_file gets updated.
        S.load_tune(cfg.dev.exp['tunefile'])

        print('setting synthesis scale')
        # hard coding it for the current fw
        S.set_synthesis_scale(band,1)

        if not setup_notches:
            print('running relock')
            S.relock(band, tone_power=cfg.dev.bands[band]['drive'])
        else:
            S.load_master_assignment(band, S.freq_resp[band]['channel_assignment'])
            S.setup_notches(band, tone_power=cfg.dev.bands[band]['drive'],
                           new_master_assignment=False)

        for _ in range(3):
            S.run_serial_gradient_descent(band)
            S.run_serial_eta_scan(band)

        print('running tracking setup')
        S.set_feedback_enable(band,1) 
        S.tracking_setup(
            band,reset_rate_khz=cfg.dev.bands[band]['flux_ramp_rate_khz'],
            fraction_full_scale=cfg.dev.bands[band]['frac_pp'],
            make_plot=True, save_plot=True, show_plot=False,
            channel=S.which_on(band)[::10], nsamp=2**18,
            lms_freq_hz=cfg.dev.bands[band]["lms_freq_hz"],
            meas_lms_freq=cfg.dev.bands[band]["meas_lms_freq"],
            feedback_start_frac=cfg.dev.bands[band]['feedback_start_frac'],
            feedback_end_frac=cfg.dev.bands[band]['feedback_end_frac'],
            lms_gain=cfg.dev.bands[band]['lms_gain']
        )


if __name__ == "__main__":
    import argparse
    from sodetlib.det_config import DetConfig

    parser = argparse.ArgumentParser()
    parser.add_argument("--bands", type=int, default=None, nargs="+",
                        help="The SMuRF bands to relock on. Will default "
                        + "to the bands listed in the pysmurf config."
                        )
    parser.add_argument('--setup-notches', default=False, action='store_true',
                        help="If set, re-runs setup notches. Otherwise relock.")
    parser.add_argument('--estimate-phase-delay', default=False, action='store_true',
                        help="If set, estimate phase delay. Otherwise don't.")

    cfg = DetConfig()
    args = cfg.parse_args(parser)

    S = cfg.get_smurf_control()

    # power amplifiers
    success = op.cryo_amp_check(S, cfg)
    if not success:
        raise OSError("Health check failed")

    uxm_relock(
        S,
        cfg,
        bands=args.bands,
        setup_notches=args.setup_notches,
        estimate_phase_delay=args.estimate_phase_delay,
    )
    S.load_tune(cfg.dev.exp['tunefile'])

    acq_time = 30
    nsamps = S.get_sample_frequency() * acq_time
    nperseg = 2 ** round(np.log2(nsamps / 5))
    noise.take_noise(
        S, cfg, acq_time=acq_time, show_plot=False, save_plot=True, nperseg=nperseg
    )
