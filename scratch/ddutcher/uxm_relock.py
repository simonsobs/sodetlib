'''
Code written in Oct 2021 by Yuhan Wang relock UFM with a given tune file
'''
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os,sys
import sodetlib.smurf_funcs.optimize_params as op

sys.path.append('/sodetlib/scratch/ddutcher')
from noise_stack_by_band import noise_stack_by_band


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

        print('setting synthesis scale')
        # hard coding it for the current fw
        S.set_synthesis_scale(band,1)

        print('running relock')
        if not setup_notches:
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
    S.load_tune(cfg.dev.exp['tunefile'])

    op.cryo_amp_check(S, cfg)

    uxm_relock(
        S,
        cfg,
        bands=args.bands,
        setup_notches=args.setup_notches,
        estimate_phase_delay=args.estimate_phase_delay,
    )

    noise_stack_by_band(
        S,
        stream_time=20,
        fmin=5,
        fmax=50,
        detrend='constant',
    )
