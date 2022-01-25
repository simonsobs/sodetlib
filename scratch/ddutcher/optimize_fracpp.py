# optimize_fracpp.py

import numpy as np
import sodetlib.util as su


def optimize_fracpp(
        S, cfg, bands=None, init_fracpp=None, n_phi0=5, update_cfg=False, apply_to_all=False,
):
    """
    Find frac_pp that yields `n_phi0` phi0 in tracking curves.

    Disables feedback, iteratively checks for desired lms_freq, then re-enables feedback.
    Can apply one fracpp to all bands in config, but runs tracking_setup only on `bands`.
    """
    if bands is None:
        bands = S.config.get("init").get("bands")
    bands = np.atleast_1d(bands)
    for band in bands:
        su.cprint(f"Band {band} frac_pp optimization", su.TermColors.HEADER)
        tk = su.get_tracking_kwargs(
            S, cfg, band, kwargs = {
                'lms_freq_hz': None, 'meas_lms_freq':True}
            )
        if init_fracpp is not None:
            tk['fraction_full_scale'] = init_fracpp
        else:
            init_fracpp = tk['fraction_full_scale']
        su.cprint(f"frac_pp used for initial tracking setup: {init_fracpp}")
        tk['make_plot'] = False

        S.set_feedback_enable(band, 0)
        stable = False
        checked_harmonic = False
        iter = 0
        while not stable:
            su.cprint(f"Iteration {iter} with feedback disabled:")
            ret = S.tracking_setup(band, **tk)
            # Calculates actual tracking params
            lms_meas = S.lms_freq_hz[band]
            lms_freq = n_phi0 * tk['reset_rate_khz'] * 1e3
            frac_pp = np.round(
                tk['fraction_full_scale'] * lms_freq / lms_meas, decimals=3
            )
            if frac_pp > 1:
                n_phi0 -= 1
                su.cprint(f"Could not achieve desired n_phi0. Reducing to {n_phi0}.")
                lms_freq = n_phi0 * tk['reset_rate_khz'] * 1e3
                frac_pp = np.round(
                    tk['fraction_full_scale'] * lms_freq / lms_meas, decimals=3
                )
            su.cprint(f"Optimum frac_pp: {frac_pp}")
            tk['fraction_full_scale'] = frac_pp
            ret = S.tracking_setup(band, **tk)
            lms_meas_again = S.lms_freq_hz[band]
            if np.abs(lms_meas - lms_meas_again) / lms_meas < 0.1:
                # We're stable, but try doubling frac_pp first
                if frac_pp < 0.5 and not checked_harmonic:
                    iter = 0
                    tk['fraction_full_scale'] = 2*frac_pp
                    su.cprint(f"Checking for aliasing by doubling frac_pp...")
                    checked_harmonic = True
                else:
                    stable = True
            elif iter > 5:
                raise OSError('Could not find stable lms_freq_hz after 5 attempts.')
            else:
                stable = False
                iter += 1
        su.cprint(f"Enabling feedback, using frac_pp={frac_pp}.", su.TermColors.OKGREEN)
        S.set_feedback_enable(band, 1)
        tk['make_plot']=True
        tk['channel'] = S.which_on(band)[::10]
        ret = S.tracking_setup(band, **tk)
        if update_cfg:
            cfg.dev.update_band(
                band,
                {'frac_pp': frac_pp, 'lms_freq_hz': lms_freq}
            )
            cfg.dev.dump(cfg.dev_file, clobber=True)

        if apply_to_all:
            su.cprint(f"Applying frac_pp={frac_pp} to all bands.", su.TermColors.HEADER)
            for band in bands:
                cfg.dev.update_band(
                    band,
                    {'frac_pp': frac_pp, 'lms_freq_hz': lms_freq}
                )
                cfg.dev.dump(cfg.dev_file, clobber=True)
            break


if __name__ == "__main__":
    import argparse
    from sodetlib.det_config import DetConfig

    parser = argparse.ArgumentParser(description='Parser for optimize_fracpp.py script.')
    parser.add_argument('--bands', type=int, nargs='+', default=None)
    parser.add_argument('--init-fracpp', type=float, default=None)
    parser.add_argument('--update-config', action='store_true', default=False)
    parser.add_argument('--apply-to-all', action='store_true', default=False)


    cfg = DetConfig()
    args = cfg.parse_args(parser)
    S = cfg.get_smurf_control(dump_configs=False, make_logfile=True)

    optimize_fracpp(
        S, cfg, bands=args.bands, init_fracpp=args.init_fracpp,
        update_cfg=args.update_config, apply_to_all=args.apply_to_all,
    )
