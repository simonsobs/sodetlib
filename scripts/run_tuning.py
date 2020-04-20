import matplotlib
matplotlib.use('Agg') # I put it in backend
from sodetlib.det_config import DetConfig
import pysmurf.client
import argparse
import os


def find_and_tune_freq(S, cfg, bands):
    """
    Find_freqs to identify resonance, measure eta parameters + setup channels
    using setup_notches, run serial gradient + eta to refine

    Parameters
    ----------
    S:  (pysmurf.client.SmurfControl)
        Pysmurf control instance
    cfg: (DetConfig)
        Detector config object
    bands : [int]
        bands to find tuned frequencies on. In range [0,7].
    """
    num_resonators_on = 0
    for band in bands:
        band_cfg = cfg.dev.bands[band]

        S.find_freq(band, drive_power=band_cfg['drive'], make_plot=band_cfg['make_plot'],
                    save_plot=band_cfg['save_plot'])
        S.setup_notches(band, drive=band_cfg['drive'])
        S.run_serial_gradient_descent(band)
        S.run_serial_eta_scan(band)

        num_resonators_on += len(S.which_on(band))

    tune_file = S.tune_file

    print(f"Total num resonators on: {num_resonators_on}")
    print(f"Tune file: {tune_file}")

    print("Updating config tunefile...")
    cfg.dev.update_experiment({'tunefile': tune_file})

    return num_resonators_on, tune_file


if __name__ == '__main__':
    cfg = DetConfig()

    parser = argparse.ArgumentParser()
    parser.add_argument('--bands', nargs='+', type=int,
                        help='input bands to tune as ints, separated by spaces. '
                             'Must be in range [0,7]. Defaults to tuning all '
                             'bands. Defaults to using all bands')
    args = cfg.parse_args(parser)
    S = cfg.get_smurf_control(dump_configs=True)

    if not args.bands:
        args.bands = list(range(8))

    find_and_tune_freq(S, cfg, args.bands)

    # Writes new device config over existing one.
    print(f"Writing new config over {cfg.dev_file}")
    cfg.dev.dump(os.path.abspath(os.path.expandvars(cfg.dev_file)), clobber=True)
