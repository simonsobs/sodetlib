import matplotlib
matplotlib.use('Agg') # I put it in backend
from sodetlib.det_config import DetConfig
import pysmurf.client
import argparse
import os


def find_and_tune_freq(S, cfg, bands, plotname_append='', new_master_assignment=True):
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

    Optional parameters
    ----------
    plotname_append : str
        Appended to the default plot filename. Default ''.
    new_master_assignment : bool
        Whether to create a new master assignment (tuning)
        file. This file defines the mapping between resonator frequency
        and channel number. Default True.
    """
    num_resonators_on = 0
    for band in bands:
        band_cfg = cfg.dev.bands[band]

        S.find_freq(band, drive_power=band_cfg['drive'],
                    make_plot=band_cfg['make_plot'],
                    save_plot=band_cfg['save_plot'], # Expecting to add a subband argument from config in here at some point
                    plotname_append=plotname_append)
        S.setup_notches(band, drive=band_cfg['drive'],
                    new_master_assignment=new_master_assignment)
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
    parser.add_argument('--plotname-append', type=str, default='',
        help="Appended to the default plot filename. Default is ''.")
    parser.add_argument('--new-master-assignment', type=bool, default=True,
        help='Whether to create a new master assignment file. This file defines the mapping between resonator frequency and channel number. Default True.')

    args = cfg.parse_args(parser)
    S = cfg.get_smurf_control(dump_configs=True)

    if not args.bands:
        args.bands = list(range(8))

    find_and_tune_freq(S, cfg, args.bands)

    # Writes new device config over existing one.
    print(f"Writing new config over {cfg.dev_file}")
    cfg.dev.dump(os.path.abspath(os.path.expandvars(cfg.dev_file)), clobber=True)
