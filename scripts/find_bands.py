
import matplotlib
matplotlib.use('Agg')
import pysmurf.client
import argparse
import numpy as np
from sodetlib.det_config import DetConfig
import os

def find_bands(S, cfg, bands=np.arange(8)):
    """
    Do a noise sweep to find the coarse position of resonators. Return a dictionary of resonator locations.
    ----------
    S : pysmurf.client.SmurfControl
        Smurf control object
    cfg : DetConfig
        sodetlib config object
    bands : int array
        bands to find resonators in. Default is 0 to 8.

    Returns
    -------
    resonances : dict
        A dictionary of {band:[list of subbands]} for each resonator in MHz.
    """
    resonances = {}
    for band in bands:
        band_cfg = cfg.dev.bands[band]
        freq, resp = S.full_band_resp(band)
        peaks = S.find_peak(freq, resp, make_plot=band_cfg['make_plots'],
                            show_plot=band_cfg['show_plots'], band=band)
        f = np.array(peaks*1.0E-6) + S.get_band_center_mhz(band)
        subbands, channels, offsets = S.assign_channels(f, band=band, as_offset=False)
        resonances[band] = subbands
        cfg.dev.update_band(band, {'resonances': subbands})
    return resonances


if __name__ == '__main__':
    cfg = DetConfig()
    parser = argparse.ArgumentParser()

    # Custom arguments for this script
    parser.add_argument('--bands', type=int, nargs='+', required=True,
                        help='range of bands to find resonators, separate by spaces')
    args = cfg.parse_args(parser)
    S = cfg.get_smurf_control(dump_configs=True)

    resonances = find_bands(S, cfg, bands=args.bands)
    cfg.dev.dump(os.path.abspath(os.path.expandvars(cfg.dev_file)),
                 clobber=True)
