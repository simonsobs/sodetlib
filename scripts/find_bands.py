
import matplotlib
matplotlib.use('Agg')
import pysmurf.client
import argparse
import numpy as np
from sodetlib.det_config import DetConfig
import os

def find_bands(S, cfg):
    """
    Do a noise sweep to find the coarse position of resonators. 
    Return active bands and a dictionary of active subbands.
    ----------
    S : pysmurf.client.SmurfControl
        Smurf control object
    cfg : DetConfig
        sodetlib config object

    Returns
    -------
    bands : int array
        Active bands

    subband_dict : dict
        A dictionary of {band:[list of subbands]} for each resonator in MHz.
    """
    bands=np.array([])
    subband_dict = {}

    AMC=S.which.bays()

    if len(AMC)==2:
        bands=np.arange(8)
    elif len(AMC)==1:
        bands=np.arange(4)
    else:
        print('No active AMC')
        return bands, subband_dict

    for band in bands:

        band_cfg = cfg.dev.bands[band]

        freq, resp = S.full_band_resp(band)
        peaks = S.find_peak(freq, resp, make_plot=band_cfg['make_plots'],
                            show_plot=band_cfg['show_plots'], band=band)
        f = np.array(peaks*1.0E-6) + S.get_band_center_mhz(band)
        subbands=S.freq_to_subband(band,f)[0] 
        subbands=np.unique(subbands)

        subband_dict[band]={}
        subband_dict[band] = subbands

        cfg.dev.update_band(band, {'resonances': subbands})

    return bands, subband_dict


if __name__ == '__main__':
    cfg = DetConfig()
    parser = argparse.ArgumentParser()

    # Custom arguments for this script

    args = cfg.parse_args(parser)
    S = cfg.get_smurf_control(dump_configs=True)

    bands, subband_dict = find_bands(S, cfg)
    cfg.dev.dump(os.path.abspath(os.path.expandvars(cfg.dev_file)),
                 clobber=True)
