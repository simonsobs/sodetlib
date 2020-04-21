
import matplotlib
matplotlib.use('Agg')
import pysmurf.client
import argparse
import numpy as np
import os


def find_bands(S,bands=np.arange(8)):
	"""
	Do a noise sweep to find the coarse position of resonators. Return a dictionary of resonator locations.
	----------
	bands : int array
		bands to find resonators in. Default is 0 to 8. 
		  
	Returns
	-------
	resonances : dict
		A dictionary of {band:[list of subbands]} for each resonator
	"""

	resonances={}

	for band in bands:
		freq, resp = S.full_band_resp(band)
		peaks = S.find_peak(freq, resp, make_plot=False, show_plot=False, band=band)
		f=np.array(peaks*1.0E-6)+S.get_band_center_mhz(band)
		subbands, channels, offsets = S.assign_channels(f, band=band, as_offset=False)
		resonances[band]=subbands

	return resonances


if __name__=='__main__':    
    parser = argparse.ArgumentParser()

    # Arguments that are needed to create Pysmurf object
    parser.add_argument('--setup', action='store_true')
    parser.add_argument('--config-file', required=True)
    parser.add_argument('--epics-root', default='smurf_server_s2')

    # Custom arguments for this script
    parser.add_argument('--bands', type=int, required=True, 
                        help='range of bands to find resonators, separate by spaces')


    # Parse command line arguments
    args = parser.parse_args()

    S = pysmurf.client.SmurfControl(
            epics_root = args.epics_root,
            cfg_file = args.config_file,
            setup = args.args.setup, make_logfile=False,
    )

    find_bands(S, bands=args.bands)






