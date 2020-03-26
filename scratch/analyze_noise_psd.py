import matplotlib
matplotlib.use('Agg')

import pysmurf.client
import argparse
import numpy as np
from scipy import signal

def analyze_noise_psd(datafile,band,chans,num_averages,flux_ramp_freq,pA_per_phi0):
    detrend = 'constant'
    nperseg = 2**16
    timestamp, phase, mask = S.read_stream_data(datafile)
    phase *= pA_per_phi0/(2.*np.pi)
    fs = flux_ramp_freq*1.0E3/num_averages
    for chan in chans:
	if chan < 0:
	    continue
	else:
	    ch_idx = mask[band,chan]
	    f, Pxx = signal.welch(phase[ch_idx], nperseg=nperseg,fs=fs, detrend=detrend)
	    Pxx = np.sqrt(Pxx)
	    popt, pcov, f_fit, Pxx_fit = S.analyze_psd(f, Pxx)
	    wl,n,f_knee = popt
	    noise[atten].append(wl)
   median_noise.append(np.median(np.asarray(noise[atten])))

    return noise, median_noise
        
if __name__=='__main__':    
    parser = argparse.ArgumentParser()

    # Arguments that are needed to create Pysmurf object
    parser.add_argument('--setup', action='store_true')
    parser.add_argument('--config-file', required=True)
    parser.add_argument('--epics-root', default='smurf_server_s2')

    # Custom arguments for this script
    parser.add_argument('--band', type=int, required=True, 
                        help='band (must be in range [0,7])')
    
    parser.add_argument('--datafile', type=int, required=True, 
                        help='.dat file from streamed data that you want to analyze psd of')

    parser.add_argument('--chans', required=True, 
                        help='Channels that you want to analyze psd of.')

    parser.add_argument('--num-averages', required=True, 
                        help='Number of flux ramp periods averaged when streaming data.')

    parser.add_argument('--flux-ramp-rate', required=True, 
                        help='Flux ramp reset rate.')

    parser.add_argument('--pA-per-phi0', required=True, 
                        help='Mutual inductance between your TES input and RF-SQUID in units of pA/phi0')

    # Parse command line arguments
    args = parser.parse_args()

    S = pysmurf.client.SmurfControl(
            epics_root = args.epics_root,
            cfg_file = args.config_file,
            setup = args.setup, make_logfile=False,
    )
