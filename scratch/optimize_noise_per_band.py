import matplotlib
matplotlib.use('Agg')

import pysmurf.client
import argparse
import numpy as np
import time
import sys
import pickle as pkl

def optimize_power_per_band(S,band,tunefile,dr,frac_pp):
    #This script assumes that you are already tuned beforehand and have a tunefile to use.
    S.set_att_uc(band,30)
    S.reload(tunefile)
    S.relock(band,dr)

    #By default we sweep over the full range of uc attenuators starting a the lowest attenuation up to the highest
    dat_files = {}
    median_noise = []
    noise = {}
    ctime = S.get_timestamp()
    plot_dir = S.plot_dir

    attens = np.arange(30,-2,-2)
    for atten in attens:
        print('Setting UC Atten to: ',atten)	
        S.set_att_uc(band, atten)
        S.run_serial_gradient_descent(band)
	S.run_serial_eta_scan(band)
	S.tracking_setup(band,reset_rate_khz=4,fraction_full_scale=0.7*(4/3), make_plot=True, save_plot=True, show_plot=False,channel=S.which_on(band), nsamp=2**18, lms_gain=6, lms_freq_hz = 16092,feedback_start_frac=1/12,feedback_end_frac=0.98)   
	dat_files[atten]=S.take_noise_psd(20, nperseg=2**16, save_data=True)
	noise[atten] = []
	nperseg = 2**16
	detrend = 'constant'
	timestamp, phase, mask = S.read_stream_data(dat_files[atten])
	phase *= S.pA_per_phi0/(2.*np.pi)
	num_averages = S.config.get('smurf_to_mce')['num_averages']
	fs = S.get_flux_ramp_freq()*1.0E3/num_averages
	for chan in S.which_on(band):
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

    #Save per channel noise dictionary to a pickle file    
    f_noise_per_chan = open(plot_dir.strip('/plots')+'/outputs/'+ctime+'_noise_per_chan_band'+str(band)+'.pkl','wb')
    pkl.dump(noise,f_noise_per_chan)
    f_noise_per_chan.close()

    #Save data filenames dictionary to a pickle file
    f_datafiles = open(plot_dir.strip('/plots')+'/outputs/'+ctime+'_noise_datafiles_band'+str(band)+'.pkl','wb')
    pkl.dump(dat_files,f_datafiles)
    f_datafiles.close()

    #Save median channel noise to a pickle file
    median_noise_dict = {}
    median_noise_dict['attens'] = attens
    median_noise_dict['median_noise'] = np.asarray(median_noise)
    f_median_noise = open(plot_dir.strip('/plots')+'/outputs/'+ctime+'_median_noise_vs_atten_band'+str(band)+'.txt','wb')
    pkl.dump(median_noise_dict,f_median_noise)
    f_median_noise.close()
        
if __name__=='__main__':    
    parser = argparse.ArgumentParser()

    # Arguments that are needed to create Pysmurf object
    parser.add_argument('--setup', action='store_true')
    parser.add_argument('--config-file', required=True)
    parser.add_argument('--epics-root', default='smurf_server_s2')

    # Custom arguments for this script
    parser.add_argument('--band', type=int, required=True, 
                        help='band (must be in range [0,7])')

    parser.add_argument('--tunefile', required=True, 
                        help='Tune file that you want to load for this analysis.')

    parser.add_argument('--tunefile', required=True, 
                        help='Tune file that you want to load for this analysis.')

    # Parse command line arguments
    args = parser.parse_args()

    if args.freq >= 1000:
        raise ValueError("Frequency must be less than 1 kHz ")

    S = pysmurf.client.SmurfControl(
            epics_root = args.epics_root,
            cfg_file = args.config_file,
            setup = args.setup, make_logfile=False,
    )
