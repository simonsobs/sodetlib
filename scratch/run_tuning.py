import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pysmurf.client
import argparse
import numpy as np
import pickle as pkl
from scipy import signal
import os

def optimize_uc_att_fix_dr(band,frac_pp,lms_freq,ctime):
	"""
	Finds the minimum median white noise level for all channels on in a band vs the uc attenuator value and tells you if you actually found a minimum vs uc attenuator within your range or not.

	Parameters
	----------
	band : int
		band to optimize noise on
	frac_pp : float
		fraction full scale of the FR DAC used for tracking_setup
	lms_freq : float
		tracking frequency used for tracking_setup
	ctime : str
		ctime used for saved data/plot titles    
		  
	Returns
	-------
	found_min : str
		tells you if the minimum of the noise is within your range, yes if true, low if its at uc_atten = 30, and high if its at uc_atten = 0
	median_noise_min : float
		minimum median white noise level at the optimized uc_atten value
	atten_min : int
		attenuator value which minizes the median noise
	"""
	
	attens = np.arange(30,-2,-2) #This is the full range of the attenuators
	median_noise = []
	for atten in attens:
		print('Setting UC Atten to: ',atten)	
		S.set_att_uc(band, atten)
		S.run_serial_gradient_descent(band)
		S.run_serial_eta_scan(band)
		#4kHz is the standard reset_rate, this can be changed but not to any arbitrary number. We can choose to expose this value as an argument later or not.
		S.tracking_setup(band,reset_rate_khz=4,fraction_full_scale=frac_pp, make_plot=False, save_plot=False, show_plot=False,channel=S.which_on(band), nsamp=2**18, lms_gain=6, lms_freq_hz = lms_freq,feedback_start_frac=1/12,feedback_end_frac=0.98)   
		dat_file_temp=S.take_noise_psd(30, nperseg=2**16, save_data=False,make_channel_plot = False)
		mn, od = analyze_noise_psd(band,dat_file_temp,ctime)
		median_noise.append(mn)
		if atten == 30:
			mn_initial = mn
		if mn > 4*mn_initial:
			print('Median noise is now greater than 4 times what it was at UC atten = 30 so exiting loop at uc_atten = '+str(atten))
			attens = np.arange(30,atten-2,-2)
			break
	plt.figure()
	plt.plot(attens,median_noise)
	plt.title(f'Drive = {S.get_amplitude_scale_channel(band, S.which_on(band)[0])} in Band {band}',fontsize = 18)
	plt.xlabel('UC Attenuator Value',fontsize = 14)
	plt.ylabel('Median Channel Noise [pA/rtHz]',fontsize = 14)
	plotname = os.path.join(S.plot_dir, f'{ctime}_noise_vs_uc_atten_b{band}.png')
	plt.savefig(plotname)
	S.pub.register_file(plotname, 'noise_vs_atten', plot=True)
	plt.close()

	median_noise = np.asarray(median_noise)
	med_min_arg = np.argmin(median_noise)
	found_min = 'yes'
	if med_min_arg == 0:
		found_min = 'low'
	if med_min_arg == (len(median_noise) - 1):
		found_min = 'high'
	return found_min, np.min(median_noise), attens[med_min_arg]


def analyze_noise_psd(band,dat_file,ctime):
	"""
	Finds the white noise level, 1/f knee, and 1/f polynomial exponent of a noise timestream and returns the median white noise level of all channels and a dictionary of fitted values per channel.

	Parameters
	----------
	band : int
		band to optimize noise on
	dat_file : str
		filepath to timestream data to analyze
	ctime : str
		ctime used for saved data/plot titles
		
	Returns
	-------
	median_noise : float
		median white noise level of all channels analyzed in pA/rtHz

	outdict : dict of{int:dict of{str:float}}
		dictionary with each key a channel number and each channel number another dictionary containing the fitted 1/f knee, 1/f exponent, and white noise level in pA/rtHz
	"""
	
	outdict = {}
	datafile = dat_file
	nperseg = 2**16
	detrend = 'constant'
	timestamp, phase, mask = S.read_stream_data(datafile)
	phase *= S.pA_per_phi0/(2.*np.pi)
	num_averages = S.config.get('smurf_to_mce')['num_averages']
	fs = S.get_flux_ramp_freq()*1.0E3/num_averages
	wls = []
	for chan in S.which_on(band):
		if chan < 0:
			continue
		ch_idx = mask[band,chan]
		f, Pxx = signal.welch(phase[ch_idx], nperseg=nperseg,fs=fs, detrend=detrend)
		Pxx = np.sqrt(Pxx)
		popt, pcov, f_fit, Pxx_fit = S.analyze_psd(f, Pxx)
		wl,n,f_knee = popt
		wls.append(wl)
		outdict[chan] = {}
		outdict[chan]['fknee']=f_knee
		outdict[chan]['noise index']=n
		outdict[chan]['white noise']=wl
	median_noise = np.median(np.asarray(wls))
	return median_noise, outdict
	
def optimize_power_per_band(S,band,tunefile,dr_start,frac_pp,lms_freq):
	"""
	Finds the drive power and uc attenuator value that minimizes the median noise within a band.

	Parameters
	----------
	band : int
		band to optimize noise on
	tunefile : str
		filepath to the tunefile for the band to be optimized
	dr_start : int
		drive power to start all channels with, default is 12
	frac_pp : float
		fraction full scale of the FR DAC used for tracking_setup
	lms_freq : float
		tracking frequency used for tracking_setup

	Returns
	-------
	min_med_noise : float
		The median noise at the optimized drive power
	atten : int
		Optimized uc attenuator value
	cur_dr : int
		Optimized dr value
	"""

	S.load_tune(tunefile)
	cur_dr = dr_start
	
	while True:
		ctime = S.get_timestamp()
		S.set_att_uc(band, 30)
		S.relock(band=band, drive=cur_dr)
		found_min, min_med_noise, atten  = optimize_uc_att_fix_dr(band,frac_pp,lms_freq,ctime)
		
		if found_min == "yes":
			break
		elif found_min == "low":
			cur_drive += 1
		elif found_min == "high":
			cur_drive -= 1
		
	print(f'found optimum dr = {cur_dr}, and optimum uc_att = {atten}')
	S.set_att_uc(band,atten)
	S.load_tune(tunefile)
	S.relock(band = band,drive = cur_dr)
	S.run_serial_gradient_descent(band)
	S.run_serial_eta_scan(band)
	return min_med_noise, atten, cur_dr
		
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

	parser.add_argument('--dr', type=int, default = 12, 
		help='Drive power at which to optimize the noise.')

	parser.add_argument('--frac-pp', type=float, required=True, 
			help='Fraction full scale of your flux ramp used in tracking setup.')

	parser.add_argument('--lms-freq', type=float, required=True, 
			help='Tracking frequency used in tracking setup.')

	# Parse command line arguments
	args = parser.parse_args()

	S = pysmurf.client.SmurfControl(
			epics_root = args.epics_root,
			cfg_file = args.config_file,
			setup = args.setup, make_logfile=False,
	)
	
	#Put your script calls here
	optimize_power_per_band(S,band = args.band,tunefile = args.tunefile,dr_start = args.dr,frac_pp = args.frac_pp,lms_freq = args.lms_freq)
