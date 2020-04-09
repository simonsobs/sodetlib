import matplotlib
matplotlib.use('Agg')

import pysmurf.client
import argparse
import matplotlib.pyplot as plt
import time
import numpy as np
import scipy.signal as signal
import scipy.optimize as opt
import pickle as pkl
pi = np.pi

def serial_corr(wave, lag=1):
	"""
	Adapted from SWH squid fitting code. Calculates the correlation 
	coefficient normalized between 0 to 1 at a fixed lag.

	Parameters
	----------
	wave : array of float
		time stream to autocorrelate
	lag : int
		number of samples delay between the two timestreams at which you
		calculate the coefficient at.
		
	Returns
	-------
	corr : float
		Normalized correlation coefficient between 0 & 1.
	"""
		
	n = len(wave)
	y1 = wave[lag:]
	y2 = wave[:n-lag]
	corr = np.corrcoef(y1, y2)[0, 1]
	return corr

def autocorr(wave):
	"""
	Adapted from SWH squid fitting code. Calculates the autocorrelation.

	Parameters
	----------
	wave : array of float
		time stream to autocorrelate
		
	Returns
	-------
	lags : tuple of ints
		The lags at which the autocorrelation is calculated
	corr : array of floats
		Array of normalized correlation coefficients for each lag
	"""
	lags = range(len(wave)//2)
	corrs = np.array([serial_corr(wave, lag) for lag in lags])
	return lags, corrs
	
def sine_fit(x,a0,a1,ph1,f):
	"""
	Sine wave.

	Parameters
	----------
	x : array of float
		points to evaluate the sine function at
	a0 : float
		sine offset from 0
	a1 : float
		sine amplitude
	ph1 : float
		sine phase
	f : float
		sine frequency
		
	Returns
	-------
	sine : array of float
		Sine wave with given parameters evaluated at all x points.
	"""
	return a0 + a1*np.sin(2*np.pi*f*x + ph1)

def calc_R_tickle(S, band, bias_group, tickle_voltage,channels):
	"""
	Takes a tickle measurement and calculates the small signal resistance
	from it.
	
	Parameters
	----------
	band : int
		band to optimize noise on
	bias_group : int
		tes bias [0,11] to apply tickle on
	tickle_voltage : float
		voltage amplitude of tickle
	channels : list of ints
		channels to analyze bias tickle
		  
	Returns
	-------
	tick_dict : dict of {dict of {str : float}}
		Dictionary containing parameters found for each channel
	"""
	print(f'Plotting in: {S.plot_dir}')
	#Set into low current mode so that this is a small signal tickle 
	#Don't want any nonlinearity
	S.set_tes_bias_low_current(bias_group)
	data_file = S.stream_data_on()  
	#Wait 5 seconds before playing tickle
	time.sleep(5) 
	#TES bias is 2 DACs each 18 bits w/ max/min output +/-10V each.
	#multiplier is the fraction of the DAC full scale
	multiplier=tickle_voltage/20 
	#This is the DAC full scale in bits
	scale = 2**18
	#This defines the signal we'll play on the DAC
	sig   = multiplier*scale*np.cos(2*np.pi*np.array(range(2048))/(2048))
	S.play_tes_bipolar_waveform(bias_group,sig)
	#Play sine wave for 15 sec
	time.sleep(15)
	S.stop_tes_bipolar_waveform(bias_group)
	#wait 5 seconds after sine wave stops to stop stream
	time.sleep(5)
	S.stream_data_off() 
	#Read back in stream data to analyze
	timestamp,phase,mask = S.read_stream_data(datafile=data_file) 
	#ctime for plot labeling
	ctime = S.get_timestamp() 
	#Convert voltage amplitude of sine wave to the current sent down bl.
	I_command = (20*multiplier)/S.bias_line_resistance
	tick_dict = {}
	for c in channels:
		tick_dict[c] = {}
		ch_idx = mask[band,c]
		pnew = phase[ch_idx]
		#Extract only the range of the phase data where we have the tickle
		pnew = pnew[1500:4250]
		#Calculate the autocorrelation to find the tickle frequency
		lags,corrs = autocorr(pnew)
		pks,_ = signal.find_peaks(corrs,distance = 100)
		tau0 = np.median(np.diff(np.sort(pks)))/S.fs
		fguess = 1/tau0
		#Plot the autocorrelation lags + corrs
		plt.figure()
		plt.plot(lags,corrs)
		plt.plot(np.asarray(lags)[pks],corrs[pks],'ro')
		plt.xlabel('lags',fontsize = 16)
		plt.ylabel('Autocorrelation',fontsize = 16)
		plt.title(f'Band {band} Channel {c}')
		plt.savefig(f'{S.plot_dir}/{ctime}_Autocorr_b{band}_ch{c}_bg_{bias_group}.png')
		plt.close()
		#Fit the tickle to a sine wave
		time_plot = np.arange(0,len(pnew))/S.fs 
		popt,pcov = opt.curve_fit(f = sine_fit,xdata = time_plot, ydata = pnew,
							p0 = [np.mean(pnew),np.abs(np.max(pnew)-np.min(pnew))/2,
									0,fguess],
							bounds = ([np.min(pnew),0,-pi,0.95*fguess],
							[np.max(pnew),20*(np.max(pnew)-np.min(pnew)),
							pi,1.05*fguess]))
		#Use the offset of the sine wave to define the 0 phase reference
		ph_0 = popt[0]
		#Plot the phase data + fit vs time w/ the fit parameters 
		#displayed in a text box on the plot.
		fig, ax = plt.subplots()
		ax.plot(time_plot,pnew,'b--')
		ax.plot(time_plot,sine_fit(time_plot,popt[0],popt[1],popt[2],popt[3]),'r-')
		txtstr = '\n'.join((
		f'$a_0 = {popt[0]}$',
		f'$a_1 = {popt[1]}$',
		f'$\phi_1= {popt[2]}$',
		f'$f = {popt[3]}$'))
		props = dict(boxstyle = 'round',facecolor = 'wheat', alpha = 0.5)
		ax.text(0.05, 0.95, txtstr, transform=ax.transAxes, fontsize = 14,
				verticalalignment='top',bbox=props)
		plt.show()
		plt.savefig(f'{S.plot_dir}/{ctime}_Fitted_Tickle_b{band}_ch{c}_bg_{bias_group}.png')
		plt.close()
		#Convert phase nparray to current through the SQUID
		I_SQ = (S.pA_per_phi0*(pnew-ph_0)/(2*np.pi))*1e-12
		#Convert the amplitude of the sine fit to current through the SQUID
		I_amp_SQ = (S.pA_per_phi0*(popt[1])/(2*np.pi))*1e-12		
		#Calculate the voltage on the bolometer from the current division
		#between the shunt + TES branch
		V_bias = (I_command - I_amp_SQ)*S.R_sh
		#Calculate the resistance from V_bolo/I_bolo
		R_bolo = V_bias/(I_amp_SQ)
		plt.figure()
		plt.plot(time_plot,I_SQ)
		plt.xlabel('Time [s]',fontsize = 16)
		plt.ylabel('$I_{SQ/bolo}$ [A]',fontsize = 16)
		plt.title(f'Band {band} Channel {c} R = {np.round(R_bolo/1e-3,4)} mOhms')
		plt.savefig(f'{S.plot_dir}/{ctime}_R_from_Tickle_b{band}_ch{c}_bg_{bias_group}.png')
		plt.close()
	
		tick_dict[c]['I_commanded'] = I_command
		tick_dict[c]['I_amp_SQ'] = I_amp_SQ
		tick_dict[c]['V_bias'] = V_bias
		tick_dict[c]['R_bolo'] = R_bolo
	#Output a table of fitted results
	tab_rows = tuple(tick_dict.keys())
	tab_columns = tuple(['$I_{commanded}$ [$\mu$A]','$I_{bolo}$ [$\mu$A]',
				'$V_{bias}$ [nV]','$R_{bolo}$ [m$\Omega$]'])
	values = []
	for key in tab_rows:
		values.append([
			np.round(tick_dict[key]['I_commanded']/1e-6,4),
			np.round(tick_dict[key]['I_amp_SQ']/1e-6,4),
			np.round(tick_dict[key]['V_bias']/1e-9,4),
			np.round(tick_dict[key]['R_bolo']/1e-3,4)])
	ax = plt.subplot(111,frame_on=False)
	ax.xaxis.set_visible(False)
	ax.yaxis.set_visible(False)
	plt.table(cellText=values,rowLabels = tab_rows,colLabels = tab_columns, loc = 'center')
	plt.savefig(f'{S.plot_dir}/{ctime}_Summary_Table.png')
	plt.close()
	
	pkl.dump(tick_dict,open(f'{S.plot_dir}/{ctime}_summary_data.pkl','wb'))
	return tick_dict

if __name__=='__main__':    
	parser = argparse.ArgumentParser()

	# Arguments that are needed to create Pysmurf object
	parser.add_argument('--setup', action='store_true')
	parser.add_argument('--config-file', required=True)
	parser.add_argument('--epics-root', default='smurf_server_s2')

	# Custom arguments for this script
	parser.add_argument('--band', type=int, required=True, 
			help='band (must be in range [0,7])')

	parser.add_argument('--biasgroup', type = int, nargs = '+', required=True, 
		help='bias group that you want to run tickles on')

	parser.add_argument('--tickle-voltage', type=float, default = 12, 
		help='Amplitude (not peak-peak) of your tickle in volts')

	parser.add_argument('--channels', type=int, nargs = '+', required=True, 
			help='Channels that you want to calculate the tickle response of')

	# Parse command line arguments
	args = parser.parse_args()

	S = pysmurf.client.SmurfControl(
			epics_root = args.epics_root,
			cfg_file = args.config_file,
			setup = args.setup, make_logfile=False,
	)
	
	#Put your script calls here
	calc_R_tickle(S, band = args.band, bias_group=args.biasgroup, 
		tickle_voltage = args.tickle_voltage,channels=args.channels)
