import numpy as np
import os
import time
from scipy import signal
import scipy.optimize as opt
from scipy import interpolate
import pickle as pkl
import sys

try:
	import matplotlib.pyplot as plt
except Exception:
	import matplotlib
	matplotlib.use("Agg")
	import matplotlib.pyplot as plt

def plot_squid_curves(datafile):
	"""
	Plots data taken with take_open_loop_squid_curve.

	Parameters
	----------
	datafile: (filepath)

	Returns
	-------

	"""
	ctime = S.get_timestamp()
	dat = np.load(datafile,allow_pickle = True)
	dat = dat.item()
	bands = []
	for key in dat.keys():
		if key == 'bias':
			bias = dat[key]
		else:
			bands.append(int(key))
	for band in bands:
		for i, ch in enumerate(dat[band]['channels']):
			fres = dat[band]['fres'][i]
			plt.figure()
			plt.plot(bias,1e3*(dat[band]['fvsfr']-np.min(dat[band]['fvsfr'])))
			plt.xlabel('Flux Bias [Fraction Full Scale FR DAC]',fontsize = 14)
			plt.ylabel('Frequency Swing [kHz]',fontsize = 14)
			plt.title(f'Band {band} Channel {ch} $f_r$ = {np.round(fres,2)}')
			plt.savefig(f'{S.plot_dir}/{ctime}_b{band}c{ch}_dc_squid_curve.png')
			plt.close()
	return

def serial_corr(wave, lag=1):
	"""
	Code from SWH: shawn@slac.stanford.edu
	"""
	n = len(wave)
	y1 = wave[lag:]
	y2 = wave[:n-lag]
	corr = np.corrcoef(y1, y2)[0, 1]
	return corr

def autocorr(wave):
	"""
	Code from SWH: shawn@slac.stanford.edu
	"""
	lags = range(len(wave)//2)
	corrs = np.array([serial_corr(wave, lag) for lag in lags])
	return lags, corrs

def estimate_fit_parameters(phi, noisy_squid_curve,
							nharmonics_to_estimate=5, debug=False,
							min_acorr_dist_from_zero_frac=0.1):
	"""
	Code from SWH: shawn@slac.stanford.edu
	
	Estimate rf-SQUID curve fit parameters.

	Args
	----
	phi : numpy.ndarray
	   Array of fixed flux ramp bias voltages.
	noisy_squid_curve : numpy.ndarray
	   Array of rf-SQUID resonator frequencies, one for each flux ramp
	   bias voltage.
	nharmonics_to_estimate : int, optional, default 5
	   ???
	debug : bool, optional, default False
	   ???
	min_acorr_dist_from_zero_frac : float, optional, default 0.1
	   ???
	
	Returns
	-------
	est : numpy.ndarray or None
	   Estimated fit parameters.  Returns None if unable to estimate
	   the phi0 using lag and autocorrelation.
	"""

	min_acorr_dist_from_zero=len(phi)*min_acorr_dist_from_zero_frac

	if debug:
		print(f'-> min_acorr_dist_from_zero_frac={min_acorr_dist_from_zero_frac}')
		print(f'-> min_acorr_dist_from_zero={min_acorr_dist_from_zero}')
		
	##find period from autocorrelation
	lags, corrs = autocorr(noisy_squid_curve)
	
	#find peaks in autocorrelation vs lag
	from scipy.signal import find_peaks
	peaks, _ = find_peaks(corrs, height=0)
	
	if debug:
		plt.figure()
		plt.plot(lags,corrs)
		plt.plot(peaks, corrs[peaks], "x")
		plt.ylabel('Auto-correlation')
		plt.xlabel('Lag')
	
	sorted_peaks=sorted([pk for _,pk in zip(corrs[peaks],peaks)])

	if debug:
		print(f'-> sorted_peaks[:4]={sorted_peaks[:4]}')
		
	try:
		phi0_idx=next(pk for pk in sorted_peaks if pk>min_acorr_dist_from_zero)
	except:
		return None
		
	phi0=np.abs(phi[phi0_idx]-phi[0])

	if debug:
		print(f'-> phi0={phi0}')
	
	#plot cosine with same amplitude and period
	ymax = np.max(noisy_squid_curve)
	ymin = np.min(noisy_squid_curve)
	yspan = ymax-ymin
	yoffset = yspan/2. + ymin
	harmonic = lambda n,ph,phoff,amplitude : (amplitude)*np.cos(n*(ph-phoff)*(2.*np.pi/phi0))
	first_harmonic_guess = lambda ph,phoff : harmonic(1,ph,phoff,(yspan/2)) + harmonic(0,ph,phoff,(yoffset))

	#now correlate the first harmonic guess against the
	#SQUID curve
	dphi=np.abs(phi[1]-phi[0])
	testphoffs=np.linspace(0,phi0,int(np.floor(phi0/dphi)+1))
	corrs=[]
	for testphoff in testphoffs:
		y1 = first_harmonic_guess(phi,testphoff)
		y2 = noisy_squid_curve
		
		y1 = (y1-np.mean(y1))
		y2 = (y2-np.mean(y2))
		
		corr = np.corrcoef(y1, y2)[0, 1]
		corrs.append(corr)

	# should just be able to find the maximum of this correlation
	phioffset=testphoffs[np.argmax(corrs)]

	# plot harmonics only over the largest possible number of SQUID periods.  May only be 1.
	lower_phi_full_cycles=(np.min(phi)+phioffset)%(phi0)+np.min(phi)
	upper_phi_full_cycles=np.max(phi)-(np.max(phi)-phioffset)%phi0
	phi_full_cycle_idxs=np.where( (phi>lower_phi_full_cycles) & (phi<upper_phi_full_cycles) )
	phi_full_cycles=phi[phi_full_cycle_idxs]

	##overplot squid curve, squid curve + noise,
	##and as many full periods as overlap 
	#plt.figure()
	#plt.plot(phi,noisy_squid_curve)
	#plt.plot(phi,squid_curve)
	#plt.plot(phi,first_harmonic_guess(phi,phioffset),'r--')

	#plt.ylim(ymin-yspan/10.,ymax+yspan/10.)
	#plt.plot([lower_phi_full_cycles,lower_phi_full_cycles],plt.gca().get_ylim(),c='gray',ls='-',lw=2,alpha=0.5)
	#plt.plot([upper_phi_full_cycles,upper_phi_full_cycles],plt.gca().get_ylim(),c='gray',ls='-',lw=2,alpha=0.5)
	
	# correlate some harmonics and overplot!
	fit_guess=np.zeros_like(noisy_squid_curve)
	
	# add constant
	fit_guess+=np.mean(noisy_squid_curve)
	
	# mean subtract the data and this harmonic
	d=noisy_squid_curve[phi_full_cycle_idxs]
	dm=np.mean(d)
	d_ms=d-dm

	est=[phi0,phioffset,dm]
	
	for n in range(1,nharmonics_to_estimate):
		# if 1/2, peak-to-peak amplitude is 1
		A=1/2.
		this_harmonic = lambda ph : harmonic(n,ph,phioffset,A)
		
		h=this_harmonic(phi_full_cycles)
		hm=np.mean(h)
		h_ms=h-hm
		
		# sort of inverse dft them
		Xh=np.sum(d_ms*h_ms)
		
		# add this harmonic
		fit_guess += Xh*this_harmonic(phi)
		est.append(Xh)
		
		#print('n=',n,'Xh=',Xh)

	# match span of harmonic guess sum and add offset from data
	normalization_factor=(np.max(d_ms)-np.min(d_ms))/(np.max(fit_guess)-np.min(fit_guess))
	fit_guess*=normalization_factor
	# also scale parameter guesses we pass back
	est=np.array(est)
	est[3:]*=normalization_factor
	
	fit_guess+=dm

	#plt.plot(phi,fit_guess,'c--')

	return est

def model(phi, *p):
	"""
	Code from SWH: shawn@slac.stanford.edu
	"""
	#est=[phi0,phioffset,dm,A1,A2,A3,...]
	phi0=p[0]
	phioffset=p[1]
	dm=p[2]
	ret = dm

	harmonic = lambda n,ph,phoff,amplitude : (amplitude)*np.cos(n*(ph-phoff)*(2.*np.pi/phi0))

	for n in range(0,len(p[3:])):
		# if 1/2, peak-to-peak amplitude is 1
		A=1/2.
		this_harmonic = lambda ph : harmonic(n+1,ph,phioffset,A*p[3+n])
		#print(n,A*p[3+n])
		ret += this_harmonic(phi)

	return ret

