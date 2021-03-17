import pysmurf.client
import argparse
import numpy as np
import os
import pandas as pd
from resonator_model import *

def resonator_fitting(S,tunefile):
	"""
	Automated fitting of resonator parameters from tunning files
	----------
	tunefile : str array
		Full directories of tunning files. Can be one file or an array of multiple files.  		  
	Returns
	-------
	df_param : dataframe
		a pandas dataframe consisting of columns:ctime, resonator_index, f0, Qi, Qc, Q, br, depth.
	"""
	if np.array([tunefile]).size==1:
		df_param=one_resonator_fitting(tunefile)
	elif np.array([tunefile]).size>1:
		df_param=pd.DataFrame( {'time':[],'resonator_index':[], 'f0':[], 'Qi':[], 
                'Qc':[],'Q':[],'br': [],'depth':[]})
		for onefile in list(tunefile):
			f=one_resonator_fitting(onefile)
			df_param=df_param.append(f)
	return df_param


def one_resonator_fitting(tunefile):
	"""
	Automated fitting of resonator parameters from one tunning file.
	----------
	tunefile : str
		Full directories of one tunning file. 		  
	Returns
	-------
	dfres : dataframe
		a pandas dataframe consisting of columns:ctime, resonator_index, f0, Qi, Qc, Q, br, depth.
	"""    
	dres = {'time':[],'resonator_index':[], 'f0':[], 'Qi':[], 
                'Qc':[],'Q':[],'br': [],'depth':[]}
	dfres =pd.DataFrame(dres)
	data=np.load(tunefile,allow_pickle=True).item()
	for band in list(data.keys()):
	    if 'resonances' in list(data[band].keys()):
	        for idx in list(data[band]['resonances'].keys()):
	            scan=data[band]['resonances'][idx]
	            f=scan['freq_eta_scan']
	            s21=scan['resp_eta_scan']
	            result=full_fit(f,s21.real,s21.imag)

	            f0 = result.best_values['f_0']
	            Q = result.best_values['Q']
	            Qc = result.best_values['Q_e_real']
	            Qi = get_qi(result.best_values['Q'], result.best_values['Q_e_real'])
	            br = get_br(result.best_values['Q'], result.best_values['f_0'])
	            res_index = scan['channel']+band*512
	            time=data[band]['find_freq']['timestamp'][0]
	            s21_mag=np.abs(result.best_fit.real+1j*result.best_fit.imag)
	            depth = max(s21_mag)-min(s21_mag)
	            dfres = dfres.append({'time':time,'resonator_index': int(res_index), 'f0': f0, 'Qi': Qi, 
	                'Qc': Qc, 'Q': Q, 'br':br, 'depth':depth}, ignore_index=True)
	return dfres
		
if __name__=='__main__':    
	parser = argparse.ArgumentParser()

	# Arguments that are needed to create Pysmurf object
	parser.add_argument('--setup', action='store_true')
	parser.add_argument('--config-file', required=True)
	parser.add_argument('--epics-root', default='smurf_server_s2')

	# Custom arguments for this script

	parser.add_argument('--tunefile', required=True, 
		help='Tune file that you want to load for this analysis.')

	# Parse command line arguments
	args = parser.parse_args()

	S = pysmurf.client.SmurfControl(
			epics_root = args.epics_root,
			cfg_file = args.config_file,
			setup = args.setup, make_logfile=False,
	)
	
	#Put your script calls here
	resonator_fitting(S,tunefile = args.tunefile)
