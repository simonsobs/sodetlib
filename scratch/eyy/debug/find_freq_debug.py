# Debugs find frequency script and plotting
import sys
sys.path.append('../../../../')
import pysmurf
import numpy as np

S = pysmurf.SmurfControl(epics_root='mitch_epics')
S.initialize(cfg_file='/home/cryo/pysmurf/cfg_files/experiment_fp27.cfg',
	output_dir_only=True)

freq = np.loadtxt('1_amp_sweep_freq.txt')
resp = np.genfromtxt('1_amp_sweep_resp.txt', dtype=complex)

S.plot_find_freq(f=freq, resp=resp, save_plot=True)

res = S.find_all_peak(freq, resp, np.arange(14,113))