"""
Module for smurf detector operations.
"""
from sodetlib.util import cprint, TermColors
import numpy as np
import os
import time
from scipy import signal
import scipy.optimize as opt
from scipy import interpolate
import pickle as pkl
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pysmurf.client.util.pub import set_action

def take_tickle(S,band,bias_group, high_current, tickle_voltage, over_bias):
    """
    Takes a tickle measurement on a particular bias_group at a specified amplitude.

    Parameters
    ----------
    band : int
        band to optimize noise on
    bias_group : int
        tes bias [0,11] to apply tickle on
    tickle_voltage : float
        voltage amplitude of tickle
    high_current : bool
        whether or not to take the tickle in high current mode
    over_bias : bool
        whether or not to overbias in high current mode before taking tickle

    Returns
    -------
    data_file : filepath
        Path to tickle data file.
    """
    #Setting bias in low current mode
    if high_current:
        S.set_tes_bias_high_current(bias_group)
    else:
        S.set_tes_bias_low_current(bias_group)
    if over_bias:
        cur_bias = S.get_tes_bias_bipolar(bias_group)
        S.set_tes_bias_high_current(bias_group)
        S.set_tes_bias_bipolar(bias_group,19)
        time.sleep(15)
        S.set_tes_bias_bipolar(bias_group,cur_bias)
        if not(high_current):
            S.set_tes_bias_low_current(bias_group)
        time.sleep(90)
    #Starting data stream
    data_file = S.stream_data_on()
    #Wait 5 seconds before playing tickle
    time.sleep(5)
    #This is the DAC full scale in bits
    scale = 2**18
    #Multiplier is the fraction of the DAC scale
    multiplier = tickle_voltage/10.
    #Get current DC bias level
    cur_dc = S.get_tes_bias_bipolar(bias_group)
    print(f'Current bias: {S.get_tes_bias_bipolar(bias_group)}')
    #This defines the signal we'll play on the DAC
    sig   = multiplier*scale*np.cos(2*np.pi*np.array(range(2048))/(2048)) + (cur_dc/10.)*scale
    S.play_tes_bipolar_waveform(bias_group,sig)
    #Play sine wave for 15 sec
    time.sleep(15)
    S.set_rtm_arb_waveform_enable(0x0)
    print(f'Current bias: {S.get_tes_bias_bipolar(bias_group)}')
    S.set_tes_bias_bipolar(bias_group,cur_dc)
    print(f'Current bias: {S.get_tes_bias_bipolar(bias_group)}')
    #wait 5 seconds after sine wave stops to stop stream
    time.sleep(5)
    S.stream_data_off()
    return data_file, cur_dc

@set_action()
def take_iv(S,band=None,channels=None,bias_groups=None,high_current_mode=False,wait_time=0.1,bias_high=19.9,bias_low=0.0,bias_step=0.1,overbias_volt=19.9,overbias_wait=2.0,cool_wait=30.0):
    
    if bias_groups is None:
        bias_groups = S._all_groups
    bias_groups = np.array(bias_groups)
    
    start_time = S.get_timestamp()
    
    iv_file = S.run_iv(bias_groups=bias_groups, wait_time=wait_time, bias=None,
               bias_high=bias_high, bias_low=bias_low, bias_step=bias_step,
               show_plot=False, overbias_wait=overbias_wait, cool_wait=cool_wait,
               make_plot=True, save_plot=True, plotname_append='',
               channels=None, band=band, high_current_mode=high_current_mode,
               overbias_voltage=overbias_volt, grid_on=True,
               phase_excursion_min=0.1, bias_line_resistance=None, do_analysis = False)
    
    stop_time = S.get_timestamp()
    
    iv_raw_file = np.load(iv_file,allow_pickle=True).item()
    
    iv_info = {}
    iv_info['plot_dir'] = S.plot_dir
    iv_info['output_dir'] = S.output_dir
    iv_info['Rsh'] = S.R_sh
    iv_info['bias_line_resistance'] = S.bias_line_resistance
    iv_info['high_low_ratio'] = S.high_low_current_ratio
    iv_info['pA_per_phi0'] = S.pA_per_phi0
    iv_info['high_current_mode'] = high_current_mode
    iv_info['iv_file'] = iv_file
    iv_info['start_time'] = start_time
    iv_info['stop_time'] = stop_time
    iv_info['basename'] = iv_raw_file['basename']
    iv_info['datafile'] = iv_raw_file['datafile']
    
    # load mask, to save into dict
    mask_fp = os.path.join(S.output_dir,f'{iv_info["basename"]}_mask.txt')
    mask_arr = np.loadtxt(mask_fp)
    
    iv_info['mask'] = mask_arr
    
    
    fp = os.path.join(S.output_dir,f'{iv_info["basename"]}_iv_info.npy')
    S.log(f'Writing IV information to {fp}.')
    np.save(fp,iv_info)   
    S.pub.register_file(fp, 'iv_info',format='npy')