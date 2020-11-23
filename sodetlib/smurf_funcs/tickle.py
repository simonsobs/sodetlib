import numpy as np
import os
import time
from scipy import signal
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scipy.optimize as opt
import pickle as pkl
pi = np.pi

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
    high_current : bool
        whether or not to take the tickle in high current mode
    tickle_voltage : float
        voltage amplitude of tickle
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


