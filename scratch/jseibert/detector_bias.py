import numpy as np


def find_bias(iv_fp,band,channels):

    """
    Replaces the pysmurf run_iv function to be more appropriate for SO-specific
    usage, as well as easier to edit as needed.  Steps the TES bias down
    slowly. Starts at bias_high to bias_low with step size bias_step. Waits
    wait_time between changing steps. After overbiasing, can choose bias point
    to allow system to cool down.

    Args
    ----
    iv_fp : str, required
        filepath to the analyzed IV dict you would like to use
    band: int, required
        band that the channels you would like to analyze are on
    channels: numpy.ndarray or list, required, defaults to all channels in IV dict
        channels you would like to analyze. If None, defaults to all channels 
        in the IV dict
    
    Returns
    -------
    good_volt : list
        List of bias voltages that will place the maximal number of detectors
        in the acceptable Rfrac range. 
    """        

    iv_data = np.load(iv_fp,allow_pickle=True).item()
    
    good_bias = {}

    for ch in channels:
        min_res = 0.0026 # This is the minimum bias resistance from the det params doc
        max_res = 0.9*iv_data[band][ch]['R_n'] # Maximum bias resistance is 90% of Rfrac

        bias_idx = np.where((iv_data[band][ch]['R'] <= max_res) & (iv_data[band][ch]['R'] >= min_res))[0]

        good_bias[ch] = iv_data[band][ch]['v_bias'][bias_idx]
        
    good_chan = {}
    for i, v in enumerate(iv_data[band][channels[0]]['v_bias']):
        good_chan[v] = []

        for ch in good_bias.keys():
            if v in good_bias[ch]:
                good_chan[v].append(ch)
    
    max_len = 0
    good_volt = []

    for v in good_chan.keys():
        new_len = len(good_chan[v])
        if new_len > max_len:
            max_len = new_len

    for v in good_chan.keys():

        if len(good_chan[v]) == max_len:

            good_volt.append(v)
            
    return good_volt
 
