import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pysmurf.client
import argparse
import numpy as np
import os
import time
import glob
from scipy import signal
import scipy.optimize as opt

def analyze_iv(phase,v_bias,R_sh,pA_per_phi0,bias_line_resistance,high_current_mode,high_low_current_ratio):
# adapted from pysmurf IV analysis

    v_bias = np.abs(v_bias)

    resp = phase * pA_per_phi0/(2.*np.pi*1e6) # convert phase to uA

    step_loc = np.where(np.diff(v_bias))[0]

    if step_loc[0] != 0:
        step_loc = np.append([0], step_loc) # starts from zero
    n_step = len(step_loc) - 1

    # arrays for holding response, I, and V
    resp_bin = np.zeros(n_step)
    v_bias_bin = np.zeros(n_step)
    i_bias_bin = np.zeros(n_step)

    r_inline = bias_line_resistance
    
    if high_current_mode:

        r_inline /= high_low_current_ratio

    i_bias = 1.0E6 * v_bias / r_inline

    ### start plots

    ### end plots

    # Find steps and then calculate the TES values in bins
    for i in np.arange(n_step):
        s = step_loc[i]
        e = step_loc[i+1]

        st = e - s
        sb = int(s + np.floor(st/2))
        eb = int(e - np.floor(st/10))

        resp_bin[i] = np.mean(resp[sb:eb])
        v_bias_bin[i] = v_bias[sb]
        i_bias_bin[i] = i_bias[sb]

    d_resp = np.diff(resp_bin)
    d_resp = d_resp[::-1]
    dd_resp = np.diff(d_resp)
    v_bias_bin = v_bias_bin[::-1]
    i_bias_bin = i_bias_bin[::-1]
    resp_bin = resp_bin[::-1]

    # PROBLEMS FROM THIS FITTING SEEM TO COME FROM HOW IT FINDS
    # SC IDX AND NB IDX

    # index of the end of the superconducting branch
    dd_resp_abs = np.abs(dd_resp)
    sc_idx = np.ravel(np.where(dd_resp_abs == np.max(dd_resp_abs)))[0] + 1
    if sc_idx == 0:
        sc_idx = 1

    # index of the start of the normal branch
    nb_idx_default = int(0.8*n_step) # default to partway from beginning of IV curve
    nb_idx = nb_idx_default
    for i in np.arange(nb_idx_default, sc_idx, -1):
        # look for minimum of IV curve outside of superconducting region
        # but get the sign right by looking at the sc branch
        if d_resp[i]*np.mean(d_resp[:sc_idx]) < 0.:
            nb_idx = i+1
            break

    nb_fit_idx = int(np.mean((n_step,nb_idx)))
    norm_fit = np.polyfit(i_bias_bin[nb_fit_idx:], resp_bin[nb_fit_idx:], 1)
    if norm_fit[0] < 0:  # Check for flipped polarity
        resp_bin = -1 * resp_bin
        norm_fit = np.polyfit(i_bias_bin[nb_fit_idx:], resp_bin[nb_fit_idx:], 1)

    resp_bin -= norm_fit[1]  # now in real current units

    sc_fit = np.polyfit(i_bias_bin[:sc_idx], resp_bin[:sc_idx], 1)

    # subtract off unphysical y-offset in superconducting branch; this is
    # probably due to an undetected phase wrap at the kink between the
    # superconducting branch and the transition, so it is *probably*
    # legitimate to remove it by hand. We don't use the offset of the
    # superconducting branch for anything meaningful anyway. This will just
    # make our plots look nicer.
    resp_bin[:sc_idx] -= sc_fit[1]
    sc_fit[1] = 0 # now change s.c. fit offset to 0 for plotting

    R = R_sh * (i_bias_bin/(resp_bin) - 1)
    R_n = np.mean(R[nb_fit_idx:])
    R_L = np.mean(R[1:sc_idx]) # THIS ENDS UP NEGATIVE A LOT, WHY?

    v_tes = i_bias_bin*R_sh*R/(R+R_sh) # voltage over TES
    i_tes = v_tes/R # current through TES # THIS MIGHT JUST BE THE SAME AS RESP
    p_tes = (v_tes**2)/R # electrical power on TES

    R_trans_min = R[sc_idx]
    R_trans_max = R[nb_idx]
    R_frac_min = R_trans_min/R_n
    R_frac_max = R_trans_max/R_n

    # DO PSAT CALCULATION BOTH WAYS AND SEE IF IT'S CONSISTENT

    #method 1, from pysmurf
    p_trans_median = np.median(p_tes[sc_idx:nb_idx])
    #thought this might not be working if sc branch is a mess, takes some nonsense values into account

    #method 2, i wrote
    #finds minimum of the IV curve (i.e. the turn, when it goes into the transition)
    #then calculates the power at the minimum

    d_i_tes = np.diff(i_tes)
    turn_idxs = np.where((d_i_tes < 0.05)*(d_i_tes > -0.05))[0] #find where derivative is 0
    # if there are more than one spot, take the one at the highest bias
    if len(turn_idxs) > 1:
        real_turn_idx = turn_idxs[-1]
    elif len(turn_idxs) == 1:
        real_turn_idx = turn_idxs[0]
    else:
        real_turn_idx = None

    if real_turn_idx is not None:
        p_turn = v_tes[real_turn_idx] * i_tes[real_turn_idx]
    else: 
        p_turn = np.nan

    #remove responsivity calculation, come back to this later

    iv_dict = {}
    iv_dict['R'] = R
    iv_dict['R_n'] = R_n
    iv_dict['trans idxs'] = np.array([sc_idx,nb_idx])
    iv_dict['p_tes'] = p_tes
    iv_dict['p_trans'] = p_trans_median
    iv_dict['p_turn'] = p_turn
    iv_dict['v_bias'] = v_bias_bin
    iv_dict['v_tes'] = v_tes
    iv_dict['i_tes'] = i_tes
        
    return iv_dict

def plot_iv(iv_dict):

    R = iv_dict['R']
    R_n = iv_dict['R_n']
    sc_idx,nb_idx = iv_dict['trans idxs']
    p_tes = iv_dict['p_tes']
    p_trans_median = iv_dict['p_trans']
    p_turn = iv_dict['p_turn']
    v_bias = iv_dict['v_bias']
    v_tes = iv_dict['v_tes']
    i_tes = iv_dict['i_tes']

    plt.figure()
    plt.plot(i_tes,v_bias)
    plt.show()


if __name__=='__main__':

    iv_info = np.load('/data/smurf_data/20201014/1602712157/outputs/1602713162_iv_info.npy',allow_pickle=True).item()
    
    parser = argparse.ArgumentParser()

    # Arguments that are needed to create Pysmurf object
    parser.add_argument('--setup', action='store_true')
    parser.add_argument('--config-file', required=True)
    parser.add_argument('--epics-root', default='smurf_server_s2')

    args = parser.parse_args()

    S = pysmurf.client.SmurfControl(
            epics_root = args.epics_root,
            cfg_file = args.config_file,
            setup = args.setup,make_logfile=False
    )

    iv_raw_data = np.load(iv_info['iv_file'], allow_pickle = True).item()

    timestamp, phase, mask, tes_bias = S.read_stream_data(iv_raw_data['datafile'],return_tes_bias=True)
                

    v_bias = 2 * tes_bias[1] * S._rtm_slow_dac_bit_to_volt

    iv_dict = analyze_iv(
                phase[mask[2,23]],v_bias,iv_info['Rsh'],iv_info['pA_per_phi0'],iv_info['bias_line_resistance'],
                False,iv_info['high_low_ratio']) 

    plot_iv(iv_dict)    
