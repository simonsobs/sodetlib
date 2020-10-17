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

def analyze_tickle(S, band, data_file, dc_level, tickle_voltage, high_current, channels = None,
                    make_channel_plots = False,R_threshold=100):
    """
    Analyzes tickle measurement and writes a pickle file and returns a dictionary
    with the channels that show a tickle response and the resistance of those
    channels.

    Parameters
    ----------
    band : int
        band to optimize noise on
    data_file : filepath
        path to data to be analyzed
    dc_level : float
        DC voltage bias at which tickle taken at.
    tickle_voltage : float
        voltage amplitude of tickle
    channels : int list
        If None will analyze all channels that are on in the specified band.
    make_channel_plots : bool
        Whether or not to show the individual channel plots with fits, defaults
        to false.
    R_threshold : float
        Resistance in mOhms of what to consider a detector.

    Returns
    -------
    tick_dict : dictionary
        Dictionary with T/F for each channel whether its a detector or not and
        for detector channels calculated resistance.
    """
    #Read back in stream data to analyze
    timestamp,phase,mask = S.read_stream_data(datafile=data_file)
    #ctime for plot labeling
    ctime = S.get_timestamp()
    #Convert voltage amplitude of sine wave to the current sent down bl.
    I_command = (tickle_voltage)/S.bias_line_resistance
    if high_current:
        I_command = I_command*S.high_low_current_ratio
    print(f'I commanded = {I_command}')
    if channels == None:
        channels = S.which_on(band)

    #Initialize some output variables
    tick_dict = {}
    det_freqs = []
    det_Rs = []

    for c in channels:

        ch_idx = mask[band,c]
        pnew = phase[ch_idx]
        #Extract only the range of the phase data where we have the tickle
        pnew = pnew[1500:4250]
        p_mean = np.mean(pnew)

        I_amp_check = (S.pA_per_phi0*(pnew-p_mean)/(2*np.pi))*1e-12
        amp_check = (np.max(I_amp_check)-np.min(I_amp_check))/2
        V_bias = (I_command - amp_check)*S.R_sh
        R_pp = V_bias / amp_check
        time_plot = np.arange(0,len(pnew))/S.fs
        if 1.1*amp_check > I_command:
            #This checks if you are not tracking properly and you're getting
            #more phase shift than you would get in the superconducting state
            #when all commanded current goes through the TES
            channel_data = {
                'detector_channel': False,
                'failure_statement': 'Not Tracking',
            }
        elif amp_check < I_command*(S.R_sh/(S.R_sh + R_threshold*1e-3)):
            #This checks if you see no response (i.e. just noise), it rejects
            #all responses whose peak to peak shift of its timestream is less
            #than what would be present w/ a 500 mOhm bolo which is 50 times
            #our bolometer designed normal reistance.
            V_bias = (I_command - amp_check)*S.R_sh
            channel_data = {
                'detector_channel': False,
                'failure_statement': 'No Response',
            }
        else:
            #Calculate the autocorrelation to find the tickle frequency
            lags,corrs = autocorr(pnew)
            pks,_ = signal.find_peaks(corrs,distance = 100)
            tau0 = np.median(np.diff(np.sort(pks)))/S.fs
            fguess = 1/tau0
            #Fit the tickle to a sine wave
            popt,pcov = opt.curve_fit(f = sine_fit,xdata = time_plot, ydata = pnew,
                                p0 = [np.mean(pnew),np.abs(np.max(pnew)-np.min(pnew))/2,
                                        0,fguess],
                                bounds = ([np.min(pnew),0,-pi,0.95*fguess],
                                [np.max(pnew),20*(np.max(pnew)-np.min(pnew)),
                                pi,1.05*fguess]))

            channel_data = {
                'detector_channel': True, 
                'failure_statement': None,
                'popts': popt,
                'pcov': pcov
            }
        channel_data['R_pp'] = R_pp
        channel_data['amplitude'] = amp_check
        channel_data['res_freq'] = S.channel_to_freq(band, c)
        tick_dict[c] = channel_data

        if make_channel_plots:
            fig, ax = plt.subplots()
            ax.plot(time_plot, pnew)
            text_lines = []
            if channel_data['detector_channel']:
                ax.plot(time_plot, sine_fit(time_plot, *popt))
                popt_str = ', '.join([f"{p:.2f}" for p in popt])
                text_lines = ["Detector Found", 
                              f"$(a_0, a_1, \phi_1, f)=({popt_str})$"]
            else:
                text_lines = [f"No Detector: {channel_data['failure_statement']}"]
            text_lines.append(f"R_pp = {channel_data['R_pp']}")
            props = {'boxstyle': 'round', 'facecolor': 'wheat', 'alpha': .8}
            ax.text(0.05, 0.95, '\n'.join(text_lines), transform=ax.transAxes, fontsize=14,
                    verticalalignment='top', bbox=props)
            ax.set_xlabel('Time [x]', fontsize=16)
            ax.set_ylabel(r'$\phi_0$', fontsize=16)
            ax.set_title(f'Band {band}, Channel {c}, Freq {channel_data["res_freq"]}')
            fig.savefig(f'{S.plot_dir}/{ctime}_tickle_response_b{band}c{c}.png')
            plt.close(fig)
        
            
    dets = [c for c in channels if tick_dict[c]['detector_channel']]
    nondets = [c for c in channels if not tick_dict[c]['detector_channel']]


    # Summary plots
    # Chan vs current scatter plot
    fig, ax = plt.subplots()
    xs = [tick_dict[c]['amplitude'] / I_command for c in dets]
    ax.scatter(xs, dets, label='Detector')

    ax.axvline(1, alpha=0.6, color='grey')  # For superconductor, I=I_command
    ax.axvline(S.R_sh / (10e-3 + S.R_sh), alpha=0.6, color='grey')  # For 10 mOhm resistor

    xs = [tick_dict[c]['amplitude'] / I_command for c in nondets]
    ax.scatter(xs, nondets, label='No Detector')

    ax.legend()
    ax.set(xlabel=r"$I / I_{command}$", ylabel="channel number")
    ax.set(xscale='log')
    fig.savefig(f'{S.plot_dir}/{ctime}_identified_detectors_b{band}.png')
    plt.close(fig)

    # Detector Resistances scatter
    fig, ax = plt.subplots()
    xs = [tick_dict[c]['res_freq'] for c in dets]
    ys = [tick_dict[c]['R_pp']/1e-3 for c in dets]
    ax.scatter(xs, ys)
    ax.set(xlabel="Detector Frequency [MHz]", ylabel="Resistance from tickle")
    ax.set(title=f'Bolo resistance at DC Bias = {np.round(dc_level,2)} V')
    fig.savefig(f'{S.plot_dir}/{ctime}_tickle_resistance_b{band}.png')
    plt.close(fig)
    
    pkl.dump(tick_dict,open(f'{S.output_dir}/{ctime}_summary_data.pkl','wb'))
    return tick_dict
