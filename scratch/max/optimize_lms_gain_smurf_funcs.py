import matplotlib
matplotlib.use('Agg')

import pysmurf.client
import argparse
import matplotlib.pyplot as plt
import time
import numpy as np
import scipy.signal as signal
import scipy.optimize as opt
from scipy import interpolate
import pickle as pkl

pi = np.pi

def lowpass_fit(x,scale,cutoff):
    fs = 4.0e3
    b,a = signal.butter(1,2*cutoff/fs)
    w,h = signal.freqz(b,a)
    x_fit = (fs*0.5/np.pi)*w
    y_fit = scale*abs(h)
    splrep = interpolate.splrep(x_fit,y_fit,s=0)
    return interpolate.splev(x,splrep,der=0)

def optimize_lms_gain(S, cfg, band, BW_target,tunefile=None,
                        reset_rate=None, frac_pp=None,
                        lms_freq=None, meas_time=None,
                        make_plot = True):
    """
    Finds the drive power and uc attenuator value that minimizes the median noise within a band.

    Parameters
    ----------
    band: (int)
        band to optimize noise on
    tunefile: (str)
        filepath to the tunefile for the band to be optimized
    reset_rate: (float)
        flux ramp reset rate in kHz used for tracking setup
    frac_pp: (float)
        fraction full scale of the FR DAC used for tracking_setup
    lms_freq: (float)
        tracking frequency used for tracking_setup
    make_plot: (bool)
        If true will make plots

    Returns
    -------

    """
    ctime = S.get_timestamp()
    band_cfg = cfg.dev.bands[band]
    if tunefile is 'devcfg':
        tunefile = cfg.dev.exp['tunefile']
        S.load_tune(tunefile)
    if tunefile is None:
        S.load_tune()
    if frac_pp is None:
        frac_pp = band_cfg['frac_pp']
    if lms_freq is None:
        lms_freq = band_cfg['lms_freq_hz']
    if reset_rate is None:
        reset_rate_khz = band_cfg['flux_ramp_rate_khz']

    S.relock(band)
    for i in range(2):
        S.run_serial_gradient_descent(band)
        S.run_serial_eta_scan(band)

    f,df,sync = S.tracking_setup(band,
                    reset_rate_khz=reset_rate_khz,
                    lms_freq_hz = lms_freq,
                    fraction_full_scale = frac_pp,
                    make_plot=True, show_plot=False,
                    channel = S.which_on(band), nsamp = 2**18,
                    feedback_start_frac = 0.02,
                    feedback_end_frac = 0.94,lms_gain=7,
                    return_data = True
                    )
    df_std = np.std(df, 0)
    f_span = np.max(f,0) - np.min(f,0)
    chans_to_consider_idx = np.where((f_span>40e3) & (f_span<120e3))
    chans_to_consider = S.which_on(band)[chans_to_consider_idx]
    datfile = S.take_stream_data(10)
    _, outdict = analyze_noise_psd(band,datfile)
    chan_noise = []
    chan_df = []
    for ch in chans_to_consider:
        chan_noise.append(outdict[ch]['white noise'])
        chan_df.append(df_std[ch])
    best_chan = chans_to_consider[np.argmin(
                np.asarray(chan_df)*np.asarray(chan_noise))]

    prev_ds_factor = S.get_downsample_factor()
    prev_filt_param = S.get_filter_disable()
    S.set_downsample_factor(1)
    S.set_filter_disable(1)

    nperseg = 2*12
    fs = S.get_flux_ramp_freq()
    detrend = 'constant'
    outdict = {}
    f3dBs = []

    lms_gain_sweep = [8,7,6,5,4,3,2]
    if make_plot = True:
        fig, (ax1,ax2) = plt.subplots(1,2)
        fig.suptitle('$f_{3dB} vs lms_gain')
        alpha = 0.8
    for lms_gain in lms_gain_sweep:
        outdict[lms_gain] = {}
        S.set_lms_gain(band,lms_gain)
        S.tracking_setup(band,
                    reset_rate_khz=reset_rate_khz,
                    lms_freq_hz = lms_freq,
                    fraction_full_scale = frac_pp,
                    make_plot=True, show_plot=False,
                    channel = best_chan, nsamp = 2**18,
                    feedback_start_frac = 0.02,
                    feedback_end_frac = 0.94,
                    lms_gain=lms_gain,
                    )
        datafile = S.take_stream_data(20)
        timestamp,phase,mask = S.read_stream_data(datafile)
        phase *= S.pA_per_phi0/(2*np.pi)
        ch_idx = mask[band,best_chan]
        f,Pxx = signal.welch(phase[ch_idx],detrend = detrend,
                            nperseg = nperseg, fs=fs)
        Pxx = np.sqrt(Pxx)
        outdict[lms_gain]['f']=f
        outdict[lms_gains]['Pxx']=Pxx
        pars,covs = opt.curve_fit(lowpass_fit,f,Pxx)
        outdict[lms_gain]['fit_params'] = pars
        f3dBs.append(pars[1])
        if make_plot = True:
            ax1.loglog(f,Pxx,alpha = alpha,label = f'lms_gain: {lms_gain}')
            ax1.loglog(f,lowpass_fit(f,pars[0],pars[1]),'--',label = 'fit lms_gain: {lms_gain}')
            alpha = alpha*0.9
        print(f'lms_gain = {lms_gain}: f_3dB fit = {pars[1]} Hz')
    if make_plot = True:
        ax1.ylim(10,100)
        ax1.legend()
        ax1.xlabel('Frequency [Hz]')
        ax1.ylabel('PSD')
        ax2.plot(lms_gain_sweep,f3dBs,'o--')
        ax2.xlabel('lms_gain')
        ax2.ylabel('f_3dB [Hz]')
        plotpath = f'{S.plot_dir}/{ctime}_f3dB_vs_lms_gain_b{band}.png'
        plt.savefig(path)
        S.pub.register_file(plotpath,'opt_lms_gain',plot=True)
    datpath = f'{S.output_dir}/{ctime}_f3dB_vs_lms_gain_b{band}.pkl'
    pkl.dump(outdict, datpath)
    S.pub.register_file(datpath,'opt_lms_gain_data',format='pkl')

    return outdict
