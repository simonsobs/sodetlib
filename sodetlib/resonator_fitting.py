'''
Given a complex s21 sweep, returns the data fit to the resonator model and the
resonator parameters and provides tools for plotting the fit results.

Based on equation 11 from Kahlil et al. and adapted from Columbia KIDs open
source analysis code.
'''
import numpy as np
from lmfit import Model
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def linear_resonator(f, f_0, Q, Q_e_real, Q_e_imag):
    '''
    Function for a resonator with asymmetry parameterized by the imaginary
    part of ``Q_e``. The real part of ``Q_e`` is what we typically refer to as
    the coupled Q, ``Q_c``.
    '''
    Q_e = Q_e_real + 1j*Q_e_imag
    return (1 - (Q * Q_e**(-1) /(1 + 2j * Q * (f - f_0) / f_0) ) )

def cable_delay(f, delay, phi, f_min):
    '''
    Function implements a time delay (phase variation linear with frequency).
    '''
    return np.exp(1j * (-2 * np.pi * (f - f_min) * delay + phi))

def general_cable(f, delay, phi, f_min, A_mag, A_slope):
    '''
    Function implements a time delay (phase variation linear with frequency) and
    attenuation slope characterizing a background RF cable transfer function.
    '''
    phase_term =  cable_delay(f,delay,phi,f_min)
    magnitude_term = ((f-f_min)*A_slope + 1)* A_mag
    return magnitude_term*phase_term

def resonator_cable(f, f_0, Q, Q_e_real, Q_e_imag, delay, phi, f_min, A_mag,
                    A_slope):
    '''
    Function that includes asymmetric resonator (``linear_resonator``) and cable
    transfer functions (``general_cable``). Which most closely matches our full
    measured transfer function.
    '''
    resonator_term = linear_resonator(f, f_0, Q, Q_e_real, Q_e_imag)
    cable_term = general_cable(f, delay, phi, f_min, A_mag, A_slope)
    return cable_term*resonator_term

def full_fit(freqs, real, imag):
    '''
    Fitting function that takes in frequency and real and imaginary parts of the
    transmission of a resonator (needs to trimmed down to only data for a
    single resonator) and returns fitted parameters to the ``resonator_cable``
    model.

    Args
    ----
    freqs : float ndarray
        Frequencies that line up with complex transmission data.
    real : float ndarray
        Real part of resonator complex transmission to be fit
    imag : float ndarray
        Imaginary part of resonator complex transmission to be fit.

    Returns
    -------
    result : (`lmfit.Model.ModelResult`)
        This is a class of lmfit that contains all of the fitted parameters as
        well as a number of other pieces of data and metadata related to the fit
        and some helper functions for plotting and manipulating the data in the
        object.
    '''
    s21_complex = np.vectorize(complex)(real, imag)

    #set our initial guesses
    argmin_s21 = np.abs(s21_complex).argmin()
    fmin = freqs.min()
    fmax = freqs.max()
    f_0_guess = freqs[argmin_s21]
    Q_min = 0.1 * (f_0_guess / (fmax - fmin))
    delta_f = np.diff(freqs)
    min_delta_f = delta_f[delta_f > 0].min()
    Q_max = f_0_guess / min_delta_f
    Q_guess = np.sqrt(Q_min * Q_max)
    s21_min = np.abs(s21_complex[argmin_s21])
    s21_max = np.abs(s21_complex).max()
    Q_e_real_guess = Q_guess / (1 - s21_min / s21_max)
    A_slope, A_offset = np.polyfit(freqs - fmin, np.abs(s21_complex), 1)
    A_mag = A_offset
    A_mag_slope = A_slope / A_mag
    phi_slope, phi_offset = np.polyfit(freqs - fmin, np.unwrap(np.angle(s21_complex)), 1)
    delay = -phi_slope / (2 * np.pi)

    #make our model
    totalmodel = Model(resonator_cable)
    params = totalmodel.make_params(f_0=f_0_guess,
                                Q=Q_guess,
                                Q_e_real=Q_e_real_guess,
                                Q_e_imag=0,
                                delay=delay,
                                phi=phi_offset,
                                f_min=fmin,
                                A_mag=A_mag,
                                A_slope=A_mag_slope)
    #set some bounds
    params['f_0'].set(min=freqs.min(), max=freqs.max())
    params['Q'].set(min=Q_min, max=Q_max)
    params['Q_e_real'].set(min=1, max=1e7)
    params['Q_e_imag'].set(min=-1e7, max=1e7)
    params['phi'].set(min=phi_offset-np.pi, max=phi_offset+np.pi)

    #fit it
    result = totalmodel.fit(s21_complex, params, f=freqs)
    return result

def get_qi(Q, Q_e_real):
    '''
    Function for deriving the internal quality factor from the fitted quality
    factors (Q and Qc).

    Args
    ----
    Q : float
        Total resonator quality factor output parameter of ``full_fit``
    Q_e_real : float
        Resonator coupled quality factor output parameter of ``full_fit``
    Returns
    -------
    Qi : float
        Resonator internal quality factor.
    '''
    Qi = (Q**-1 - Q_e_real**-1)**-1
    return Qi

def get_br(Q, f_0):
    '''
    Function for deriving the resonator bandwidth from the fit results.

    Args
    ----
    Q : float
        Total resonator quality factor output parameter of ``full_fit``
    f_0 : float
        Resonance frequency output parameter of ``full_fit`` in Hz.
    Returns
    -------
    br : float
        Resonator bandwidth in Hz.
    '''
    br = f_0/Q
    return br

def reduced_chi_squared(ydata, ymod, n_param=9, sd=None):
    '''
    Reduced chi squared in lmfit does not return something reasonable so this
    is a handwritten function to calculate it since we want standard deviation
    to be the complex error.

    Args
    ----
    ydata : float ndarray
        complex data to calculate reduced chi squared on.
    ymod : float ndarray
        model fit result of ydata at same x-locations as ydata is sampled.
    n_param : int
        Number of parameters in the fit.
    sd : float ndarray
        standard deviation of data
    Returns
    -------
    br : float
        Resonator bandwidth in Hz.
    '''
    if sd is None:
        sdr = np.std(np.real(ydata))
        sdi = np.std(np.imag(ydata))
        sd = sdr + 1j*sdi
    chisq = np.sum((np.real(ydata) - np.real(ymod))**2/((np.real(sd))**2)) +\
            np.sum((np.imag(ydata) - np.imag(ymod))**2/((np.imag(sd))**2))
    nu=2*ydata.size-n_param     #multiply  the usual by 2 since complex
    red_chisq = chisq/nu
    return chisq, red_chisq

def fit_tune(tunefile):
    """
    Automated fitting of resonator parameters from one tuning file.

    Args
    ----
    tunefile : str, filepath
        Full directories of one tunning file.
    Returns
    -------
    dres : dict
        a dictionary containing all of the fit results for all resonances in the provided tunefile. The keys are organized as::
        
          {
            tunefile: str, filepath - full directories of one tunning file
            band: 1d int array with shape (ndets) - smurf band for each detector
            channels : 1d int array with shape (ndets) - Smurf channel for each
                       detector. If unassigned, set to -1
            res_index: 1d int array with shape (ndets) -- Index of the resonator within each band
            res_freqs: 1d int array of resonance freqs (as is found in the tunefile)
            model_params : dictionary for 9 parameters of resonator_cable 
                f_0 : 1d array with shape (ndets)
                Q : 1d array with shape (ndets)
                Q_e_real : 1d array with shape (ndets)
                Q_e_imag: 1d array with shape (ndets)
                delay: 1d array with shape (ndets)
                phi: 1d array with shape (ndets)
                f_min: 1d array with shape (ndets)
                A_mag: 1d array with shape (ndets)
                A_slope: 1d array with shape (ndets)
            derived_params : dictionary for derived parameters
                Qi: 1d array with shape (ndets)
                br: 1d array with shape (ndets)
                depth: 1d array with shape (ndets)
            find_freq_ctime : 1d string array with shape (ndets) - ctime find_freq was taken
            S21: 2d array (ndets, nsamps) of measured S21
            scan_freqs: 2d array (ndets, ndsamps) of freqs used for S21 scan
            chi2 : 2d float array of shape (ndets x 2) - chi-squared goodness of fit
            sid : str, session ID
          }
    """
    dres = {}
    data = np.load(tunefile,allow_pickle=True).item()

    dres['tunefile'] = tunefile
    model_params = {'f_0': [], 'Q': [], 'Q_e_real' : [], 'Q_e_imag': [],
                   'delay': [], 'phi': [], 'f_min': [], 'A_mag': [],
                    'A_slope': []}
    derived_params = {'Qi': [], 'br': [], 'depth': []}
    bands = []
    channels = []
    res_index = []
    chi2 = []
    sid = []
    find_freq_ctime = []
    S21s = []
    scan_freqs = []
    res_freqs = []
    
    for band in list(data.keys()):
        if 'resonances' in list(data[band].keys()):
            for idx in list(data[band]['resonances'].keys()):
                scan=data[band]['resonances'][idx]
                f=scan['freq_eta_scan']
                S21=scan['resp_eta_scan']
                result=full_fit(f,S21.real,S21.imag)

                # Need to check if this plays well with being in a dict/pickling
                S21_mod = result.best_fit.real+1j*result.best_fit.imag                
                
                bands.append(band)
                channels.append(scan['channel'])
                res_index.append(idx)
                res_freqs.append(scan['freq'])
                chi2.append(reduced_chi_squared(S21, S21_mod))
                find_freq_ctime.append(data[band]['find_freq']['timestamp'][0])
                S21s.append(S21)
                scan_freqs.append(f)

                model_params['f_0'].append(result.best_values['f_0'])
                model_params['Q'].append(result.best_values['Q'])
                model_params['Q_e_real'].append(result.best_values['Q_e_real'])
                model_params['Q_e_imag'].append(result.best_values['Q_e_imag'])
                model_params['delay'].append(result.best_values['delay'])
                model_params['phi'].append(result.best_values['phi'])
                model_params['f_min'].append(result.best_values['f_min'])
                model_params['A_mag'].append(result.best_values['A_mag'])
                model_params['A_slope'].append(result.best_values['A_slope'])
                
                Qi = get_qi(result.best_values['Q'], result.best_values['Q_e_real'])
                br = get_br(result.best_values['Q'], result.best_values['f_0'])
                depth = np.ptp(np.abs(S21_mod))
                derived_params['Qi'].append(Qi)
                derived_params['br'].append(br) 
                derived_params['depth'].append(depth)
                
    bands = np.array(bands)
    channels = np.array(channels)
    res_index = np.array(res_index)
    S21s = np.array(S21s)
    scan_freqs = np.array(scan_freqs)
    chi2 = np.array(chi2)
    sid = np.array(sid)
    for par in model_params:
        model_params[par] = np.array(model_params[par])
    for par in derived_params:
        derived_params[par] = np.array(derived_params[par])

    dres['bands'] = bands
    dres['channels'] = channels
    dres['res_index'] = res_index
    dres['res_freqs'] = res_freqs
    dres['sid'] = sid
    dres['S21'] = S21s
    dres['scan_freqs'] = scan_freqs
    dres['chi2'] = chi2
    dres['model_params'] = model_params
    dres['derived_params'] = derived_params
    return dres

def get_resfit_plot_txt(resfit_dict, band, rix):
    '''
    Function to assemble some key fit information out of the fit dictionary
    into a text block for adding to plots.

    Args
    ----
    resfit_dict : dict
        Dictionary with fit results output from ``fit_tune``
    band : int
        Smurf band of channel to get plot text for.
    rix : int
        Resonator index in tunefile (sorted by frequency order of setup_notches
        channels) of channel to get plot text for.

    Returns
    -------
    text : str
        text block to add to resonator fit channel plot.
    '''
#     Get index of band, rix pair provided by user
    idx = np.where(
        (resfit_dict['res_index'] == rix)
        & (resfit_dict['bands'] == band)
    )[0][0]
    # idx = int(np.where((resfit_dict['res_index'][:,0] == band) \
    #              & (resfit_dict['res_index'][:,1] == rix))[0])
            
    mparams = resfit_dict['model_params']
    dparams = resfit_dict['derived_params']
    channel = resfit_dict['channels'][idx]
    chi2 = resfit_dict['chi2'][idx]
    if channel == -1:
        text = f'Channel unassigned'
    else:
        text = f'Band {band} Channel {channel}'
    text += '\n$f_r$: '+ f"{np.round(mparams['f_0'][idx],1)} MHz"
    text += '\n$Q_i$: '+ f"{int(dparams['Qi'][idx])}"
    text += '\nBW: '+ f"{int(dparams['br'][idx]*1e3)} kHz"
#     text += '\nfit $\chi^2$: ' + f"{np.round(chi2,3)}"
    return text

def plot_channel_fit(fit_dict, idx):
    '''
    Function for plotting single channel eta_scan data from a tunefile with
    a fit to an asymmetric resonator model ``resonator_cable``.

    Args
    ----
    fit_dict : dict
        fit results dictionary from ``fit_tune``
    band : int
        smurf band of resonator to plot
    channel : int
        smurf channel of resonator to plot
    '''

    rix = fit_dict['res_index'][idx]
    band = fit_dict['bands'][idx]
    
    freqs = fit_dict['scan_freqs'][idx]
    res_freq = fit_dict['res_freqs'][idx]
    freqs_plot = 1e3*(freqs - res_freq)
    resp_data = fit_dict['S21'][idx]
    
    params = {}
    for p_opt in fit_dict['model_params']:
        params[p_opt] = fit_dict['model_params'][p_opt][idx]
    
    resp_model = resonator_cable(freqs,**params)
    fr_idx = np.argmin(np.abs(freqs - fit_dict['model_params']['f_0'][idx]))

    fig = plt.figure(figsize = (12,6),constrained_layout=True)
    gs = GridSpec(2, 6, figure=fig)
    ax1 = fig.add_subplot(gs[:1,0:2])
    ax1.plot(freqs_plot,20*np.log10(np.abs(resp_data)),
             'C0o',label = 'Data')
    ax1.plot(freqs_plot,20*np.log10(np.abs(resp_model)),
             'C1-',label = 'Fit')
    ax1.plot(freqs_plot[fr_idx],20*np.log10(np.abs(resp_data[fr_idx])),
             'rx',ms = 12,label = '$f_r$ - fit')
    ax1.set_xlabel('Offset $(f-f_{min})$ [kHz]')
    ax1.set_ylabel('$|S_{21}|$ [dB]')
    ax1.legend(loc = 'lower left',fontsize = 12)
    ax2 = fig.add_subplot(gs[1:,0:2])
    ax2.plot(freqs_plot,np.rad2deg(np.unwrap(np.angle(resp_data))),'C0o')
    ax2.plot(freqs_plot,np.rad2deg(np.unwrap(np.angle(resp_model))),'C1-')
    ax2.plot(freqs_plot[fr_idx],np.rad2deg(np.unwrap(np.angle(resp_data))[fr_idx]),
             'rx',ms = 12)
    ax2.set_xlabel('Offset $(f-f_r)$ [kHz]')
    ax2.set_ylabel('Phase $S_{21}$ [$^{\\circ}$]')
    ax3 = fig.add_subplot(gs[:,2:])
    ax3.plot(np.real(resp_data),np.imag(resp_data),'C0o')
    ax3.plot(np.real(resp_model),np.imag(resp_model),'C1-')
    ax3.plot(np.real(resp_model[fr_idx]),np.imag(resp_model[fr_idx]),
             'rx',ms = 12)
    ax3.set_xlabel('Re($S_{21}$) [I]')
    ax3.set_ylabel('Im($S_{21}$) [Q]')
    mrange = 1.1*np.max(np.concatenate((np.abs(np.real(resp_data)),
                                   np.abs(np.imag(resp_data)),
                                   np.abs(np.real(resp_model)),
                                   np.abs(np.imag(resp_model)))))
    ax3.set_xlim(-mrange,mrange)
    ax3.set_ylim(-mrange,mrange)
    ax3.axhline(0,color = 'k')
    ax3.axvline(0,color = 'k')

    restext = get_resfit_plot_txt(fit_dict, band, rix)
    ax3.text(0.025,0.05,restext,
             bbox=dict(facecolor='wheat',
                       alpha=0.5,
                      boxstyle="round",),
            transform=ax3.transAxes)
    return fig, np.array([ax1, ax2, ax3])

def plot_fit_summary(fit_dict, plot_style=None, quantile=0.98):
    '''
    Function for plotting full wafer eta_scan data from a tunefile.
        Plots distributions of Qi, dip depth, bandwidth, and frequencies
        for all resonators within specified quantile.

    Args
    ----
    fit_dict : dict
        fit results dictionary from ``fit_tune``
    plot_style : dict
        keyword arguments to pass to the histogram plotting for formatting.
    quantile: float
        inner X percent to graph
    '''
    
    Qis = fit_dict['derived_params']['Qi']
    depths = fit_dict['derived_params']['depth']
    bws = fit_dict['derived_params']['br']
    frs = fit_dict['model_params']['f_0']

    Qi_quant = Qis[(Qis>=np.nanquantile(Qis, 1-quantile)) \
                   & (Qis<=np.nanquantile(Qis, quantile))]
    depth_quant = depths[(depths>=np.nanquantile(depths, 1-quantile)) \
                   & (depths<=np.nanquantile(depths, quantile))]
    bw_quant = bws[(bws>=np.nanquantile(bws, 1-quantile)) \
                   & (bws<=np.nanquantile(bws, quantile))]
    fr_quant = frs[(frs>=np.nanquantile(frs, 1-quantile)) \
                   & (frs<=np.nanquantile(frs, quantile))]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 8))

    if plot_style is None:
        plot_style = {'bins': 30,
                      'color': 'gold',
                      'alpha': 0.5,
                      'edgecolor': 'orange',
                      'lw': 2}
    #Qi plot
    ax = axes[0, 0]
    ax.hist(Qi_quant/1e5, **plot_style)
    Qi_med = np.median(Qi_quant/1e5)
    ax.axvline(np.median(Qi_quant/1e5),color = 'purple',
                label = f'Median: {np.round(Qi_med,2)} $\\times10^5$')
    ax.legend(loc = 'upper right')
    ax.set_xlabel('$Q_i\\times10^5$')
    ax.set_ylabel('Counts')

    #Dip depth plot
    ax = axes[0, 1]
    ax.hist(20*np.log10(depth_quant),**plot_style)
    dep_med = np.median(20*np.log10(depth_quant))
    ax.axvline(dep_med,color = 'purple',
                label = f'Median: {np.round(dep_med,2)} dB')
    ax.legend(loc = 'upper right')
    ax.set_xlabel('Dip Depth [dB]')
    ax.set_ylabel('Counts')

    #Bandwidth plot
    plt.subplot(2,2,3)
    ax = axes[1, 0]
    ax.hist(bw_quant*1e3, **plot_style)
    bw_med = np.median(bw_quant*1e3)
    ax.axvline(bw_med,color = 'purple',
                label = f'Median: {np.round(bw_med,2)} kHz')
    ax.legend(loc = 'upper right')
    ax.set_xlabel('Bandwidth [kHZ]')
    ax.set_ylabel('Counts')

    #Frequency Separation plot
    plt.subplot(2,2,4)
    ax = axes[1, 1]
    seps = np.diff(np.sort(fr_quant))
    mean, std = np.nanmean(seps), np.std(seps)
    nstd = 2
    rng = (
        max(np.min(seps), mean - nstd * std),
        min(np.max(seps), mean + nstd * std),
    )

    ax.hist(seps, range=rng, **plot_style)
    fsep_med = np.median(np.diff(np.sort(fr_quant)))
    ax.axvline(fsep_med,color = 'purple',
                label = f'Median: {np.round(fsep_med,2)} MHz')
    ax.legend(loc = 'upper right')
    ax.set_xlabel('Resonator Separation [Mhz]')
    ax.set_ylabel('Counts')

    plt.tight_layout()
    return fig, axes
