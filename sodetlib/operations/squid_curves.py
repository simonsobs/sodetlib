import numpy as np
import os
import time
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from pysmurf.client.util.pub import set_action
import sodetlib as sdl
from sodetlib.analysis import squid_fit as sqf
import matplotlib.pyplot as plt
from tqdm.auto import tqdm


def autocorr(wave):
    """
    Code to calculate the autocorrelation function of signal ``wave``.

    Args
    ----
    wave : float, ndarray
        signal to calculate autocorrelation on
    Returns
    -------
    lags : float, ndarray
        time/x-shift of signal relative to itself
    corrs : float, ndarray
        correlation coefficient at each lag
    """
    n = len(wave)
    lags = range(len(wave) // 2)
    corrs = np.zeros(len(lags))
    for ix, lag in enumerate(lags):
        y1 = wave[lag:]
        y2 = wave[: n - lag]
        corrs[ix] = np.corrcoef(y1, y2)[0, 1]
    return lags, corrs


def estimate_fit_parameters(phi, noisy_squid_curve, nharmonics_to_estimate=5,
                            min_acorr_dist_from_zero_frac=0.1):
    """
    Estimate rf-SQUID curve fit parameters to pass to ``fit_squid_curves`` as
    an initial guess for fitter.

    Args
    ----
    phi : ndarray
       Array of fixed flux ramp bias voltages.
    noisy_squid_curve : ndarray
       Array of rf-SQUID resonator frequencies, one for each flux ramp
       bias voltage.
    nharmonics_to_estimate : int, optional, default 5
       Number of harmonics of fundamental squid curve frequency to estimate.
    min_acorr_dist_from_zero_frac : float, optional, default 0.1
       Minimum distance to first peak in autocorrelation (used to estimate what
       how much fr bias current corresponds to 1 phi0) in units fraction of
       the input ``phi`` array (between 0 to 1)

    Returns
    -------
    est : ndarray or None
       Estimated fit parameters. Returns None if unable to estimate
       the phi0 using lag and autocorrelation.
    """
    min_acorr_dist_from_zero = len(phi) * min_acorr_dist_from_zero_frac

    # find period from autocorrelation
    lags, corrs = autocorr(noisy_squid_curve)

    # find peaks in autocorrelation vs lag
    peaks, _ = find_peaks(corrs, height=0)
    sorted_peaks = sorted([pk for _, pk in zip(corrs[peaks], peaks)])

    try:
        phi0_idx = next(pk for pk in sorted_peaks if pk >
                        min_acorr_dist_from_zero)
    except:
        return None

    phi0 = np.abs(phi[phi0_idx] - phi[0])

    # plot cosine with same amplitude and period
    yspan = np.ptp(noisy_squid_curve)
    yoffset = yspan / 2.0 + np.min(noisy_squid_curve)

    def harmonic(n, ph, phoff, amp): return (amp) *\
        np.cos(n*(ph-phoff)*(2*np.pi/phi0))

    def first_harmonic_guess(ph, phoff): return yoffset + \
        harmonic(1, ph, phoff, (yspan / 2))

    # now correlate the first harmonic guess against the SQUID curve
    dphi = np.abs(phi[1] - phi[0])
    testphoffs = np.linspace(0, phi0, int(np.floor(phi0 / dphi) + 1))
    corrs = []
    for testphoff in testphoffs:
        y1 = first_harmonic_guess(phi, testphoff)
        y2 = noisy_squid_curve
        y1 = y1 - np.mean(y1)
        y2 = y2 - np.mean(y2)
        corr = np.corrcoef(y1, y2)[0, 1]
        corrs.append(corr)

    # should just be able to find the maximum of this correlation
    phioffset = testphoffs[np.argmax(corrs)]

    # plot harmonics only over the largest possible number of SQUID periods.  May only be 1.
    lower_phi_full_cycles = (np.min(phi) + phioffset) % (phi0) + np.min(phi)
    upper_phi_full_cycles = np.max(phi) - (np.max(phi) - phioffset) % phi0
    phi_full_cycle_idxs = np.where(
        (phi > lower_phi_full_cycles) & (phi < upper_phi_full_cycles)
    )
    phi_full_cycles = phi[phi_full_cycle_idxs]
    # correlate some harmonics and overplot!
    fit_guess = np.zeros_like(noisy_squid_curve) + np.mean(noisy_squid_curve)
    # mean subtract the data and this harmonic
    d = noisy_squid_curve[phi_full_cycle_idxs]
    dm = np.mean(d)
    d_ms = d - dm

    est = [phi0, phioffset, dm]

    for n in range(1, nharmonics_to_estimate + 1):
        def this_harmonic(ph): return harmonic(n, ph, phioffset, 1/2)
        h = this_harmonic(phi_full_cycles)
        hm = np.mean(h)
        h_ms = h - hm
        # sort of inverse dft them
        Xh = np.sum(d_ms * h_ms)
        # add this harmonic
        fit_guess += Xh * this_harmonic(phi)
        est.append(Xh)

    # match span of harmonic guess sum and add offset from data
    normalization_factor = np.ptp(d_ms) / np.ptp(fit_guess)
    fit_guess *= normalization_factor
    # also scale parameter guesses we pass back
    est = np.array(est)
    est[3:] *= normalization_factor
    return est


def squid_curve_model(phi, *p):
    """
    Functional form of squid curve, basically a fourier expansion (with
    n-harmonics).

    Args
    ----
    phi : float, ndarray
        depedent variable in function
    p : ndarray
        indices::
            0: period in phi0s
            1: offset phase in phi0s
            2: central value (mean) of squid curve
            3 to N: amplitude of the (N-3)th harmonic of the squid curve.
    """
    phi0 = p[0]
    phioffset = p[1]
    dm = p[2]
    ret = dm

    def harmonic(n, ph, phoff, amp): return (amp) *\
        np.cos(n*(ph-phoff)*(2*np.pi/phi0))
    for n in range(0, len(p[3:])):
        def this_harmonic(ph): return harmonic(n+1, ph, phioffset, 0.5*p[3+n])
        ret += this_harmonic(phi)
    return ret


def plot_squid_fit_summary(fit_dict, nbins = 35, quantile = .95):
    '''
    Makes summary plots of squid curve fit parameters. 
    Specifically, Phi_offset, df, dfdI, hhpwr, phi0.
    
    Args
    ----
    fit_dict: dict
        Dictionary output from fit_squid_curves
    quantile: float
        Plot data that falls within the quantile given.
    '''
#     Plot offsets
    fig = plt.figure(figsize = (20,15))
    fig.set_facecolor('white')
    offsets = fit_dict['model_params'][:,1] / fit_dict['model_params'][:,0]
    offsets = offsets[(offsets>=np.nanquantile(offsets, 1-quantile)) & (offsets<=np.nanquantile(offsets, quantile))]
    plt.subplot(3,2,1)
    plt.hist(offsets,bins = nbins,alpha = 0.7)
    plt.axvline(np.median(offsets),color = 'C1',ls = ':',lw = 3)
    plt.legend([f'Median : {np.round(np.median(offsets),3)}', f'Data (Inner {quantile*100}%)\n {offsets.size} / {fit_dict["bands"].size} channels used'],
              fontsize = 14)
    plt.xlabel('$\\Phi_0$ Offset',fontsize = 16)
    plt.ylabel('Counts',fontsize = 16)
    plt.tick_params(axis='both', which='major', labelsize=12)
    print(offsets.size)

#     Plot dfs
    dfs = np.zeros(fit_dict['bands'].size)
    for idx, params in enumerate(fit_dict['derived_params']):
        dfs[idx] = params['df']
    dfs = dfs[(dfs>=np.nanquantile(dfs, 1-quantile)) & (dfs<=np.nanquantile(dfs,quantile))]
    plt.subplot(3,2,2)
    plt.hist(dfs,bins = nbins,alpha = 0.7)
    plt.axvline(np.median(dfs),color = 'C1',ls = ':',lw = 3)
    plt.legend([f'Median : {np.round(np.median(dfs),1)}', f'Data (Inner {quantile*100}%)\n {dfs.size} / {fit_dict["bands"].size} channels used'],
              fontsize = 14)
    plt.xlabel('$df_{pp}$ [kHz]',fontsize = 16)
    plt.ylabel('Counts',fontsize = 16)
    plt.tick_params(axis='both', which='major', labelsize=12)

#     Plot dfdIs
    dfdIs = np.zeros(fit_dict['bands'].size)
    for idx, params in enumerate(fit_dict['derived_params']):
        dfdIs[idx] = params['dfdI']
    dfdIs = dfdIs[(dfdIs>=np.nanquantile(dfdIs, 1-quantile)) & (dfdIs<=np.nanquantile(dfs,quantile))]    
    plt.subplot(3,2,3)
    plt.hist(dfdIs/1e-3,bins = nbins,alpha = 0.7)
    plt.axvline(np.median(dfdIs)/1e-3,color = 'C1',ls = ':',lw = 3)
    plt.legend([f'Median : {np.round(np.median(dfdIs)/1e-3,2)}', f'Data (Inner {quantile*100}%)\n {dfdIs.size} / {fit_dict["bands"].size} channels used'],
              fontsize = 14)
    plt.xlabel('<df/dI> [mHz/pA]',fontsize = 16)
    plt.ylabel('Counts',fontsize = 16)
    plt.tick_params(axis='both', which='major', labelsize=12)

#     Plot hhpwrs
    hhpwrs = np.zeros(fit_dict['bands'].size)
    for idx, params in enumerate(fit_dict['derived_params']):
        hhpwrs[idx] = params['hhpwr']
    hhpwrs = hhpwrs[(hhpwrs>=np.nanquantile(hhpwrs, 1-quantile)) & (hhpwrs<=np.nanquantile(hhpwrs,quantile))]
    plt.subplot(3,2,4)
    plt.hist(hhpwrs*100,bins = nbins,alpha = 0.7)
    plt.axvline(np.median(hhpwrs)*100,color = 'C1',ls = ':',lw = 3)
    plt.legend([f'Median : {np.round(np.median(hhpwrs)*100,2)}', f'Data (Inner {quantile*100}%)\n {hhpwrs.size} / {fit_dict["bands"].size} channels used'],
              fontsize = 14)
    plt.xlabel('Higher Harmonic Power [%]',fontsize = 16)
    plt.ylabel('Counts',fontsize = 16)
    plt.tick_params(axis='both', which='major', labelsize=12)
    
#     Plot phio0s
    phi0s = fit_dict['model_params'][:,0]
    phi0s = phi0s[(phi0s>=np.nanquantile(phi0s, 1-quantile))& (phi0s<=np.nanquantile(phi0s,quantile))]
    plt.subplot(3,2,5)
    plt.hist(phi0s,bins = nbins,alpha = 0.7)
    plt.axvline(np.median(phi0s),color = 'C1',ls = ':',lw = 3)
    plt.legend([f'Median : {np.round(np.median(phi0s),3)}', f'Data (Inner {quantile*100}%)\n {phi0s.size} / {fit_dict["bands"].size} channels used'],
              fontsize = 14)
    plt.xlabel('$\\Phi_0$ [ff]',fontsize = 16)
    plt.ylabel('Counts',fontsize = 16)
    plt.tight_layout()
    plt.tick_params(axis='both', which='major', labelsize=12)
    return


def plot_squid_fit(data, fit_dict, band, channel, save_plot=False, S=None,
                   plot_dir=None):
    """
    Plots data taken with ``take_squid_curve`` against fits from
    ``fit_squid_curve``.

    Args
    ----
    data: dict
        Output data product from take_squid_curves
    fit_dict: dict
        Dictionary of fits. Product of fit_squid_curves
    band: int
        Band number
    channel: int
        Channel number
    save_plot: bool
        Boolean to toggle saving of plot to disk
    S: PysmurfControl object, optional
        Used to grab plot_dir and publish plots
    plot_dir: str
        Overrides plot_dir from S, if specified
    """
    if save_plot:
        if plot_dir is not None:
            pass
        elif S is not None:
            plot_dir = S.plot_dir
        else:
            raise ValueError("Either S or ``plot_dir`` must be specified.")
    idx = np.where((data['bands']==band) & (data['channels']==channel))[0][0]
    biases = data['fluxramp_ffs']

    fres = data['res_freq'][idx]
    fig = plt.figure()
    fig.set_facecolor('white')
    plt.plot(biases, data["res_freq_vs_fr"][idx, :], 'co')
    plt.plot(biases, squid_curve_model(biases,
                                       *fit_dict['model_params'][idx, :]),
             'C1--')
    ax = plt.gca()
    plt.text(0.0175, 0.975, fit_dict['plt_txt'][idx], horizontalalignment='left',
             verticalalignment='top', transform=ax.transAxes, fontsize=10,
             bbox=dict(facecolor='wheat', alpha=0.5, boxstyle='round'))
    plt.xlabel("Flux Bias [Fraction Full Scale FR DAC]", fontsize=14)
    plt.ylabel("Frequency Swing [MHz]", fontsize=14)
    plt.title(f"Band {band} Channel {channel} $f_r$ = {np.round(fres,2)} [MHz]")
    if save_plot:
        ctime = int(time.time())
        fig_name = f"{plot_dir}/{ctime}_b{band}c{channel}_dc_squid_curve.png"
        plt.savefig(fig_name)
        if S is not None:
            S.pub.register_file(fig_name, 'dc_squid_curve', plot=True)
    return

def fit_squid_curves(squid_data, fit_args=None):
    '''
    Function for fitting squid curves taken with ``take_squid_curve``.
    
    Args
    ----
    squid_data: dictionary
        Output data product from take_squid_curves
    fit_args: int
        Number of harmonics to fit to squid curves, defaults to 5.
    
    Returns
    -------
    fit_out: dictionary
        Dictionary of fit parameters.
    '''
    fit_out = {'biases': squid_data['fluxramp_ffs'],
               'bands': squid_data['bands'],
               'channels': squid_data['channels']}

    if fit_args is None:
        nharm = 5
    else:
        if 'nharmonics_to_estimate' in fit_args.keys():
            nharm = fit_args['nharmonics_to_estimate']
        else:
            nharm = 5

    nchans = len(squid_data['channels'])
    fit_guess = np.zeros((nchans, nharm+3))
    fit_result = np.zeros((nchans, nharm+3))
    plt_txt = []
    derived_params = []
    for nc in range(nchans):
        if fit_args:
            fit_guess[nc, :] = estimate_fit_parameters(squid_data['fluxramp_ffs'],
                                                       squid_data['res_freq_vs_fr'][nc],
                                                       **fit_args)
        else:
            fit_guess[nc, :] = estimate_fit_parameters(squid_data['fluxramp_ffs'],
                                                       squid_data['res_freq_vs_fr'][nc])
        fit_result[nc, :], _ = curve_fit(squid_curve_model, squid_data['fluxramp_ffs'],
                                         squid_data['res_freq_vs_fr'][nc, :], p0=fit_guess[nc, :])
        txt, params = get_derived_params_and_text(squid_data, fit_result[nc, :], nc)
        params['1h'] = fit_guess[nc, 3]
        params['2h'] = fit_guess[nc, 4]
        params['3h'] = fit_guess[nc, 5]
        params['4h'] = fit_guess[nc, 6]
        params['5h'] = fit_guess[nc, 7]
        derived_params.append(params)
        plt_txt.append(txt)
        
    fit_out['initial_guess'] = fit_guess
    fit_out['model_params'] = fit_result
    fit_out['derived_params'] = derived_params
    fit_out['plt_txt'] = plt_txt
    return fit_out


def dfduPhi0_to_dfdI(dfduphi0, M_in=227e-12):
    '''
    Function to convert averaged squid slope (average squid gain) from units of
    Hz/micro-Phi0 to H per pA given the mutual inductance between the squid and
    TES circuit.

    Args
    ----
    dfduphi0 : float
        SQUID gain in Hz per micro-Phi0
    M_in : float
        Mutual inductance between SQUID and TES in units of Henries.
    Returns
    -------
    dfdI_Hz_per_pA : float
        SQUID gain in Hz per pA
    '''
    M_in_phi0_per_A = M_in/2.067833848e-15  # Phi0/A = [Wb/A]/[Wb/Phi0]
    dfdphi0 = dfduphi0*1e6  # Hz/phi0 = [Hz/uphi0]*[uphi0/phi0]
    dfdI_Hz_per_A = dfdphi0*M_in_phi0_per_A  # Hz/A = [HZ/Phi0]*[Phi0/A]
    dfdI_Hz_per_pA = dfdI_Hz_per_A*1e-12  # Hz/pA = [Hz/A]*[A/pA]
    return dfdI_Hz_per_pA


def get_derived_params_and_text(data, model_params, idx):
    '''
    Function to calculate some useful parameters derived from the primary squid
    curve fit parameters and return a text block to add to channel plots.

    Args
    ----
    data : ndarray
        Output data dictionary from ``take_squid_curve``.
    model_params : dict
        Model parameters fitted to ``squid_curve_model``.
    idx : int
        Index of output data array for channel to get text for.
    Returns
    -------
    fitresulttxt : str
        String to add to squid fit channel plot.
    fit_dict : dict
        Dictionary of useful parameters derived from the model fit.
    '''
    fit_curve = sqf.model(data['fluxramp_ffs'], *model_params)
    df_khz = np.ptp(fit_curve)*1000
    phi_over_one_cycle = np.linspace(0, model_params[0], 10000)+model_params[1]
    fit_curve_over_one_cycle = squid_curve_model(
        phi_over_one_cycle, *model_params)
    phi_over_one_cycle /= model_params[0]
    avg_dfdphi_Hzperuphi0 = np.mean(np.abs(np.gradient(fit_curve_over_one_cycle)) /
                                    (np.gradient(phi_over_one_cycle)))
    hhpwr = (np.square(model_params[3]) /
             (np.sum(np.square(model_params[4:]))))**-1
    dfdI = dfduPhi0_to_dfdI(avg_dfdphi_Hzperuphi0, 227e-12)
    plot_txt = ('$f_{res}$' + f' = {data["res_freq"][idx]:.1f} MHz\n' +
                '$\\Phi_0$' + f' = {model_params[0]:.3f} ff\n' +
                '$\\Phi_{offset}$' +
                f' = {model_params[1]/model_params[0]:.3f} Phi0\n' +
                f'df = {df_khz:.1f} kHz\n' +
                '$<df/dI>$' + f' = {dfdI/1e-3:.1f} mHz/pA\n' +
                f'hhpwr = {hhpwr:.3f}')
    derived_params = {'df': df_khz,
                      'dfdI': dfdI,
                      'hhpwr': hhpwr}
    return plot_txt, derived_params

@set_action()
def take_squid_curve(S, cfg, wait_time=0.1, Npts=4, Nsteps=500,
                     bands=None, channels=None, lms_gain=None, out_path=None,
                     run_analysis=True, analysis_kwargs=None, show_pb=False,
                     run_serial_ops=True, frac_full_scale_max=0.3):
    """
    Takes data in open loop (only slow integral tracking) and steps through flux
    values to trace out a SQUID curve. This can be compared against the tracked
    SQUID curve which might not perfectly replicate this if these curves are
    poorly approximated by a sine wave (or ~3 harmonics of a fourier expansion).

    Args
    ----
    S : ``pysmurf.client.base.smurf_control.SmurfControl``
        ``pysmurf`` control object
    cfg : ``sodetlib.det_config.DeviceConfig``
        device config object.
    wait_time: float
        how long you wait between flux step point in seconds
    Npts : int
        number of points you take at each flux bias step to average
    Nsteps : int
        Number of flux points you will take total.
    bands : int, list
        list of bands to take dc SQUID curves on
    channels : dict
        default is None and will run on all channels that are on
        otherwise pass a dictionary with a key for each band
        with values equal to the list of channels to run in each band.
    lms_gain : int
        gain used in tracking loop filter and set in ``tracking_setup``
        defaults to ``None`` and pulls from ``det_config``
    out_path : str, filepath
        directory to output npy file to. defaults to ``None`` and uses pysmurf
        plot directory (``S.plot_dir``)
    run_serial_ops : bool
        If true, will run serial grad descent and eta scan
    frac_full_scale_max : float
        Max value of fraction full scale to use.
    Returns
    -------
    data : dict
        This contains the flux bias array, channel array, and frequency
        shift at each bias value for each channel in each band.
    """
    cur_mode = S.get_cryo_card_ac_dc_mode()
    if cur_mode == 'AC':
        S.set_mode_dc()
    ctime = S.get_timestamp()
    if out_path is None:
        out_path = os.path.join(S.output_dir, f'{ctime}_fr_sweep_data.npy')

    # This calculates the amount of flux ramp amplitude you need for 1 phi0
    # and then sets the range of flux bias to be enough to achieve the Nphi0s
    # specified in the fucnction call.
    if bands is None:
        bands = np.arange(8)
    if channels is None:
        channels = {}
        for band in bands:
            channels[band] = S.which_on(band)

    bias_peak = frac_full_scale_max

    # This is the step size calculated from range and number of steps
    bias_step = np.abs(2*bias_peak)/float(Nsteps)
    if bands is None:
        bands = np.arange(8)
    if channels is None:
        channels = {}
        for band in bands:
            channels[band] = S.which_on(band)

    channels_out = []
    bands_out = []
    for band in bands:
        channels_out.extend(channels[band])
        bands_out.extend(list(np.ones(len(channels[band]))*band))
    biases = np.arange(-bias_peak, bias_peak, bias_step)

    # final output data dictionary
    data = {}
    data['meta'] = sdl.get_metadata(S, cfg)
    data['bands'] = np.asarray(bands_out)
    data['channels'] = np.asarray(channels_out)
    data['fluxramp_ffs'] = biases
    data['res_freq_vs_fr'] = []

    unique_bands = np.unique(np.asarray(bands_out, dtype=int))
    prev_lms_enable1 = {}
    prev_lms_enable2 = {}
    prev_lms_enable3 = {}
    prev_lms_gain = {}
    for band in unique_bands:
        band_cfg = cfg.dev.bands[band]
        if lms_gain is None:
            lms_gain = band_cfg['lms_gain']
        S.log(f'{len(channels[band])} channels on in band {band},'
              ' configuring band for simple, integral tracking')
        S.log(f'-> Setting lmsEnable[1-3] and lmsGain to 0 for band {band}.')
        prev_lms_enable1[band] = S.get_lms_enable1(band)
        prev_lms_enable2[band] = S.get_lms_enable2(band)
        prev_lms_enable3[band] = S.get_lms_enable3(band)
        prev_lms_gain[band] = S.get_lms_gain(band)
        S.set_lms_enable1(band, 0)
        S.set_lms_enable2(band, 0)
        S.set_lms_enable3(band, 0)
        S.set_lms_gain(band, lms_gain)

    fs = {}
    S.log(
        '\rSetting flux ramp bias to 0 V\033[K before tune'.format(-bias_peak))
    S.set_fixed_flux_ramp_bias(0.)

    for band in unique_bands:
        fs[band] = []
        if run_serial_ops:
            S.run_serial_gradient_descent(band)
            S.run_serial_eta_scan(band)
        S.toggle_feedback(band)

    small_steps_to_starting_bias = np.arange(-bias_peak, 0, bias_step)[::-1]

    # step from zero (where we tuned) down to starting bias
    S.log('Slowly shift flux ramp voltage to place where we begin.')
    for b in small_steps_to_starting_bias:
        S.set_fixed_flux_ramp_bias(b, do_config=False)
        time.sleep(wait_time)

    # make sure we start at bias_low
    S.log(f'\rSetting flux ramp bias low at {-bias_peak} V')
    S.set_fixed_flux_ramp_bias(-bias_peak, do_config=False)
    time.sleep(wait_time)

    S.log('Starting to take flux ramp.')

    for b in tqdm(biases, disable=(not show_pb)):
        S.set_fixed_flux_ramp_bias(b, do_config=False)
        time.sleep(wait_time)
        for band in unique_bands:
            fsamp = np.zeros(shape=(Npts, len(channels[band])))
            for i in range(Npts):
                fsamp[i, :] = S.get_loop_filter_output_array(band)[
                    channels[band]]
            fsampmean = np.mean(fsamp, axis=0)
            fs[band].append(fsampmean)

    S.log('Done taking flux ramp data.')
    fres = []

    for i, band in enumerate(unique_bands):
        fres_loop = S.channel_to_freq(band).tolist()
        fres.extend(fres_loop)
        # stack
        lfovsfr = np.dstack(fs[band])[0]
        fvsfr = np.array([arr+fres for (arr, fres) in zip(lfovsfr, fres_loop)])
        if i == 0:
            data['res_freq_vs_fr'] = fvsfr
        else:
            data['res_freq_vs_fr'] = np.concatenate(
                (data['res_freq_vs_fr'], fvsfr), axis=0)

    data['res_freq'] = np.asarray(fres)
    data['filepath'] = out_path

    # save dataset for each iteration, just to make sure it gets
    # written to disk
    np.save(out_path, data)
    S.pub.register_file(out_path, 'dc_squid_curve', format='npy')

    # done - zero and unset
    S.set_fixed_flux_ramp_bias(0, do_config=False)
    S.unset_fixed_flux_ramp_bias()
    for band in unique_bands:
        S.set_lms_enable1(band, prev_lms_enable1[band])
        S.set_lms_enable2(band, prev_lms_enable2[band])
        S.set_lms_enable3(band, prev_lms_enable3[band])
        S.set_lms_gain(band, lms_gain)

    if cur_mode == 'AC':
        S.set_mode_ac()

    if run_analysis:
        if analysis_kwargs is not None:
            fit_dict = fit_squid_curves(data, **analysis_kwargs)
        else:
            fit_dict = fit_squid_curves(data)
        return data, fit_dict
    else:
        return data
