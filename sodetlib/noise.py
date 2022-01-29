from scipy.signal import welch
from scipy.optimize import curve_fit
from sodetlib.util import get_psd, cprint
import numpy as np
import matplotlib.pyplot as plt

def noise_model(freq, wl, n, f_knee):
    """
    Crude model for noise modeling.
    Args
    ----
    wl : float
        White-noise level.
    n : float
        Exponent of 1/f^(n/2) component.
    f_knee : float
        Frequency at which white noise = 1/f^n component
    """
    A = wl*(f_knee**n)
    return wl*np.sqrt((f_knee/freq)**n + 1)


def fit_noise_psd(f, Pxx, wl_f_range=(10,30), p0=None):
    """
    Return model fit for a PSD.
    p0 (float array): initial guesses for model fitting: [white-noise level
    in pA/rtHz, exponent of 1/f^n component, knee frequency in Hz]
    Args
    ----
    f : float array
        The frequency information.
    Pxx : float array
        The power spectral data.
    wl_f_range: float tuple
        tuple contains (f_low, f_high), f_high is used to determine the
        range of the ASD to fit to the noise model and if fit fails then the
        white noise is calculated as the median between f_low and f_high.
    p0 : float array or None, optional, default None
        Initial guess for fitting PSDs.  If None, sets to p0 =
        [100.0,0.5,0.01].
    Returns
    -------
    popt : float array
        The fit parameters - [white_noise_level, n, f_knee].
    """
    if p0 is None:
        p0 = [100.,0.5,0.01]

    bounds_low = [0.,0.,0.] # constrain 1/f^n to be red spectrum
    bounds_high = [np.inf,np.inf,np.inf]
    bounds = (bounds_low,bounds_high)

    fit_idxmax = np.argmin(np.abs(f-wl_f_range[1]))
    try:
        popt, pcov = curve_fit(noise_model, f[1:fit_idxmax], Pxx[1:fit_idxmax],
                                p0=p0, bounds=bounds)
    except Exception:
        idxmin = np.argmin(np.abs(f-wl_f_range[0]))
        wl = np.median(Pxx[idxmin:fit_idxmax])
        cprint('Unable to fit noise model. '+
                f'Reporting mean noise: {wl:.2f} pA/rtHz')

        popt = [wl, 1., 0.]
        pcov = None
    return popt

def get_noise_params(am, wl_f_range=(10,30),
                     fit=False, nperdecade=10, **psd_args):
    """
    Function to calculate the psd from an axis manager and then calculate the
    white noise, and fknee (and n-index of 1/f^{n/2} if fit=True). The fit=True
    option is a lot slower.
    Args:
        am: AxisManager
            axis manager loaded using G3tSmurf with timestamps and signal keys.
        wl_f_range: float tuple
            tuple contains (f_low, f_high), if fit=True see `fit_noise_psd`.
            The white noise is calculated as the median of the ASD (pA/rtHz)
            between f_low and f_high.
        fit: bool
            if true will fit the ASD using `fit_noise_psd` function
        nperdecade: int
            number of bins per decade (i.e. between 0.01 to 0.1 or 0.1 to 1) to
            use to average the ASD over to avoid peaks skewing the search for
            fknee when not using the fit=True option. If nperdecade = 10 for
            example then the bins between 0.01 and 1 would be:
            np.concatenate((np.linspace(0.01,0.1,10),np.linspace(0.1,1,10)[1:]))
    Returns:
        outdict: dict
            dictionary that contains all of the calculated noise parameters by
            channel, band averaged white noise levels, and f, pxx ndarrays from
            the calculated ASD the keys are:
            wls_tot: ndarray
                shape is [nchans,3] the 3 items in axis=1 are: 0 = white noise,
                1 = n (1/f index nan if fit=False), and 2 = fknee.
            band_medians: ndarray
                shape is [8,1] median white noise level for each band.
            f: ndarray
                frequency array from welch periodogram
            pxx: ndarray
                square root of welch output PSD shape is [nchans,len(f)]
    """
    nlref_10mHz = 65*np.sqrt((0.1/0.01)+1)
    f, pxx = get_psd(am.timestamps, am.signal, **psd_args)
    idx10mHz = np.argmin(np.abs(f-0.01))
    wls_tot = np.zeros((np.shape(pxx)[0],3))
    if fit == False:
        #Find white noise
        fmask = (wl_f_range[0] < f) & (f < wl_f_range[1])
        wls = np.median(pxx[:, fmask], axis=1)
        band_medians = np.zeros(8)
        for i in range(8):
            m = am.ch_info.band == i
            band_medians[i] = np.median(wls[m])
        #Calculate f_knee
        decades = np.arange(np.floor(np.log10(f[1])),2)
        binedges = np.asarray([])
        for i, dec in enumerate(decades[:-1]):
            if i == 0:
                binedges = np.concatenate((binedges,
                                            np.linspace(10**dec,
                                                        10**decades[i+1],
                                                        nperdecade)))
            else:
                binedges = np.concatenate((binedges,
                                            np.linspace(10**dec,
                                            10**decades[i+1],
                                            nperdecade)[1:]))
        start_bin = min(np.argsort(np.abs(binedges-f[1]))[0:2])
        binedges = binedges[start_bin:]
        bincenters = binedges[:-1]+np.diff(binedges)/2
        lowfn = np.zeros((len(wls),len(bincenters)))
        for ii,be in enumerate(binedges[:-1]):
            m = (f > be) & (f < binedges[ii+1])
            lowfn[:,ii] = np.median(pxx[:,m],axis = 1)
        fknees = [bincenters[np.nanargmin(np.abs(lowfn[idx]-wls[idx]*np.sqrt(2)))] for idx in range(len(wls))]
        fknees = np.asarray(fknees)
        m = [np.nanmax(lowfn[idx]) < np.sqrt(2)*wls[idx] for idx in range(len(wls))]
        fknees[m] = f[1]
        #Set wls_tot to have same structure as when we fit.
        wls_tot[:,0] = wls
        wls_tot[:,1] = np.full(len(wls),np.nan)
        wls_tot[:,2] = fknees
    else:
        bincenters = np.logspace(np.log10(f[1]),np.log10(wl_f_range[1]),30)
        lowfn = np.zeros((len(wls_tot[:,0]),len(bincenters)))
        for i, ppxx in enumerate(pxx):
            wls_tot[i,:] = fit_noise_psd(f, ppxx, wl_f_range=wl_f_range)
            lowfn[i,:] = noise_model(bincenters,*wls_tot[i,:])
        band_medians = np.zeros(8)
        for i in range(8):
            m = am.ch_info.band == i
            band_medians[i] = np.median(wls_tot[m,0])
    nl_10mHz_rat = pxx[:,idx10mHz]/nlref_10mHz
    outdict = {'wls_tot': wls_tot,
               'band_medians': band_medians,
               'f': f,
               'pxx': pxx,
               'bincenters': bincenters,
               'lowfn': lowfn,
               'low_f_10mHz': nl_10mHz_rat}
    return outdict

def plot_band_noise(am, nbins=40, noisedict=None, wl_f_range=(10,30),
                    fit=False, nperdecade=10, show_plot=True, save_plot=False,
                    save_path=None, **psd_args):
    """
    Makes a summary plot w/ subplots per band of a histogram of the white noise
    levels and another plot with histograms of the fknees. If an axis AxisManager
    is passed without a noisedict then `get_noise_params` will be called to
    generat a noisedict otherwise those parameters will be skipped.
    Args:
        am: AxisManager
            axis manager loaded using G3tSmurf with timestamps and signal keys
            to be analyzed.
        nbins: int
            number of bins in the histograms.
        noisedict: dict
            dictionary returned by `get_noise_params` see that docstring for
            details on dictionary keys.
        wl_f_range: float tuple
            tuple contains (f_low, f_high), if fit=True see `fit_noise_psd`.
            The white noise is calculated as the median of the ASD (pA/rtHz)
            between f_low and f_high.
        fit: bool
            if true will fit the ASD using `fit_noise_psd` function
        nperdecade: int
            used to calculate fknee, see `get_noise_params` doc string.
        show_plot: bool
            plot only displayed if true.
        save_plot: bool
            plot only saved if true. If true then `save_path` is required to
            save properly otherwise will just return without saving.
        save_path: str
            path where plots are saved. Required if `save_plot` is True.
    Returns:
        fig_wnl:
            matplotlib figure object for white noise plot
        axes_wnl:
            matplotlib axes object for white noise plot
        fig_fk:
            matplotlib figure object for fknee plot
        axes_fk:
            matplotlib axes object for fknee plot
    """
    bands = am.ch_info.band
    if noisedict == None:
        noisedict = get_noise_params(am, wl_f_range=wl_f_range, fit=fit,
                                      nperdecade=nperdecade, **psd_args)
    wls = noisedict['wls_tot'][:,0]
    fknees = noisedict['wls_tot'][:,2]
    band_medians = noisedict['band_medians']

    #Check if matplotlib is isinteractive mode so we can reset state at end.
    init_interactive = plt.isinteractive()
    plt.ioff()

    #Plot white noise histograms
    fig_wnl, axes_wnl = plt.subplots(4, 2, figsize=(16, 8),
                             gridspec_kw={'hspace': 0})
    fig_wnl.patch.set_facecolor('white')
    bins = np.logspace(1, 4, nbins)
    max_bins = 0

    for b in range(8):
        ax = axes_wnl[b % 4, b // 4]
        m = bands == b
        x = ax.hist(wls[m], bins=bins)
        text  = f"Median: {band_medians[b]:0.2f}\n"
        text += f"Chans pictured: {np.sum(x[0]):0.0f}"
        ax.text(0.75, .7, text, transform=ax.transAxes)
        ax.axvline(np.median(wls[m]), color='red')
        max_bins = max(np.max(x[0]), max_bins)
        ax.set(xscale='log', ylabel=f'Band {b}')

    axes_wnl[0][0].set(title="AMC 0")
    axes_wnl[0][1].set(title="AMC 1")
    axes_wnl[-1][0].set(xlabel="White Noise (pA/rt(Hz))")
    axes_wnl[-1][1].set(xlabel="White Noise (pA/rt(Hz))")
    for _ax in axes_wnl:
        for ax in _ax:
            ax.set(ylim=(0, max_bins * 1.1))
    if show_plot:
        plt.show()
    if save_plot:
        if save_path==None:
            cprint('save_path must be provided to save plots.')
        else:
            ctime = int(am.timestamps[0])
            plt.savefig(save_path+f'{ctime}_white_noise_summary.png')
        if not(show_plot):
            plt.close()

    #Plot f_knee histograms
    fig_fk, axes_fk = plt.subplots(4, 2, figsize=(16, 8),
                             gridspec_kw={'hspace': 0})
    fig_fk.patch.set_facecolor('white')
    bins = np.logspace(np.floor(np.log10(np.min(fknees[fknees>0]))),
                       np.ceil(np.log10(np.max(fknees[fknees>0]))),
                       nbins)
    max_bins = 0

    for b in range(8):
        ax = axes_fk[b % 4, b // 4]
        m = (bands == b) & (fknees > 0)
        x = ax.hist(fknees[m], bins=bins)
        text  = f"Median: {np.median(fknees[m]):0.2f}\n"
        text += f"Chans pictured: {np.sum(x[0]):0.0f}"
        ax.text(0.75, .7, text, transform=ax.transAxes)
        ax.axvline(np.median(fknees[m]), color='red')
        max_bins = max(np.max(x[0]), max_bins)
        ax.set(xscale='log', ylabel=f'Band {b}')

    axes_fk[0][0].set(title="AMC 0")
    axes_fk[0][1].set(title="AMC 1")
    axes_fk[-1][0].set(xlabel="$f_{knee}$ (Hz)")
    axes_fk[-1][1].set(xlabel="$f_{knee}$ (Hz)")
    for _ax in axes_fk:
        for ax in _ax:
            ax.set(ylim=(0, max_bins * 1.1))
    if show_plot:
        plt.show()
    if save_plot:
        if save_path != None:
            ctime = int(am.timestamps[0])
            plt.savefig(save_path+f'{ctime}_fknee_summary.png')
        if not(show_plot):
            plt.close()

    if init_interactive:
        plt.ion()
    return fig_wnl, axes_wnl, fig_fk, axes_fk

def plot_channel_noise(am, rc, save_path=None, noisedict=None, wl_f_range=(10,30),
                       fit=False, show_plot=False, save_plot=True, nperdecade=10,
                       plot1overfregion=False, **psd_args):
    """
    Function for plotting the tod and psd with white noise and fknee identified
    for a single channel.
    Args:
        am: AxisManager
            axis manager loaded using G3tSmurf with timestamps and signal keys
            to be analyzed.
        rc: int
            Readout channel (i.e. index of am.signal) to plot.
        noisedict: dict
            dictionary returned by `get_noise_params` see that docstring for
            details on dictionary keys.
        wl_f_range: float tuple
            tuple contains (f_low, f_high), if fit=True see `fit_noise_psd`.
            The white noise is calculated as the median of the ASD (pA/rtHz)
            between f_low and f_high.
        fit: bool
            if true will fit the ASD using `fit_noise_psd` function
        show_plot: bool
            plot only displayed if true.
        save_plot: bool
            plot only saved if true.
        nperdecade: int
            used to calculate fknee, see `get_noise_params` doc string.
        plot1overfregion: bool
            if true plots a line and shaded region that represents the SO
            passing low-f requirement (i.e. fknee set by wl = 65pA/rtHz and
            slope must be <= 1/f^{1/2} in the ASD)
    Returns:
        fig:
            matplotlib figure object for plot
        axes:
            matplotlib axes object for plot
    """
    if (save_plot) & (save_path==None):
        return
    if noisedict == None:
        noisedict = get_noise_params(am, wl_f_range=wl_f_range, fit=fit,
                                     nperdecade=nperdecade, **psd_args)
    f,pxx = noisedict['f'],noisedict['pxx']
    wls_tot = noisedict['wls_tot']

    #Check if matplotlib is isinteractive mode so we can reset state at end.
    init_interactive = plt.isinteractive()
    plt.ioff()
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

    plt.figure()
    fig, axes = plt.subplots(2, 1)
    ax1 = axes[0]
    ax1.plot((am.timestamps-am.timestamps[0]),am.signal[rc])
    ax1.set_xlabel('Elapsed Time [sec]',fontsize = 14)
    ax1.set_ylabel('Signal [pA]',fontsize = 14)
    band = am.ch_info.band[rc]
    chan = am.ch_info.channel[rc]
    ttlstr = f'Band {band}, Channel {chan}'
    ax1.set_title(ttlstr,fontsize = 18)
    ax2 = axes[1]
    ax2.loglog(f,pxx[rc],color = 'grey',alpha = 0.7)
    ax2.axhline(wls_tot[rc,0], xmin=0.5, xmax=1, lw=2, ls=':',
                color='k')
    wl_fknee = pxx[rc,np.argmin(np.abs(f-wls_tot[rc,2]))]
    ax2.plot(wls_tot[rc,2], wl_fknee, '*', markersize=8, color='green')
    if plot1overfregion:
        #plt.axhline(np.sqrt(2)*wls_tot[rc,0],color = 'C0')
        #plt.axvline(wls_tot[rc,2])
        plt.plot(noisedict['bincenters'],
                 noisedict['lowfn'][rc],'.',lw = 3,color = 'r')
        l1 = wls_tot[rc,0]*np.ones(len(f[f<=0.1]))
        with np.errstate(divide='ignore'):
            l2 = 65*np.sqrt(0.1/f[f<=0.1])
        ax2.plot(f[f<=0.1],l2,'--',color = 'C1')
        ax2.fill_between(f[f<=0.1],l2,l1,color = 'wheat',alpha = 0.3)
    text = f'White Noise: {np.round(float(wls_tot[rc,0]),2)} pA/rtHz\n'
    text += 'f$_{knee}$: '+f'{np.round(wls_tot[rc,2],4)} Hz'
    ax2.text(0.55, 0.7, text, bbox=props, transform=ax2.transAxes)
    ax2.set_xlabel('Frequency [Hz]', fontsize=14)
    ax2.set_ylabel('ASD [pA/rtHz]', fontsize=14)

    if show_plot:
        plt.show()
    if save_plot:
        ctime = int(am.timestamps[0])
        plt.savefig(save_path+f'{ctime}_b{band}c{chan}_noise.png')
        if not(show_plot):
            plt.close()
    if init_interactive:
        plt.ion()
    return fig, axes

def take_noise(S, cfg, acq_time=30, plot_band_summary=True, nbins=40,
               plot_channel_noise=False, show_plot=True, save_plot=False,
               rchans=None, wl_f_range=(10,30), fit=False, nperdecade=10,
               plot1overfregion=False, save_path=None, **psd_args):
    """
    Streams data for specified amount of time and then calculated the ASD and
    calculates the white noise levels and fknees for all channels. Optionally
    the band medians of the fitted parameters can be plotted and/or individual
    channel plots of the TOD and ASD with white noise and fknee called out.
    Args:
        S:
            pysmurf control object
        cfg:
            detconfig object
        acq_time: float
            acquisition time for the noise timestream.
        plot_band_summary: bool
            if true will plot band summary of white noise and  fknees.
        plot_channel_noise: bool
            if true will plot tod and psd for each channel in rchans. If rchans
            is left as None this step will be skipped.
        show_plot: bool
            if true will display plots.
        rchans: int list
            list of readout channels (i.e. index of am.signal) to make channel
            plots for.
        wl_f_range: float tuple
            tuple contains (f_low, f_high), if fit=True see `fit_noise_psd`.
            The white noise is calculated as the median of the ASD (pA/rtHz)
            between f_low and f_high.
        fit: bool
            if true will fit the ASD using `fit_noise_psd` function
        nperdecade: int
            used to calculate fknee, see `get_noise_params` doc string.
        plot1overfregion: bool
            if true plots a line and shaded region that represents the SO
            passing low-f requirement (i.e. fknee set by wl = 65pA/rtHz and
            slope must be <= 1/f^{1/2} in the ASD)
    Returns:
        am: AxisManager
            AxisManager from the timestream acquired to calculate noise
            parameters.
        outdict: dict
            dictionary that contains all calculated noise parameters and
            figure and axes objects for all plots generated. The keys are:
            'noisedict':
                dictionary returned by `get_noise_params` see that doc string
            'fig_wnl':
                matplotlib figure object for white noise band summary plot
                Only returned if plot_band_summary is True.
            'axes_wnl':
                matplotlib axes object for white noise band summary plot
                Only returned if plot_band_summary is True.
            'fig_fk':
                matplotlib figure object for fknee band summary plot
                Only returned if plot_band_summary is True.
            'axes_fk':
                matplotlib axes object for fknee band summary plot
                Only returned if plot_band_summary is True.
            'channel_plots':
                nested dictionary that has a key for each readout channe in the
                rchans list and contains a matplotlib figure and axis for each
                readout channel. Only returned if plot_channel_noise is True.
    """
    sid = sdl.take_g3_data(S, 30)
    am = sdl.load_session(cfg.stream_id, sid)
    if save_path == None:
        save_path = S.plot_dir()
    noisedict = get_noise_params(am, wl_f_range=wl_f_range, fit=fit,
                                 nperdecade=nperdecade, **psd_args)
    outdict['noisedict'] = noisedict
    if plot_band_summary:
        fig_wnl, axes_wnl, fig_fk, axes_fk = \
        plot_band_noise(am, nbins=nbins, noisedict=noisedict,
                        show_plot=show_plot, save_plot=save_plot,
                        save_path=save_path, **psd_args)
        outdict['fig_wnl'] = fig_wnl
        outdict['axes_wnl'] = axes_wnl
        outdict['fig_fk'] = fig_fk
        outdict['axes_fk'] = axes_fk

    if plot_channel_noise:
        if rchans == None:
            cprint('An rchans list must be provided to make channel plots')
            return am, outdict
        outdict['channel_plots'] = {}
        for rc in rchans:
            outdict['channel_plots'][rc] = {}
            fig, axes = plot_channel_noise(am, rc, save_path=save_path,
                                           noisedict=noisedict,
                                           show_plot=show_plot,
                                           save_plot=save_plot,
                                           plot1overfregion=plot1overfregion,
                                           **psd_args)
            outdict['channel_plots'][rc]['fig'] = fig
            outdict['channel_plots'][rc]['axes'] = axes
    return am, outdict
