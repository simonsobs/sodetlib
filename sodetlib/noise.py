from scipy.signal import welch
from scipy.optimize import curve_fit
from sodetlib.util import get_asd, cprint
import sodetlib as sdl
import numpy as np
import matplotlib.pyplot as plt
import os


def noise_model(freq, wl, n, f_knee):
    """
    Crude model for noise modeling

    Args
    ----
    freq : float
        independent variable in function (frequency Hz)
    wl : float
        White-noise level.
    n : float
        Exponent of 1/f^(n/2) component.
    f_knee : float
        Frequency at which white noise = 1/f^n component
    Returns
    -------
    y : float
        dependent variable in function noise in (pA/rtHz)
    """
    y = wl*np.sqrt((f_knee/freq)**n+1)
    return y

def fit_noise_asd(f, Axx, wl_f_range=(10,30), p0=None):
    """
    Return model fit for a ASD.

    Args
    -----

    f : float array
        The frequency information.
    Axx : float array
        The power spectral data.
    wl_f_range: float tuple
        tuple contains (f_low, f_high), f_high is used to determine the
        range of the ASD to fit to the noise model and if fit fails then the
        white noise is calculated as the median between f_low and f_high.
    p0 : float array or None, optional, default None
        Initial guess for fitting ASDs.  If None, sets to ``p0=[100.0,0.5,0.01]``
        which corresponds to: 

          - white-noise level in pA/rtHz
          - exponent of 1/f^n component
          - knee
          - frequency in Hz

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

    fit_idxmax = np.nanargmin(np.abs(f-wl_f_range[1]))
    try:
        popt, pcov = curve_fit(noise_model, f[1:fit_idxmax], Axx[1:fit_idxmax],
                                p0=p0, bounds=bounds)
    except Exception:
        idxmin = np.nanargmin(np.abs(f-wl_f_range[0]))
        wl = np.median(Axx[idxmin:fit_idxmax])
        popt = [wl, np.nan, np.nan]
        pcov = None
    return popt

def get_noise_params(am, wl_f_range=(10,30),
                     fit=False, nperdecade=10, **asd_args):
    """
    Function to calculate the ASD from an axis manager and then calculate the
    white noise, and fknee (and n-index of 1/f^{n/2} if fit=True). The fit=True
    option is a lot slower.

    Args
    ----
    am: AxisManager
        axis manager loaded using G3tSmurf with timestamps and signal keys.
    wl_f_range: float tuple
        tuple contains (f_low, f_high), if fit=True see `fit_noise_ASD`.
        The white noise is calculated as the median of the ASD (pA/rtHz)
        between f_low and f_high.
    fit: bool
        if true will fit the ASD using `fit_noise_ASD` function
    nperdecade: int
        number of bins per decade (i.e. between 0.01 to 0.1 or 0.1 to 1) to
        use to average the ASD over to avoid peaks skewing the search for
        fknee when not using the fit=True option. If nperdecade = 10 for
        example then the bins between 0.01 and 1 would be:
        np.concatenate((np.linspace(0.01,0.1,10),np.linspace(0.1,1,10)[1:]))
    Returns
    -------
    outdict: dict
        dictionary that contains all of the calculated noise parameters by
        channel, band averaged white noise levels, and f, axx ndarrays from
        the calculated ASD the keys are:

        noise_pars: ndarray
            shape is [nchans,3] the 3 items in axis=1 are: 0 = white noise,
            1 = n (1/f index nan if fit=False), and 2 = fknee.
        band_medians: ndarray
            shape is [8,1] median white noise level for each band.
        f: ndarray
            frequency array from welch periodogram
        axx: ndarray
            square root of welch output PSD shape is [nchans,len(f)]
        bincenters: ndarray
            Frequencies where the low frequency spectrum is binned when using
            `Fit = False`, if `Fit=True` then just the array of frequencies
            that can be used to plot the fit results in `lowfn`.
        lowfn: ndarray
            shape = [nchans, len(bincenters)]. Binned noise levels
            (if `Fit=False`) or fit noise levels (if Fit=`True`) for the low
            frequency part of the ASD.
        low_f_10mHz: ndarray
            shape = [nchans].Ratio of the ASD at 10mHz relative to a reference
            low-f component that has a white noise component = 65 pA/rtHz and
            low-f scaling of 1/f^{1/2}.
    """
    f, axx = get_asd(am, **asd_args)
    idx10mHz = np.nanargmin(np.abs(f-0.01))
    nlref_10mHz = 65*np.sqrt((0.1/f[idx10mHz])+1)
    noise_pars = np.zeros((np.shape(axx)[0],3))
    if fit == False:
        #Find white noise
        fmask = (wl_f_range[0] < f) & (f < wl_f_range[1])
        wls = np.median(axx[:, fmask], axis=1)
        band_medians = np.zeros(8)
        for i in range(8):
            m = am.ch_info.band == i
            band_medians[i] = np.median(wls[m])

        #Calculate f_knee
        decades = np.arange(np.floor(np.log10(f[1])),2)
        binedges = np.asarray([])
        for i, dec in enumerate(decades[:-1]):
            if i == 0:
                binedges = np.concatenate(
                    (binedges, np.linspace(10**dec, 10**decades[i+1],
                                           nperdecade)))
            else:
                binedges = np.concatenate(
                    (binedges, np.linspace(10**dec, 10**decades[i+1],
                                           nperdecade)[1:]))
        start_bin = min(np.argsort(np.abs(binedges-f[1]))[0:2])
        binedges = binedges[start_bin:]
        bincenters = binedges[:-1]+np.diff(binedges)/2
        lowfn = np.zeros((len(wls),len(bincenters)))
        for ii,be in enumerate(binedges[:-1]):
            m = (f > be) & (f < binedges[ii+1])
            lowfn[:,ii] = np.median(axx[:,m],axis = 1)
        fknees = [
            bincenters[np.nanargmin(np.abs(lowfn[idx]-wls[idx]*np.sqrt(2)))]
            for idx in range(len(wls))
        ]
        fknees = np.asarray(fknees)
        m = [np.nanmax(lowfn[idx]) < np.sqrt(2)*wls[idx]
             for idx in range(len(wls))]
        fknees[m] = f[1]
        #Set noise_pars to have same structure as when we fit.
        noise_pars[:, 0] = wls
        noise_pars[:, 1] = np.full(len(wls),np.nan)
        noise_pars[:, 2] = fknees
    else:
        bincenters = np.logspace(np.log10(f[1]),np.log10(wl_f_range[1]),30)
        lowfn = np.zeros((len(noise_pars[:,0]),len(bincenters)))
        for i, aaxx in enumerate(axx):
            noise_pars[i,:] = fit_noise_asd(f, aaxx, wl_f_range=wl_f_range)
            lowfn[i,:] = noise_model(bincenters,*noise_pars[i,:])
        band_medians = np.zeros(8)
        for i in range(8):
            m = am.ch_info.band == i
            band_medians[i] = np.median(noise_pars[m,0])
    nl_10mHz_rat = axx[:,idx10mHz]/nlref_10mHz
    outdict = {'noise_pars': noise_pars,
               'bands': am.ch_info.band,
               'channels': am.ch_info.channel,
               'band_medians': band_medians,
               'f': f,
               'axx': axx,
               'bincenters': bincenters,
               'lowfn': lowfn,
               'low_f_10mHz': nl_10mHz_rat}
    return outdict

def plot_band_noise(am, nbins=40, noisedict=None, wl_f_range=(10,30),
                    fit=False, nperdecade=10, show_plot=True, save_plot=False,
                    save_dir=None, **asd_args):
    """
    Makes a summary plot w/ subplots per band of a histogram of the white noise
    levels and another plot with histograms of the fknees. If an axis AxisManager
    is passed without a noisedict then `get_noise_params` will be called to
    generate a noisedict otherwise those parameters will be skipped.

    Args
    ----
    am: AxisManager
        axis manager loaded using G3tSmurf with timestamps and signal keys
        to be analyzed.
    nbins: int
        number of bins in the histograms.
    noisedict: dict
        dictionary returned by `get_noise_params` see that docstring for
        details on dictionary keys.
    wl_f_range: float tuple
        tuple contains (f_low, f_high), if fit=True see `fit_noise_asd`.
        The white noise is calculated as the median of the ASD (pA/rtHz)
        between f_low and f_high.
    fit: bool
        if true will fit the ASD using `fit_noise_asd` function
    nperdecade: int
        used to calculate fknee, see `get_noise_params` doc string.
    show_plot: bool
        plot only displayed if true.
    save_plot: bool
        plot only saved if true. If true then `save_dir` is required to
        save properly otherwise will just return without saving.
    save_dir: str
        directory where plots are saved. Required if `save_plot` is True.
    Returns
    -------
    fig_wnl: `matplotlib.figure.Figure`
        matplotlib figure object for white noise plot
    axes_wnl: `matplotlib.axes.Axes`
        matplotlib axes object for white noise plot
    fig_fk: `matplotlib.figure.Figure` 
        matplotlib figure object for fknee plot
    axes_fk: `matplotlib.axes.Axes`
        matplotlib axes object for fknee plot
    """
    if save_plot:
        if save_dir==None:
            raise ValueError('save_dir must be provided to save plots.')
    bands = am.ch_info.band
    ctime = int(am.timestamps[0])

    if noisedict == None:
        noisedict = get_noise_params(am, wl_f_range=wl_f_range, fit=fit,
                                      nperdecade=nperdecade, **asd_args)
    wls = noisedict['noise_pars'][:,0]
    fknees = noisedict['noise_pars'][:,2]
    band_medians = noisedict['band_medians']

    #Only turns off interactive mode inside with so that won't effect setting.
    try:
        isinteractive = plt.isinteractive
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
            text  = f"Median: {band_medians[b]:0.1f} pA/rtHz\n"
            text += f"Chans pictured: {np.sum(x[0]):0.0f}"
            ax.text(0.7, .7, text, transform=ax.transAxes)
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
        plt.suptitle(
            f'Total yield {len(wls)}, Overall median noise {np.nanmedian(wls):0.1f} pA/rtHz')
        if save_plot:
            plt.savefig(os.path.join(save_dir,
                                    f'{ctime}_white_noise_summary.png'))
        if show_plot:
            plt.show()
        else:
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
            ax.text(0.72, .7, text, transform=ax.transAxes)
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
            plt.savefig(os.path.join(save_dir,f'{ctime}_fknee_summary.png'))
            if not(show_plot):
                plt.close()

        #Plot ASDs
        fig_asd, axes_asd = plt.subplots(4, 2, figsize=(16, 8),
                                 gridspec_kw={'hspace': 0})
        fig_asd.patch.set_facecolor('white')

        min_x, max_x = (1, 0)
        for b in range(8):
            ax = axes_asd[b % 4, b // 4]
            m = bands == b
            med_wl = np.nanmedian(wls[m])
            f_arr = np.tile(noisedict['f'], (sum(m),1))
            x = ax.loglog(f_arr.T, noisedict['axx'][m].T, color='C0', alpha=0.1)
            ax.axhline(med_wl, color='red', alpha=0.6,
                       label=f'Med. WL: {med_wl:.1f} pA/rtHz')
            ax.set(ylabel=f'Band {b}\nASD (pA/rtHz)')
            ax.grid(linestyle='--', which='both')
            ax.legend(loc='upper right')
            min_x = min(ax.get_xlim()[0], min_x)
            max_x = max(ax.get_xlim()[1], max_x)

        axes_asd[0][0].set(title="AMC 0")
        axes_asd[0][1].set(title="AMC 1")
        axes_asd[-1][0].set(xlabel="Frequency (Hz)")
        axes_asd[-1][1].set(xlabel="Frequency (Hz)")
        for _ax in axes_asd:
            for ax in _ax:
                ax.set(xlim=[min_x, max_x], ylim=[1, 5e3])
        if save_plot:
            plt.savefig(os.path.join(save_dir,
                                    f'{ctime}_band_asds.png'))
        if show_plot:
            plt.show()
        else:
            plt.close()

    finally:
        if isinteractive:
            plt.ion()
    return fig_wnl, axes_wnl, fig_fk, axes_fk, fig_asd, axes_asd

def plot_channel_noise(am, rc, save_dir=None, noisedict=None, wl_f_range=(10,30),
                       fit=False, show_plot=False, save_plot=False, nperdecade=10,
                       plot1overfregion=False, **asd_args):
    """
    Function for plotting the tod and psd with white noise and fknee identified
    for a single channel.

    Args
    ----
    am: `sotodlib.core.AxisManager`
        axis manager loaded using G3tSmurf with timestamps and signal keys
        to be analyzed.
    rc: int
        Readout channel (i.e. index of am.signal) to plot.
    noisedict: dict
        dictionary returned by `get_noise_params` see that docstring for
        details on dictionary keys.
    wl_f_range: float tuple
        tuple contains (f_low, f_high), if fit=True see `fit_noise_asd`.
        The white noise is calculated as the median of the ASD (pA/rtHz)
        between f_low and f_high.
    fit: bool
        if true will fit the ASD using `fit_noise_asd` function
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
    Returns
    -------
    fig: `matplotlib.figure.Figure`
        matplotlib figure object for plot
    axes: `matplotlib.axes.Axes`
        matplotlib axes object for plot
    """
    if (save_plot) & (save_dir==None):
        raise ValueError('Must provide save path, exiting.')

    if noisedict == None:
        noisedict = get_noise_params(am, wl_f_range=wl_f_range, fit=fit,
                                     nperdecade=nperdecade, **asd_args)
    f,axx = noisedict['f'],noisedict['axx']
    noise_pars = noisedict['noise_pars']

    #Only turns off in with so that doesn't effect outside setting.
    try:
        isinteractive = plt.isinteractive
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
        ax2.loglog(f,axx[rc],color = 'grey',alpha = 0.7)
        ax2.axhline(noise_pars[rc,0], xmin=0.5, xmax=1, lw=2, ls=':',
                    color='k')
        wl_fknee = axx[rc,np.nanargmin(np.abs(f-noise_pars[rc,2]))]
        ax2.plot(noise_pars[rc,2], wl_fknee, '*', markersize=8, color='green')
        if plot1overfregion:
            #plt.axhline(np.sqrt(2)*noise_pars[rc,0],color = 'C0')
            #plt.axvline(noise_pars[rc,2])
            plt.plot(noisedict['bincenters'],
                     noisedict['lowfn'][rc],'.',lw = 3,color = 'r')
            l1 = noise_pars[rc,0]*np.ones(len(f[f<=0.1]))
            with np.errstate(divide='ignore'):
                l2 = 65*np.sqrt(0.1/f[f<=0.1])
            ax2.plot(f[f<=0.1],l2,'--',color = 'C1')
            ax2.fill_between(f[f<=0.1],l2,l1,color = 'wheat',alpha = 0.3)
        text = f'White Noise: {np.round(float(noise_pars[rc,0]),1)} pA/rtHz\n'
        text += 'f$_{knee}$: '+f'{np.round(noise_pars[rc,2],4)} Hz'
        ax2.text(0.03, 0.1, text, bbox=props, transform=ax2.transAxes)
        ax2.set_xlabel('Frequency [Hz]', fontsize=14)
        ax2.set_ylabel('ASD [pA/rtHz]', fontsize=14)

        if save_plot:
            ctime = int(am.timestamps[0])
            plt.savefig(os.path.join(save_dir,
                                     f'{ctime}_b{band}c{chan}_noise.png'))
        if show_plot:
            plt.show()
        else:
            plt.close()

    finally:
        if isinteractive:
            plt.ion()
    return fig, axes

@sdl.set_action()
def take_noise(S, cfg, acq_time=30, plot_band_summary=True, nbins=40,
               show_plot=True, save_plot=False, plotted_rchans=None,
               wl_f_range=(10,30), fit=False,
               nperdecade=10, plot1overfregion=False, save_dir=None,
               g3_tag=None,
               **asd_args):
    """
    Streams data for specified amount of time and then calculated the ASD and
    calculates the white noise levels and fknees for all channels. Optionally
    the band medians of the fitted parameters can be plotted and/or individual
    channel plots of the TOD and ASD with white noise and fknee called out.

    Args
    ----

    S : `pysmurf.client.base.smurf_control.SmurfControl`
        pysmurf control object
    cfg : `sodetlib.det_config.DetConfig`
        detconfig object
    acq_time : float
        acquisition time for the noise timestream.
    plot_band_summary : bool
        if true will plot band summary of white noise and  fknees.
    show_plot : bool
        if true will display plots.
    plotted_rchans : int list
        list of readout channels (i.e. index of am.signal) to make channel
        plots for.
    wl_f_range : float tuple
        tuple contains (f_low, f_high), if fit=True see `fit_noise_asd`.
        The white noise is calculated as the median of the ASD (pA/rtHz)
        between f_low and f_high.
    fit : bool
        if true will fit the ASD using `fit_noise_asd` function
    nperdecade : int
        used to calculate fknee, see `get_noise_params` doc string.
    plot1overfregion : bool
        if true plots a line and shaded region that represents the SO
        passing low-f requirement (i.e. fknee set by wl = 65pA/rtHz and
        slope must be <= 1/f^{1/2} in the ASD)
    g3_tag: string, optional
        if not None, overrides default tag "oper,noise" sent to g3 file
    
    Returns
    -------
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
                nested dictionary that has a key for each readout channel in the
                plotted_rchans list and contains a matplotlib figure and axis for
                each readout channel. Only returned if plot_channel_noise is True.
    """
    if save_dir is None:
        save_dir = S.plot_dir

    if g3_tag is None:
        g3_tag = "oper,noise"
    sid = sdl.take_g3_data(S, acq_time, tag=g3_tag)
    am = sdl.load_session(cfg.stream_id, sid, base_dir=cfg.sys['g3_dir'])
    ctime = int(am.timestamps[0])
    noisedict = get_noise_params(am, wl_f_range=wl_f_range, fit=fit,
                                 nperdecade=nperdecade, **asd_args)

    sdl.set_session_data(S, 'noise', {
        'band_medians': noisedict['band_medians']
    })

    outdict = noisedict.copy()
    outdict['sid'] = sid
    outdict['meta'] = sdl.get_metadata(S, cfg)
    if plot_band_summary:
        fig_wnl, axes_wnl, fig_fk, axes_fk, fig_asd, axes_asd = plot_band_noise(
            am, nbins=nbins, noisedict=noisedict, show_plot=show_plot,
            save_plot=False, save_dir=save_dir, **asd_args)
        if save_plot:
            savename = os.path.join(save_dir, f'{ctime}_white_noise_summary.png')
            fig_wnl.savefig(savename)
            S.pub.register_file(savename, 'take_noise', plot=True)
            savename = os.path.join(save_dir, f'{ctime}_fknee_summary.png')
            fig_fk.savefig(savename)
            S.pub.register_file(savename, 'take_noise', plot=True)
            savename = os.path.join(save_dir, f'{ctime}_asd_summary.png')
            fig_asd.savefig(savename)
            S.pub.register_file(savename, 'take_noise', plot=True)

    if plotted_rchans is not None:
        outdict['channel_plots'] = {}
        for rc in np.atleast_1d(plotted_rchans):
            outdict['channel_plots'][rc] = {}
            fig, axes = plot_channel_noise(am, rc, save_dir=save_dir,
                                           noisedict=noisedict,
                                           show_plot=show_plot,
                                           save_plot=save_plot,
                                           plot1overfregion=plot1overfregion,
                                           **asd_args)
            outdict['channel_plots'][rc]['fig'] = fig
            outdict['channel_plots'][rc]['axes'] = axes
    fname = os.path.join(S.output_dir, f'{ctime}_take_noise.npy')
    outdict['path'] = fname 
    sdl.validate_and_save(fname, outdict, S=S, cfg=cfg, make_path=False)
    return am, outdict

def plot_noise_all(res, range=(0, 200), text_loc=(0.5, 0.8)):
    pars = res['noise_pars']
    wls = pars[:, 0]
    fig, ax = plt.subplots()
    hs = ax.hist(wls, range=range, bins=40)
    wlmed = np.nanmedian(wls)
    ch_pict = int(np.sum(hs[0]))
    ch_tot = len(wls)
    ax.axvline(wlmed, color='red', ls='--')
    txt = '\n'.join([
        f'Median: {wlmed:0.2f} pA/rt(Hz)',
        f'{ch_pict}/{ch_tot} chans pictured',
        f"sid: {res['sid']}",
        f"stream_id: {res['meta']['stream_id']}",
        f"path: {os.path.basename(res['path'])}"
    ])
    ax.text(*text_loc, txt, transform=ax.transAxes,
            bbox=dict(facecolor='wheat', alpha=0.7))
    ax.set_xlabel("White Noise (pA/rt(Hz))")
    return fig, ax
