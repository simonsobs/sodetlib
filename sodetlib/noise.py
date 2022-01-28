from scipy.signal import welch, freqz
from scipy.optimize import curve_fit
import numpy as np

def fit_noise_psd(f, Pxx, stream_id, session_id, fs=None, p0=None,
                flux_ramp_freq=None, filter_a=None, filter_b=None):
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
    fs : float or None, optional, default None
        Sampling frequency. If None, loads in the current sampling
        frequency.
    p0 : float array or None, optional, default None
        Initial guess for fitting PSDs.  If None, sets to p0 =
        [100.0,0.5,0.01].
    flux_ramp_freq : float, optional, default None
        The flux ramp frequency in Hz.
    filter_a : float

    filter_b : float

    Returns
    -------
    popt : float array
        The fit parameters - [white_noise_level, n, f_knee].
    pcov : float array
        Covariance matrix.
    f_fit : float array
        The frequency bins of the fit.
    Pxx_fit : float array
        The amplitude.
    """
    # incorporate timestream filtering
    status = load_session_status(stream_id, session_id)
    if p0 is None:
        p0 = [100.,0.5,0.01]
    if filter_b is None:
        filter_b = status.filter_b
    if filter_a is None:
        filter_a = status.filter_a

    if flux_ramp_freq is None:
        flux_ramp_freq = status.flux_ramp_rate_hz

    if fs is None:
        if status.downsample_enabled:
            fs = flux_ramp_freq/status.downsample_factor
        else:
            fs = flux_ramp_freq

    def noise_model(freq, wl, n, f_knee):
        """
        Crude model for noise modeling.
        Args
        ----
        wl : float
            White-noise level.
        n : float
            Exponent of 1/f^n component.
        f_knee : float
            Frequency at which white noise = 1/f^n component
        """
        A = wl*(f_knee**n)

        # The downsample filter is at the flux ramp frequency
        w, h = freqz(filter_b, filter_a, worN=freq,
                    fs=flux_ramp_freq)
        tf = np.absolute(h) # filter transfer function

        return (A/(freq**n) + wl)*tf

    bounds_low = [0.,0.,0.] # constrain 1/f^n to be red spectrum
    bounds_high = [np.inf,np.inf,np.inf]
    bounds = (bounds_low,bounds_high)

    try:
        popt, pcov = curve_fit(noise_model, f[1:], Pxx[1:],
                                p0=p0, bounds=bounds)
    except Exception:
        wl = np.mean(Pxx[1:])
        cprint('Unable to fit noise model. '+
                f'Reporting mean noise: {wl:.2f} pA/rtHz')

        popt = [wl, 1., 0.]
        pcov = None
    return popt

def get_white_noise_levels(am, stream_id, session_id, wl_f_range=(10,30),
                            fit = False,**psd_args):
    """

    Args:
        am: AxisManager
            axis manager loaded using G3tSmurf with timestamps and signal keys.
        stream_id: str

        session_id: int

        wl_f_range: float tuple

        fit: bool

    Returns:
        wls:

        band_medians:
    """
    f,pxx = get_psd(am.timestamps, am.signal, **psd_args)
    wls_tot = np.zeros((np.shape(pxx)[0],3))
    if fit = False:
        #Find white noise
        fmask = (wl_f_range[0] < f) & (f < wl_f_range[1])
        wls = np.median(pxx[:, fmask], axis=1)
        band_medians = np.zeros(8)
        for i in range(8):
            m = am.ch_info.band == i
            band_medians[i] = np.median(wls[m])
        #Calculate f_knee
        octaves = np.arange(np.floor(np.log10(f[1])),2)
        binedges = np.asarray([])
        for i, oc in enumerate(octaves[:-1]):
            if i == 0:
                binedges = np.concatenate((binedges,np.linspace(10**oc,10**octaves[i+1],10)))
            else:
                binedges = np.concatenate((binedges,np.linspace(10**oc,10**octaves[i+1],10)[1:]))
        start_bin = min(np.argsort(np.abs(binedges-f[1]))[0:2])
        binedges = binedges[start_bin:]
        bincenters = binedges[:-1]+np.diff(binedges)/2
        lowfn = np.zeros((len(wls),len(bincenters)))
        for ii,be in enumerate(binedges[:-1]):
            m = (f > be) & (f < binedges[ii+1])
            lowfn[:,ii] = np.median(Pxx[:,m],axis = 1)
        fknees = [lowfn[idx,np.nanargmin(np.abs(lowfn[idx]-wls[idx]*np.sqrt(2)))] for idx in range(len(wls))]
        m = [np.nanmax(lowfn[idx]) < np.sqrt(2)*wls[idx] for idx in range(len(wls))]
        fknees[m] = f[1]
        #Set wls_tot to have same structure as when we fit.
        wls_tot[:,0] = wls
        wls_tot[:,1] = np.full(len(wls),np.nan)
        wls_tot[:,2] = fknees
        return wls_tot, band_medians
    else:
        for i, ppxx in enumerate(pxx):
            wls_tot[i,:] = fit_noise_psd(f, ppxx, stream_id, session_id)
        band_medians = np.zeros(8)
        for i in range(8):
            m = am.ch_info.band == i
            band_medians[i] = np.median(wls_tot[m,0])
        return wls_tot, band_medians

def plot_band_noise(am, stream_id, session_id, nbins=40):
    """

    Args:
        am: AxisManager
            axis manager loaded using G3tSmurf with timestamps and signal keys.
        stream_id: str

        session_id: int

        nbins: int

    Returns:
        fig_wnl:

        axes_wnl:

        fig_fk:

        axes_fk:
    """
    bands = am.ch_info.band
    wls_tot, _ = get_white_noise_levels(am, stream_id, session_id)
    wls = wls_tot[:,0]
    fknees = wls_tot[:,2]

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
        text  = f"Median: {np.median(wls[m]):0.2f}\n"
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

    #Plot f_knee histograms
    fig_fk, axes_fk = plt.subplots(4, 2, figsize=(16, 8),
                             gridspec_kw={'hspace': 0})
    fig_fk.patch.set_facecolor('white')
    bins = np.logspace(np.floor(np.log10(np.min(fknees))),
                       np.ceil(np.log10(np.max(fknees))),
                       nbins)
    max_bins = 0

    for b in range(8):
        ax = axes_fk[b % 4, b // 4]
        m = bands == b
        x = ax.hist(fknees[m], bins=bins)
        text  = f"Median: {np.median(fknees[m]):0.2f}\n"
        text += f"Chans pictured: {np.sum(x[0]):0.0f}"
        ax.text(0.75, .7, text, transform=ax.transAxes)
        ax.axvline(np.median(fknees[m]), color='red')
        max_bins = max(np.max(x[0]), max_bins)
        ax.set(xscale='log', ylabel=f'Band {b}')

    axes_fk[0][0].set(title="AMC 0")
    axes_fk[0][1].set(title="AMC 1")
    axes_fk[-1][0].set(xlabel="$f_{knee} (Hz)")
    axes_fk[-1][1].set(xlabel="$f_{knee} (Hz)")
    for _ax in axes_fk:
        for ax in _ax:
            ax.set(ylim=(0, max_bins * 1.1))

    return fig_wnl, axes_wnl, fig_fk, axes_fk

def plot_channel_noise(am, stream_id, session_id, rchans = None):
    """

    Args:
        am: AxisManager
            axis manager loaded using G3tSmurf with timestamps and signal keys.
        stream_id: str

        session_id: int

        rchans: 

    Returns:
        fig_wnl:

        axes_wnl:

        fig_fk:

        axes_fk:
    """
    if rchans == None:
        f,pxx = get_psd(am.timestamps, am.signal, **psd_args)
        wls_tot, _ = get_white_noise_levels(am, stream_id, session_id)
    else:
        f,pxx = get_psd(am.timestamps, am.signal[rchans,:], **psd_args)
        wls_tot, _ = get_white_noise_levels(am, stream_id, session_id)
        wls_tot = wls_tot[rchans,:]
    for p in pxx:
        plt.figure()
