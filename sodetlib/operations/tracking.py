import sodetlib as sdl
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def compute_tracking_quality(S, f, df, sync):
    """
    Computes the tracking quality parameter from tracking_setup results.
    Tracking quality is a measure of how periodic the tracking response is with
    respect to the flux-ramp. It is defined as the r-squared value between the
    measured frequency response (f+df), and the (f+df) response averaged across
    all flux-ramp periods. If the signal is regular with respect to flux-ramp,
    the true freq response will be similar to the average freq response,
    resulting in a high r-squared value. If the signal is irregular, the true
    and average freq response will differ, resulting in a lower r-squared
    value. This can be used in conjunction with the f and df peak-to-peak
    values to determine which channels are tracking correctly.

    Args
    ------
    S : SmurfControl
        Pysmurf instance
    f : np.ndarray
        Array of the tracked frequency for each channelk (kHz), as returned by
        tracking_setup
    df : np.ndarray
        Array of the tracked frequency error for each channel (kHz), as
        returned by tracking_setup
    sync : np.ndarray
        Array containing tracking sync flags, as returned by tracking_setup
    """
    sync_idxs = S.make_sync_flag(sync)
    seg_size = np.min(np.diff(sync_idxs))
    nstacks = len(sync_idxs) - 1
    nchans = len(f[0])
    sig = f + df

    fstack = np.zeros((seg_size, nchans))
    for i in range(nstacks):
        si = sync_idxs[i]
        fstack[:seg_size] += (sig)[si:si+seg_size] / nstacks

    # calculates quality of estimate wrt real data
    y_real = (sig)[sync_idxs[0]:sync_idxs[0] + nstacks * seg_size, :]
    y_est = np.vstack([fstack for _ in range(nstacks)])

    # Force these to be the same len in case all segments are not the same size
    y_est = y_est[:len(y_real)]

    with np.errstate(invalid='ignore'):
        sstot = np.sum((y_real - np.nanmean(y_real, axis=0))**2, axis=0)
        ssres = np.sum((y_real - y_est)**2, axis=0)
        r2 = 1 - ssres/sstot

    return r2


class TrackingResults:
    """
    Class for storing, saving, and interpreting results from tracking_setup.
    This class can store tracking results for multiple bands at a time.
    When created, all results array are initialized to be empty. To add
    results from individual bands one at a time, use the ``add_band_data``
    function.

    Args
    ----
    S : SmurfControl
        Pysmurf Instance
    cfg : DetConfig
        DetConfig

    Attributes
    --------------
    meta : dict
        Metadata dictionary
    f_ptp_range : np.ndarray
        Array of len(2) containing the min and max allowed f_ptp. This will
        be pulled from the dev cfg.
    df_ptp_range : np.ndarray
        Array of len(2) containing the min and max allowed df_ptp. This will
        be pulled from the dev cfg.
    r2_min : float
        Min value of r2 for a channel to be considered good
    bands : np.ndarray
        Array of len (nchans) containing the band of each channel
    channels : np.ndarray
        Array of shape (nchans) containing the smurf-channel of each channel
    f : np.ndarray
        Array of shape (nchans, nsamps) containing the tracked freq response
        (kHz) throughout the tracking setup call
    df : np.ndarray
        Array of shape (nchans, nsamps) containing the untracked freq response
        (kHz) throughout the tracking setup call
    sync_idx : np.ndarray
        Array of shape (nchans, num_fr_periods) containing the indices where
        the flux ramp resetted
    r2 : np.ndarray
        Array of shape (nchans) containing the r-squared value computed by
        tracking-quality for each channel
    f_ptp : np.ndarray
        Array of shape (nchans) containing the f_ptp of each channel
    df_ptp : np.ndarray
        Array of shape (nchans) containing the df_ptp of each channel
    is_good : np.ndarray
        Array of shape (nchans) containing True if the channel passes cuts
        and False otherwise
    """
    def __init__(self, *args, **kwargs):
        if len(args) > 0:
            self.initialize(*args, **kwargs)

    def initialize(self, S, cfg):
        self._S = S
        self._cfg = cfg
        self.meta = sdl.get_metadata(S, cfg)
        self.f_ptp_range = np.array(cfg.dev.exp['f_ptp_range'])
        self.df_ptp_range = np.array(cfg.dev.exp['df_ptp_range'])
        self.r2_min = cfg.dev.exp['r2_min']

        self.bands = np.array([], dtype=int)
        self.channels = np.array([], dtype=int)
        self.subband_centers = np.array([], dtype=float)
        self.nchans = 0
        self.ngood = 0

        self.f = None
        self.df = None
        self.tracking_kwargs = [None for _ in range(8)]
        self.sync_idxs = [None for _ in range(8)]
        self.r2 = np.array([], dtype=float)
        self.f_ptp = np.array([], dtype=float)
        self.df_ptp = np.array([], dtype=float)
        self.is_good = np.array([], dtype=bool)

    def add_band_data(self, band, f, df, sync, tracking_kwargs=None):
        """
        Computes tracking-related data based on the tracking response,
        and updates the arrays described in the attributes with channels
        from a new band

        Args
        -----
        band : int
            Band of the data you're adding
        f : np.ndarray
            Tracked freq response as returned by tracking_setup
        df : np.ndarray
            Untracked freq response as returned by tracking_setup
        sync : np.ndarray
            sync arrayas returned by tracking_setup
        """
        if band in self.bands:
            raise ValueError(f"Data for band {band} has already been added!")

        dfptp = np.ptp(df, axis=0)
        m = dfptp != 0

        channels = np.where(m)[0]
        nchans = len(channels)
        bands = np.array([band for _ in channels])
        sb_centers = np.array(
            self._S.get_subband_centers(band, as_offset=False)[1]
        )
        self.channels = np.concatenate((self.channels, channels))
        self.bands = np.concatenate((self.bands, bands))
        self.subband_centers = np.concatenate((
                self.subband_centers, sb_centers[channels]))

        self.tracking_kwargs[band] = tracking_kwargs
        r2 = compute_tracking_quality(self._S, f, df, sync)

        if self.f is not None:
            # It's possible for different tracking setup calls to have a
            # slightly different number of samples, so we have to make sure
            # we cut/elongate results so they can fit into the f/df arrays
            nsamps = len(self.f[0])
            _f = np.full((nchans, nsamps), np.nan)
            _df = np.full((nchans, nsamps), np.nan)
            fi = min(nsamps, len(f))
            _f[:, :fi] = f.T[m, :fi] * 1000
            _df[:, :fi] = df.T[m, :fi] * 1000
            if fi < nsamps:
                # Fill with the last data point to not mess up ptp calcs....
                _f[:, fi:] = _f[:, fi-1][:, None]
                _df[:, fi:] = _df[:, fi-1][:, None]
        else:
            _f = f.T[m] * 1000
            _df = df.T[m] * 1000

        self.r2 = np.concatenate((self.r2, r2[m]))

        f_ptp = np.ptp(_f, axis=1)
        self.f_ptp = np.concatenate((self.f_ptp, f_ptp))

        df_ptp = np.ptp(_df, axis=1)
        self.df_ptp = np.concatenate((self.df_ptp, df_ptp))

        is_good = np.ones_like(r2, dtype=bool)
        self.is_good = np.concatenate((self.is_good, is_good))

        if self.f is None:
            self.f = _f
            self.df = _df
        else:
            nsamps = len(self.f[0])
            self.f = np.concatenate((self.f, _f[:, :nsamps]), axis=0)
            self.df = np.concatenate((self.df, _df[:, :nsamps]), axis=0)

        self.sync_idxs[band] = self._S.make_sync_flag(sync)
        self.nchans += len(_f)
        self.find_bad_chans()

    def find_bad_chans(self, f_ptp_range=None, df_ptp_range=None, r2_min=None):
        """
        Recomputes the ``is_good`` array based on cuts ranges.
        """
        if f_ptp_range is None:
            f_ptp_range = self.f_ptp_range
        if df_ptp_range is None:
            df_ptp_range = self.df_ptp_range
        if r2_min is None:
            r2_min = self.r2_min

        f0, f1 = f_ptp_range
        df0, df1 = df_ptp_range

        self.is_good = np.ones_like(self.r2, dtype=bool)
        self.is_good[self.r2 < r2_min] = 0
        self.is_good[~((f0 < self.f_ptp) & (self.f_ptp < f1))] = 0
        self.is_good[~((df0 < self.df_ptp) & (self.df_ptp < df1))] = 0
        self.ngood = np.sum(self.is_good)

    def save(self, path=None):
        saved_fields = [
            'meta', 'bands', 'channels',
            'f_ptp_range', 'df_ptp_range', 'r2_min',
            'f', 'df', 'sync_idxs', 'r2', 'is_good',
            'tracking_kwargs', 'f_ptp', 'df_ptp',
        ]
        data = {}
        for field in saved_fields:
            data[field] = getattr(self, field)
        if path is None:
            path = sdl.make_filename(self._S, 'tracking_results.npy')

        self.filepath = path
        np.save(path, data, allow_pickle=True)
        if self._S is not None:
            self._S.pub.register_file(path, 'tracking_data', format='npy')

    @classmethod
    def load(cls, path):
        self = cls()
        for k, v in np.load(path, allow_pickle=True).item().items():
            setattr(self, k, v)
        return self


def plot_tracking_channel(tr, idx, show_text=True):
    """
    Plots single tracking channel from results
    """
    f = tr.f[idx]
    df = tr.df[idx]
    fig, ax = plt.subplots(figsize=(12, 4))
    band = int(tr.bands[idx])
    channel = int(tr.channels[idx])

    ax.set_title(f"Band {band} Channel {channel}")
    ax.plot(df+f, color='grey', alpha=0.8, label='Freq Response')
    ax.plot(f, label='Tracked Freq')
    ax.set_xticks([])
    if hasattr(tr, 'subband_centers'):
        ax.set_ylabel(f"Freq offset from {tr.subband_centers[idx]:0.2f} [kHz]")
    else:
        ax.set_ylabel(f"Freq offset [kHz]")
    ax.legend(loc='upper left')

    if show_text:
        txt = '\n'.join([
            f'fptp = {tr.f_ptp[idx]:0.2f}',
            f'dfptp = {tr.df_ptp[idx]:0.2f}',
            f'r2 = {tr.r2[idx]:0.2f}',
            f'is_good = {tr.is_good[idx]}',
        ])
        bbox = dict(facecolor='white', alpha=0.8)
        ax.text(0.02, 0.1, txt, transform=ax.transAxes, bbox=bbox)
    for s in tr.sync_idxs[band]:
        ax.axvline(s, color='grey', ls='--')
    return fig, ax


def plot_tracking_summary(tr):
    """
    Plots summary of tracking results
    """
    # fig, axes = plt.subplots(figsize=(12, 8))
    mosaic = """
    BBBB.
    AAAAC
    AAAAC
    AAAAC
    AAAAC
    """
    gs = dict(hspace=0, wspace=0)
    fig, axes = plt.subplot_mosaic(mosaic, figsize=(12, 8), gridspec_kw=gs)
    f0, f1 = tr.f_ptp_range
    df0, df1 = tr.df_ptp_range

    m = tr.is_good
    ax = axes['A']
    ax.scatter(tr.f_ptp[m], tr.df_ptp[m], marker='.', alpha=0.8)
    ax.scatter(tr.f_ptp[~m], tr.df_ptp[~m], marker='.', color='red', alpha=0.2)
    xlim = (-5, f1 * 1.3)
    ylim = (-5, df1 * 1.3)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    rect = Rectangle((f0, df0), f1-f0, df1-df0, fc='green', alpha=0.2,
                     linestyle='--', ec='black', lw=3)
    ax.add_patch(rect)
    ax.set_ylabel(r"$df_\mathrm{ptp}$ (kHz)", fontsize=20)
    ax.set_xlabel(r"$f_\mathrm{ptp}$ (kHz)", fontsize=20)

    text = f"{tr.ngood} / {tr.nchans} good tracking channels"
    bbox = dict(fc='wheat', alpha=0.8)
    ax.text(0.02, 0.9, text, transform=ax.transAxes, bbox=bbox, fontsize=16)

    ax = axes['B']
    ax.set_xticks([])
    ax.hist(tr.f_ptp, range=xlim, bins=40)
    ax.set_xlim(*xlim)
    ax.axvspan(f0, f1, color='green', alpha=0.2)

    ax = axes['C']
    ax.set_yticks([])
    ax.hist(tr.df_ptp, range=ylim, bins=40, orientation='horizontal')
    ax.set_ylim(*ylim)
    ax.axhspan(df0, df1, color='green', alpha=0.2)

    return fig, ax


def disable_bad_chans(S, tr, bands=None, **kwargs):
    """
    Disables cut channels based on a TrackingResults object.
    """
    tr.find_bad_chans(**kwargs)
    if bands is None:
        bands = np.arange(8)

    bands = np.atleast_1d()
    for b in bands:
        m = tr.bands == b
        asa = S.get_amplitude_scale_array(b)
        fea = S.get_feedback_enable_array(b)
        for good, c in zip(tr.is_good[m], tr.channels[m]):
            if not good:
                asa[c] = 0
                fea[c] = 0
        S.set_amplitude_scale_array(b, asa)
        S.set_feedback_enable_array(b, asa)


@sdl.set_action()
def setup_tracking_params(S, cfg, bands, update_cfg=True, show_plots=False):
    """
    Sets up tracking parameters by determining correct frac-pp and lms-freq
    for each band.

    Args
    -----
    S : SmurfControl
        Pysmurf instance
    cfg : DetConfig
        DetConfig instance
    bands : np.ndarray, int
        Band or list of bands to run on
    update_cfg : bool
        If true, will update the device cfg and save the file.
    show_plots : bool
        If true, will show summary plots
    """

    bands = np.atleast_1d(bands)
    exp = cfg.dev.exp

    res = TrackingResults(S, cfg)

    tk = {
        'reset_rate_khz': exp['flux_ramp_rate_khz'],
        'make_plot': False, 'show_plot': False, 'channel': [],
        'nsamp': 2**18, 'return_data': True,
        'feedback_start_frac': exp['feedback_start_frac'],
        'feedback_end_frac': exp['feedback_end_frac'],
    }
    for band in bands:
        sdl.pub_ocs_log(S, f"Setting up trackng params: band {band}")
        bcfg = cfg.dev.bands[band]
        tk.update({
            'lms_freq_hz':         None,
            'meas_lms_freq':       True,
            'fraction_full_scale': exp['init_frac_pp'],
            'reset_rate_khz':      exp['flux_ramp_rate_khz'],
            'lms_gain': bcfg['lms_gain'],
            'feedback_gain': bcfg['feedback_gain'],
        })

        f, df, sync = S.tracking_setup(band, **tk)
        tr = TrackingResults(S, cfg)
        tr.add_band_data(band, f, df, sync)
        asa_init = S.get_amplitude_scale_array(band)
        disable_bad_chans(S, tr, bands=band, r2_min=0.95)

        # Calculate trracking parameters
        S.tracking_setup(band, **tk)
        lms_meas = S.lms_freq_hz[band]
        lms_freq = exp['nphi0'] * tk['reset_rate_khz'] * 1e3
        frac_pp = tk['fraction_full_scale'] * lms_freq / lms_meas

        # Re-enables all channels and re-run tracking setup with correct params
        S.set_amplitude_scale_array(band, asa_init)
        tk.update({
            'meas_lms_freq': False,
            'fraction_full_scale': frac_pp,
            'lms_freq_hz': lms_freq,
        })

        f, df, sync = S.tracking_setup(band, **tk)
        res.add_band_data(band, f, df, sync, tracking_kwargs=tk)

        # Update det config
        if update_cfg:
            cfg.dev.update_band(band, {
                'frac_pp':            frac_pp,
                'lms_freq_hz':        lms_freq,
            }, update_file=True)

    res.save()
    is_interactive = plt.isinteractive()
    try:
        if not show_plots:
            plt.ioff()
        fig, ax = plot_tracking_summary(res)
        path = sdl.make_filename(S, 'tracking_results.png', plot=True)
        fig.savefig(path)
        S.pub.register_file(path, 'tracking_summary', plot=True, format='png')
        if not show_plots:
            plt.close(fig)
    finally:
        if is_interactive:
            plt.ion()
    return res


@sdl.set_action()
def relock_tracking_setup(S, cfg, bands, reset_rate_khz=None, nphi0=None,
                          feedback_gain=None, lms_gain=None, show_plots=False):
    """
    Sets up tracking for smurf. This assumes you already have optimized
    lms_freq and frac-pp for each bands in the device config. This function
    will chose the flux-ramp fraction-full-scale by averaging the optimized
    fractions across the bands you're running on.

    This function also allows you to set reset_rate_khz and nphi0. The
    fraction-full-scale, and lms frequencies of each band will be automatically
    adjusted based on their pre-existing optimized values.

    Additional keyword args specified will be passed to S.tracking_setup.

    Args
    -----
    S : SmurfControl
        Pysmurf instance
    cfg : DetConfig
        Det config instance
    reset_rate_khz : float, optional
        Flux Ramp Reset Rate to set (kHz), defaults to the value in the dev cfg
    nphi0 : int, optional
        Number of phi0's to ramp through. Defaults to the value that was used
        during setup.
    disable_bad_chans : bool
        If true, will disable tones for bad-tracking channels

    Returns
    --------
    res : dict
        Dictionary of results of all tracking-setup calls, with the bands number
        as key.
    """
    bands = np.atleast_1d(bands)
    nbands = len(bands)
    exp = cfg.dev.exp

    # Arrays containing the optimized tracking parameters for each band
    frac_pp0 = np.zeros(nbands)
    lms_freq0 = np.zeros(nbands)  # Hz
    reset_rate_khz0 = exp['flux_ramp_rate_khz']
    init_nphi0 = exp['nphi0']

    for i, b in enumerate(bands):
        bcfg = cfg.dev.bands[b]
        frac_pp0[i] = bcfg['frac_pp']
        lms_freq0[i] = bcfg['lms_freq_hz']

    # Choose frac_pp to be the mean of all running bands.
    # This is the frac-pp at the flux-ramp-rate used for optimization
    fpp0 = np.median(frac_pp0)

    # Adjust fpp, lms_freq, and flux-ramp-rate depending on desired
    # flux-ramp-rate and nphi0
    fpp, lms_freqs = fpp0, lms_freq0
    if nphi0 is not None:
        fpp *= nphi0 / init_nphi0
        lms_freqs *= fpp / fpp0
    if reset_rate_khz is not None:
        lms_freqs *= reset_rate_khz / reset_rate_khz0
    else:
        reset_rate_khz = reset_rate_khz0

    res = TrackingResults(S, cfg)
    tk = {
        'reset_rate_khz': reset_rate_khz, 'fraction_full_scale': fpp,
        'make_plot': False, 'show_plot': False, 'channel': [],
        'nsamp': 2**18, 'return_data': True,
        'feedback_start_frac': 0.02, 'feedback_end_frac': 0.94,
    }

    if lms_gain is not None:
        tk['lms_gain'] = lms_gain
    if feedback_gain is not None:
        tk['feedback_gain'] = feedback_gain

    for i, band in enumerate(bands):
        tk.update({'lms_freq_hz': lms_freqs[i]})
        f, df, sync = S.tracking_setup(band, **tk)
        res.add_band_data(band, f, df, sync, tracking_kwargs=tk)

    res.find_bad_chans()
    res.save()

    is_interactive = plt.isinteractive()
    try:
        if not show_plots:
            plt.ioff()
        fig, ax = plot_tracking_summary(res)
        path = sdl.make_filename(S, 'tracking_results.png', plot=True)
        fig.savefig(path)
        S.pub.register_file(path, 'tracking_summary', plot=True, format='png')
        if not show_plots:
            plt.close(fig)
    finally:
        if is_interactive:
            plt.ion()

    return res
