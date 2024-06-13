import numpy as np
from tqdm.auto import tqdm, trange
import time
import numpy as np
import sodetlib as sdl
from sodetlib.constants import *
from scipy.signal import welch, hilbert
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from sotodlib.tod_ops.filters import gaussian_filter, fourier_filter 
from sotodlib.core import IndexAxis, AxisManager, OffsetAxis, LabelAxis


def new_ci_dset(S, cfg, bands, chans, freqs, run_kwargs=None, ob_path=None,
                sc_path=None):
    """
    Creates a new CIData AxisManager. If ob_path and sc_path are set, they
    will be loaded with a remapped "dets" axis.
    """
    ndets = len(chans)
    nsteps = len(freqs)

    ds = AxisManager(
        LabelAxis('dets', vals=[f"r{x:0>4}" for x in range(ndets)]),
        IndexAxis('steps', count=nsteps),
        IndexAxis('biaslines', count=NBGS),
    )
    ds.wrap('meta', sdl.dict_to_am(sdl.get_metadata(S, cfg), skip_bad_types=True))
    ds.meta.wrap('g3_dir', cfg.sys['g3_dir'])
    if run_kwargs is not None:
        ds.wrap('run_kwargs', sdl.dict_to_am(run_kwargs))

    ds.wrap('bands', bands, [(0, 'dets')])
    ds.wrap('channels', chans, [(0, 'dets')])
    ds.wrap('freqs', freqs, [(0, 'steps')])

    ds.wrap_new('start_times', ('biaslines', 'steps'))
    ds.wrap_new('stop_times', ('biaslines', 'steps'))
    ds.wrap_new('sids', ('biaslines',), dtype=int)

    # Load bgmap stuff
    bgmap, polarity = sdl.load_bgmap(
        bands, chans, cfg.dev.exp['bgmap_file']
    )
    ds.wrap('bgmap', bgmap, [(0, 'dets')])
    ds.wrap('polarity', polarity, [(0, 'dets')])


    ob_path = cfg.dev.exp.get('complex_impedance_ob_path')
    sc_path = cfg.dev.exp.get('complex_impedance_sc_path')
    if ds.run_kwargs.state == 'transition':
        if ob_path is None:
            raise ValueError("No ob CI path found in the device cfg")
        if sc_path is None:
            raise ValueError("No ob CI path found in the device cfg")

        ob = sdl.remap_dets(AxisManager.load(ob_path), ds, load_axes=False)
        ds.wrap('ob', ob)

        sc = sdl.remap_dets(AxisManager.load(sc_path), ds, load_axes=False)
        ds.wrap('sc', sc)

    return ds

def A_per_bit(ds):
    return 2 * ds.meta['rtm_bit_to_volt']          \
        / ds.meta['bias_line_resistance']          \
        * ds.meta['high_low_current_ratio']

def load_tod(ds, bg, arc=None):
    """
    Loads a TOD for a biasgroup given a CI dset.

    Args
    -----
    ds : AxisManager
        CI Dataset
    bg : int
        Bias group to load data for
    arc : G3tSmurf, optional
        If a G3tSmurf archive is passed this will be used to load data.
    """
    if arc is not None:
        start = ds.start_times[bg, 0]
        stop = ds.stop_times[bg, -1]
        seg = arc.load_data(start, stop, show_pb=False)
    else:
        sid = ds.sids[bg]
        seg = sdl.load_session(ds.meta.stream_id, sid, base_dir=ds.meta.g3_dir)
    return seg

################################################################################
# CI Analysis
################################################################################
def analyze_seg(ds, tod, bg, i):
    """
    Analyze segment of CI data. The main goal of this is to calculate
    the Ites phasor, containing the amplitude (A) and phase (relative
    tot he commanded) of the TES response to an incoming sine wave.
    This performs the following steps:
     1. Restrict full TOD to a single excitation frequency. This will put
        everything units of A, correct for channel polarity. This will also
        correct timestamps based on the FrameCounter.
     2. Takes PSD of the bias data to obtain the reference freq. Filters signal
        using gaussian filter around reference freq.
     3. Use lock-in amplification with bias as reference to extract
        amplitude and phase of the filtered signal with respect to the
        commanded bias.

    Args
    -----
    ds : AxisManager
        CI Dataset
    tod : AxisManger
        axis-manager containing tod for a given bias group
    bg : int
        Bias group that is being analyzed
    i : int
        Freq index. 0 will analyze the first freq segment taken.

    Returns
    ---------
    am : AxisManager
        Returns an souped-up tod axis-manager corresponding to this freq with
        the following fields:
         - timestmaps, biases, signal, ch_info. Standard tod axismanager stuff
           but with units converted into A, timestamps fixed, and offsets
           subtracted out.
         - sample_rate, cmd_freq: Floats with the sample rate and commanded
           freq
         - filt_sig: Filtered signal (using gaussian filter around commanded
           freq)
         - lockin_x, lockin_y: Lockin x and y signals, used to calc amp and phase
           across the tod
         - Ites: Phasor for Ites. Amplitude is the amp of the sine wave response,
           and angle is the phase relative to the commanded bias.
    """
    t0, t1 = ds.start_times[bg, i], ds.stop_times[bg, i]
    am = sdl.restrict_to_times(tod, t0, t1, in_place=False)

    sample_rate = 1./np.median(np.diff(am.timestamps))

    # Convert everything to A
    am.signal = am.signal * ds.meta['pA_per_phi0'] / (2*np.pi) * 1e-12

    # Index mapping from am readout channel to sweep chan index.
    chan_idxs = sdl.map_band_chans(
        am.ch_info.band, am.ch_info.channel,
        ds.bands, ds.channels
    )
    am.signal *= ds.polarity[chan_idxs, None]

    am.biases = am.biases * A_per_bit(ds)

    # Remove offset from signal
    am.signal -= np.mean(am.signal, axis=1)[:, None]
    am.biases -= np.mean(am.biases, axis=1)[:, None]

    # Fix up timestamps based on frame-counter
    t0, t1 = am.timestamps[0], am.timestamps[-1]
    fc = am.primary["FrameCounter"]
    fc = fc - fc[0]
    ts = t0 + fc/fc[-1] * (t1 - t0)
    am.timestamps = ts
    am.wrap('sample_rate', sample_rate)
    am.wrap('cmd_freq', ds.freqs[i])
    
    # Get psds cause we'll want that
    nsamp = len(am.signal[0])
    fxx, bias_pxx = welch(am.biases[bg], fs=sample_rate, nperseg=nsamp)

    # Gaussian filter around peak freq of bias asd.
    f0, f1 = .9 * am.cmd_freq, 1.1*am.cmd_freq
    m = (f0 < fxx) & (fxx < f1)

    idx = np.argmax(bias_pxx[m])
    f = fxx[m][idx]

    filt = gaussian_filter(f, f_sigma=f / 5)
    filt_sig = fourier_filter(am, filt)
    am.wrap('filt_sig', filt_sig, [(0, 'dets'), (1, 'samps')])

    # Lock in amplification!
    # To get ref + ref offset by 90-deg, take Hilbert transform, and then 
    # real part gives you ref and imag part is offset by 90 deg.
    sig = filt_sig
    ref = hilbert(am.biases[bg] / np.max(am.biases[bg]))
    X = sig * ref.real
    Y = sig * ref.imag
    # We're averaging over enough periods where we don't really need to
    # restrict to an int number of periods...
    xmean = np.mean(X, axis=1)
    ymean = np.mean(Y, axis=1)
    phase = -np.arctan2(ymean, xmean)
    amp = 2*np.sqrt(xmean**2 + ymean**2)
    Ites = amp * np.exp(1.0j * phase)
    am.wrap('lockin_x', X)
    am.wrap('lockin_y', Y)
    am.wrap('Ites', Ites)
    return am

def analyze_tods(ds, bgs=None, tod=None, arc=None, show_pb=True):
    """
    Analyzes TODS for a CIData set. This will add the following fields to the
    dataset:
      - Ites(dets, steps): Ites phasor for each detector / frequency
        combination. Amplitude is the amp of the current response (A), and angle
        is the phase relative to commanded bias.
      - Ibias(biaslines): Amplitude (A) of the sinewave used for each
        biasline.
      - Ibias_dc(biaslines): DC bias current (A) for each biasline
      - res_freqs(dets): Resonance frequency of each channel detectors.
    """
    if bgs is None:
        bgs = ds.run_kwargs.bgs
    bgs = np.atleast_1d(bgs)

    # Delete temp fields if they exist
    for f in ['_Ites', '_Ibias', '_Ibias_dc', '_res_freqs']:
        if f in ds._fields:
            ds.move(f, None)

    nsteps = len(ds.freqs)
    ds.wrap_new('_Ites', ('dets', nsteps), cls=np.full,
                fill_value=np.nan, dtype=np.complex128)
    ds.wrap_new('_Ibias', (NBGS,), cls=np.full, fill_value=np.nan)
    ds.wrap_new('_Ibias_dc', (NBGS,), cls=np.full, fill_value=np.nan)
    ds.wrap_new('_res_freqs', ('dets',), cls=np.full, fill_value=np.nan)

    ntot = len(bgs) * len(ds.freqs)
    pb = tqdm(total=ntot, disable=(not show_pb))
    for bg in bgs:
        if tod is None:
            pb.set_description(f"Loading tod for bg {bg}")
            _tod = load_tod(ds, bg, arc=arc)
        else:
            _tod = tod
        chmap = sdl.map_band_chans(
            _tod.ch_info.band, _tod.ch_info.channel,
            ds.bands, ds.channels
        )

        pb.set_description(f"Analyzing segments for bg {bg}")
        for i in range(len(ds.freqs)):
            try:
                seg = analyze_seg(ds, _tod, bg, i)
            except sdl.RestrictionException:
                # Means there's no data at the specified time
                pb.update()
                continue
            if i == 0:
                ds._Ibias_dc[bg] = np.mean(seg.biases[bg])
                ds._Ibias[bg] = 0.5 * np.ptp(seg.biases[bg])
                ds._res_freqs[chmap] = seg.ch_info.res_frequency

            ds._Ites[chmap, i] = seg.Ites
            pb.update()
        del _tod

    for f in ['_Ites', '_Ibias', '_Ibias_dc', '_res_freqs']:
        ds.move(f, f[1:])

    return ds

def get_ztes(ds):
    """
    Calculates Ztes for in-transition CIData. Adds the following fields to the
    CI dataset:
      - Rn (dets): Normal resistances based off of low-f overbiased data points.
      - Rtes (dets): TES Resistance, based off of low-f in-transition segment
      - Vth (dets): Thevenin equiv voltage (V)
      - Zeq (dets): Equiv impedance
      - Ztes (dets): TES complex impedance
    """
    ob, sc = ds.ob, ds.sc

    fields = ['_Rn', '_Rtes', '_Vth', '_Zeq', '_Ztes']
    for f in fields:
        if f in ds._fields:
            ds.move(f, None)
 
    ds.wrap_new('_Rn', ('dets',))
    ds.wrap_new('_Rtes', ('dets',))
    ds.wrap_new('_Vth', ('dets',))
    ds.wrap_new('_Zeq', ('dets',))
    ds.wrap_new('_Ztes', ('dets',))
   
    # Calculates Rn
    Ib_ob = ob.Ibias[ob.bgmap][:, None]
    ds._Rn = ob.meta.R_sh * (np.abs(Ib_ob / ob.Ites) - 1)[:, 0]

    # Calculate Rtes for in-transition dets
    Ib = ds.Ibias_dc[ds.bgmap]

    dIrat = np.real(ds.Ites[:, 0]) / np.abs(ds.Ibias[ds.bgmap])
    I0 = Ib * dIrat / (2 * dIrat - 1)
    Pj = I0 * ds.meta.R_sh * (Ib - I0)
    ds._Rtes = np.abs(Pj / I0**2)

    Ites_ob = np.zeros_like(ds.Ites)
    Ites_sc = np.zeros_like(ds.Ites)
    for rc in range(len(ob.channels)):
        Ites_ob[rc, :] = np.interp(ds.freqs, ob.freqs, ob.Ites[rc])
        Ites_sc[rc, :] = np.interp(ds.freqs, sc.freqs, sc.Ites[rc])
        
    ds._Vth = 1./((1./Ites_ob - 1./Ites_sc) / ds._Rn[:, None])
    ds._Zeq = ds._Vth / Ites_sc

    ds._Ztes = ds._Vth / ds.Ites - ds._Zeq

    for f in fields:
        ds.move(f, f[1:])

    return ds


def Ztes_fit(f, R, beta, L, tau):
    """
    Ztes equation from Irwin/Shaw eq 42 
    """
    return R * (1 + beta) \
        + R * L / (1 - L) * (2 + beta) / (1 + 2j * np.pi * f * tau)

def guess_fit_params(ds, idx):
    """
    Gets initial params for fit at a particular freq idx
    """
    R = ds.Rtes[idx]

    min_idx = np.argmin(np.imag(ds.Ztes[idx]))
    tau_guess = -1./(2*np.pi*ds.freqs[min_idx])

    beta_guess = np.abs(ds.Ztes[idx, -1]) / R - 1

    L_guess = 1000

    return (R, beta_guess, L_guess, tau_guess)


def fit_single_det_params(ds, idx, x0=None, weights=None, fmax=None):
    """
    Fits detector parameters for a single channel
    """
    R = ds.Rtes[idx]
    if x0 is None:
        x0 = guess_fit_params(ds, idx)

    if weights is None:
        weights = np.ones_like(ds.freqs)

    if fmax is not None:
        weights[ds.freqs > fmax] = 0

    def chi2(x):
        zfit = Ztes_fit(ds.freqs, *x)
        c2 = np.nansum(weights * np.abs(ds.Ztes[idx] - zfit)**2)
        return c2

    res = minimize(chi2, x0)
    if list(res.x) == list(x0):
        res.success = False

    return res

def fit_det_params(ds, pb=False, fmax=None):
    """
    Fits detector params for a sweep.

    Args
    -----
    ds : AxisManager
        CIData
    pb : bool
        If True, wil display progressbar. 
    fmax : optional, float
        If set, will only fit using freq values less than fmax.
    """
    fields = ['_fit_x', '_fit_labels', '_tau_eff', '_Rfit', '_beta_I', '_L_I',
        '_tau_I']

    for f in fields:
        if f in ds._fields:
            ds.move(f, None)

    for f in ['_tau_eff', '_Rfit', '_beta_I', '_L_I', '_tau_I']:
        ds.wrap_new(f, ('dets', ), cls=np.full, fill_value=np.nan)
    ds.wrap('_fit_labels', np.array(['R', 'beta_I', 'L_I', 'tau_I']))
    ds.wrap_new('_fit_x', ('dets', 4), cls=np.full, fill_value=np.nan)

    for i in trange(len(ds.channels), disable=(not pb)):
        if ds.bgmap[i] == -1: continue

        res = fit_single_det_params(ds, i, fmax=fmax)
        ds._fit_x[i, :] = res.x
    
    # Compute tau_eff
    RL = ds.meta.R_sh
    ds._Rfit, ds._beta_I, ds._L_I, ds._tau_I = ds._fit_x.T
    ds.tau_eff = ds._tau_I * (1 - ds._L_I) * (1 + ds._beta_I + RL / ds._Rfit) \
        / (1 + ds._beta_I + RL / ds._Rfit + ds._L_I * (1 - RL / ds._Rfit))

    for f in fields:
        ds.move(f, f[1:])

    return True

def analyze_full(ds):
    """
    Performs the full CI analysis on a dataset.
    """
    analyze_tods(ds)
    if ds.run_kwargs.state == 'transition':
        get_ztes(ds)
        fit_det_params(ds)
    return ds


###########################################################################
# Plotting functions
###########################################################################
def plot_transfers(d, rc):
    """
    Plot the SC, OB, and in-transition transfer functions for a channel
    """
    bg = d.bgmap[rc]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    labels = ['Transition', 'superconducting', 'overbiased']

    for i, s in enumerate([d, d.sc, d.ob]):
        mag = np.abs(s.Ites[rc])/s.Ibias[bg]
        phase = np.unwrap(np.angle(s.Ites[rc]))
        axes[0].plot(s.freqs, mag, '.', label=labels[i])
        axes[1].plot(s.freqs, phase, '.', label=labels[i])
    axes[0].set(xscale='log', yscale='log')
    axes[0].legend()
    axes[1].legend()
    return fig, axes

def plot_ztes(ds, rc, x=None, write_text=True):
    """
    plots Ztes data

    Args
    -----
    ds : AxisManager
        analyzed CIData object
    rc : int
        Channel whose data to plot
    x : optional, list
        Fit params to plot instead of the ones stored in the CIData object.
    write_text : bool
        If textbox with param data should be written.
    """
    dims = np.array([2.5, 1])
    fig, axes = plt.subplots(1, 2, figsize=5 * dims)

    ztes = 1000 * ds.Ztes[rc]
    fs = np.linspace(0, np.max(ds.freqs), 1000)
    if x is None:
        x = ds.fit_x[rc]
    zfit = 1000 * Ztes_fit(fs, *x)

    # Circ plot
    ax = axes[0]
    ax.scatter(np.real(ztes), np.imag(ztes), c=np.log(ds.freqs), marker='.')
    ax.plot(np.real(zfit), np.imag(zfit), color='black', ls='--', alpha=0.6)
    ax.set_xlabel(r'Re[$Z_\mathrm{TES}$] (m$\Omega$)', fontsize=16)
    ax.set_ylabel(r'Im[$Z_\mathrm{TES}$] (m$\Omega$)', fontsize=16)

    ## Param summary
    txt = '\n'.join([
        r'$\tau_\mathrm{eff}$ = ' + f'{ds.tau_eff[rc]*1000:.2f} ms',
        r'$R_\mathrm{fit}$ = ' + f'{x[0]*1000:.2f} '+ r'm$\Omega$',
        r'$\beta_I$ = ' + f'{x[1]:.2f}',
        r'$\mathcal{L}_I$ = ' + f'{x[2]:.2f}',
        r'$\tau_I$ = ' + f'{x[3]*1000:.2f} ms',
        # r'$r^2$ = ' + f'{ds.ztes_rsquared[rc]:.2f}',
    ])
    if write_text:
        ax.text(0.05, 0.05, txt, transform=ax.transAxes, fontsize=12,
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'))

    # Im / Re plot
    ax = axes[1]
    ax.plot(ds.freqs, np.real(ztes), '.', color='C0')
    ax.plot(ds.freqs, np.imag(ztes), '.', color='C1')
    ax.plot(fs, np.real(zfit), color='C0', alpha=0.8, ls='--')
    ax.plot(fs, np.imag(zfit), color='C1', alpha=0.8, ls='--')
    ax.set(xscale='log')
    ax.set_xlabel("Freq (Hz)", fontsize=16)
    ax.set_ylabel(r"$Z_\mathrm{TES}$ (m$\Omega$)", fontsize=16)

    return fig, ax

###########################################################################
# Data Taking Functions
###########################################################################
@sdl.set_action()
def take_complex_impedance(
        S, cfg, bgs, freqs=None, state='transition', nperiods=500,
        max_meas_time=20., tickle_voltage=0.005, run_analysis=False):
    """
    Takes a complex impedance sweep. This will play sine waves on specified
    bias-groups over the current DC bias voltage. This returns a CISweep object.

    Args
    ----
    S : SmurfControl
        Pysmurf Instance
    cfg : DetConfig
        Det config instance
    bgs : array, int
        List of bias groups to run on
    freqs : array, optional
        List of frequencies to sweep over.
    state : str
        Current detector state. Must be 'ob', 'sc', or 'transition'
    nperiods : float
        Number of periods to measure for at each frequency. If the meas_time 
        ends up larger than ``max_meas_time``, ``max_meas_time`` will be used
        instead. This makes it so we don't spend unreasonably long amounts of
        time at higher freqs.
    max_meas_time : float
        Maximum amount of time to wait at any given frequency
    tickle_voltage : float
        Tickle amplitude in low-current-mode volts.
    """
    if state not in ['ob', 'sc', 'transition']:
        raise ValueError("State must be 'ob', 'sc', or 'transition'")

    bgs = np.atleast_1d(bgs)

    if freqs is None:
        freqs = np.logspace(0, np.log10(4e3), 20)
    freqs = np.atleast_1d(freqs)

    run_kwargs = {k: v for k, v in locals().items() if k not in ['S', 'cfg']}

    # First, determine which bands and channels we'll try to run on.
    scale_array = np.array([S.get_amplitude_scale_array(b) for b in range(8)])
    bands, channels = np.where(scale_array > 0)

    # Main dataset
    ds = new_ci_dset(S, cfg, bands, channels, freqs, run_kwargs=run_kwargs)

    initial_ds_factor = S.get_downsample_factor()
    initial_filter_disable = S.get_filter_disable()

    pb = tqdm(total=len(freqs)*len(bgs), disable=False)
    try:
        S.set_downsample_factor(1)
        S.set_filter_disable(1)
        sdl.set_current_mode(S, bgs, 1)
        tickle_voltage /= S.high_low_current_ratio

        init_biases = S.get_tes_bias_bipolar_array()
        for bg in bgs:
            m = ds.bgmap == bg
            channel_mask = ds.bands[m] * S.get_number_channels() + ds.channels[m]

            ds.sids[bg] = sdl.stream_g3_on(
                S, channel_mask=channel_mask, subtype='complex_impedance')
            for j, freq in enumerate(freqs):
                meas_time = min(1./freq * nperiods, max_meas_time)
                S.log(f"Tickle with bg={bg}, freq={freq}")
                S.play_sine_tes(bg, tickle_voltage, freq)
                ds.start_times[bg, j] = time.time()
                time.sleep(meas_time)
                ds.stop_times[bg, j] = time.time()
                S.set_rtm_arb_waveform_enable(0)
                S.set_tes_bias_bipolar(bg, init_biases[bg])
                pb.update()
            sdl.stream_g3_off(S)
    finally:
        sdl.set_current_mode(S, bgs, 0)
        S.set_downsample_factor(initial_ds_factor)
        S.set_filter_disable(initial_filter_disable)
        sdl.stream_g3_off(S)

    fname = sdl.make_filename(S, f'ci_sweep_{state}.h5')
    ds.wrap('filepath', fname)
    ds.save(fname)
    S.pub.register_file(fname, 'ci', format='h5', plot=False)
    S.log(f"Saved unanalyzed datafile to {fname}")

    if run_analysis:
        analyze_full(ds)
        fname = sdl.make_filename(S, f'ci_sweep_{state}.h5')
        ds.filepath = fname
        ds.save(fname)
        S.pub.register_file(fname, 'ci', format='h5', plot=False)
        S.log(f"Saved analyzed datafile to {fname}")

    return ds

def take_complex_impedance_ob_sc(S, cfg, bgs, overbias_voltage=19.9,
                                 tes_bias=15.0, overbias_wait=5.0,
                                 cool_wait=30., **ci_kwargs):
    """
    Takes overbiased and superconducting complex impedance sweeps. These are
    required to analyze any in-transition sweeps.

    Args
    -----
    S : SmurfControl
        Pysmurf Instance
    cfg : DetConfig
        Det config instance
    bgs : array, int
        List of bias groups to run on
    overbias_voltage : float
        Voltage to use to overbias detectors
    tes_bias : float
        Voltage to set detectors to after overbiasing
    overbias_wait : float
        Time to wait at the overbias_voltage
    cool_wait : float
        Time to wait at the tes_bias after overbiasing
    **ci_kwargs : 
        Any additional kwargs will be passed directly to the
        ``take_complex_impedance`` function.
    """
    bgs = np.atleast_1d(bgs)

    # Takes SC sweep
    for bg in bgs:
        S.set_tes_bias_bipolar(bg, 0)
    sc = take_complex_impedance(S, cfg, bgs, state='sc', **ci_kwargs)

    S.overbias_tes_all(bias_groups=bgs, overbias_voltage=overbias_voltage,
                       tes_bias=tes_bias, overbias_wait=overbias_wait,
                       cool_wait=cool_wait)
    ob = take_complex_impedance(S, cfg, bgs, state='ob', **ci_kwargs)

    cfg.dev.update_experiment({
        'complex_impedance_sc_path': sc.filepath,
        'complex_impedance_ob_path': ob.filepath
    }, update_file=True)

    return sc, ob
