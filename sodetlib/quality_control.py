import sodetlib as sdl
import numpy as np
import time
from scipy import signal
from tqdm.auto import trange
from pysmurf.client.base.smurf_control import SmurfControl
import matplotlib.pyplot as plt

def check_packet_loss(Ss, cfgs, dur=10, fr_khz=4, nchans=2000, slots=None):
    """
    Takes a short G3 Stream on multiple slots simultaneously and checks for
    dropped samples. This function is strange since it requires simultaneous
    streaming on multiple slots to properly test, so it doesn't follow the
    standard sodetlib function / data format.

    Args
    -----
    Ss : dict[SmurfController]1
        Dict of pysmurf instances where the key is the slot-number
    cfgs : dict[DetConfig]
        Dict of DetConfigs where the key is the slot number
    dur : float
        Duration of data stream (sec)
    fr_khz : float
        Frequency of FR rate (khz)
    nchans : int
        Number of channels to stream
    slots : list
        Which slots to stream data on. If None, will stream on all slots in the
        Ss object.

    Returns
    --------
    ams : dict[AxisManagers]
        Dict of axis managers indexed by slot-number
    res : dict
        Dict where the key is the slot number, and the values are dicts
        containing frame counters, number of dropped frames, etc.
    """
    if slots is None:
        slots = Ss.keys()

    for s in slots:
        S = Ss[s]
        S.flux_ramp_setup(fr_khz, 0.4, band=0)
        sdl.stream_g3_on(
            S, channel_mask=np.arange(nchans), downsample_factor=1
        )

    time.sleep(dur)
    
    sids = {}
    for s in slots:
        sids[s] = sdl.stream_g3_off(Ss[s])

    ams = {}
    for s, sid in sids.items():
        ams[s] = sdl.load_session(cfgs[s].stream_id, sid)

    res = {}
    for s, am in ams.items():
        dropped_samps = np.sum(np.diff(am.primary['FrameCounter']) - 1)
        total_samps = len(am.primary['FrameCounter'])
        res[s] = {
            'sid': sids[s],
            'meta': sdl.get_metadata(Ss[s], cfgs[s]),
            'frame_counter': am.primary['FrameCounter'],
            'dropped_samples': dropped_samps,
            'dropped_frac': dropped_samps / total_samps,
        }

    return ams, res

@sdl.set_action()
def measure_bias_line_resistances(
    S: SmurfControl, cfg, vstep=0.001, bgs=None, sleep_time=2.0):
    """
    Function to measure the bias line resistance and high-low-current-ratio for
    each bias group. This needs to be run with the smurf hooked up to the
    cryostat and the detectors superconducting.

    Args
    -------
    S : SmurfControl
        Pysmurf instance
    cfg : DetConfig
        Det Config instance
    vstep : float
        Voltage step size (in low-current-mode volts)
    bgs : list
        Bias lines to measure. Will default to active bias lines
    sleep_time : float
        Time to wait at each step.
    """
    if bgs is None:
        bgs = cfg.dev.exp['active_bgs']
    bgs = np.atleast_1d(bgs)

    vbias = S.get_tes_bias_bipolar_array()
    vb_low = vbias.copy()
    vb_low[bgs] = 0
    vb_high = vbias.copy()
    vb_high[bgs] = vstep
    segs = []

    S.set_tes_bias_bipolar_array(vb_low)
    sdl.set_current_mode(S, bgs, 0, const_current=False)

    def take_step(bias_arr, sleep_time, wait_time=0.2):
        S.set_tes_bias_bipolar_array(bias_arr)
        time.sleep(wait_time)
        t0 = time.time()
        time.sleep(sleep_time)
        t1 = time.time()
        return (t0, t1)

    sdl.stream_g3_on(S)
    time.sleep(0.5)

    segs.append(take_step(vb_low, sleep_time, wait_time=0.5))
    segs.append(take_step(vb_high, sleep_time, wait_time=0.5))

    S.set_tes_bias_bipolar_array(vb_low)
    time.sleep(0.5)
    sdl.set_current_mode(S, bgs, 1, const_current=False)

    segs.append(take_step(vb_low, sleep_time, wait_time=0.05))
    segs.append(take_step(vb_high, sleep_time, wait_time=0.05))

    sid = sdl.stream_g3_off(S)

    am = sdl.load_session(cfg.stream_id, sid)
    ts = am.timestamps
    sigs = []
    for (t0, t1) in segs:
        m = (t0 < ts) & (ts < t1)
        sigs.append(np.mean(am.signal[:, m], axis=1) * S.pA_per_phi0 / (2*np.pi))

    Rbl_low = vstep / (np.abs(sigs[1] - sigs[0]) * 1e-12)
    Rbl_high = vstep / (np.abs(sigs[3] - sigs[2]) * 1e-12)
    high_low_ratio = Rbl_low / Rbl_high

    cfg.dev.exp['bias_line_resistance'] = np.nanmedian(Rbl_low)
    cfg.dev.exp['high_low_current_ratio'] = np.nanmedian(high_low_ratio)
    cfg.dev.update_file()

    path = sdl.make_filename(S, 'measure_bias_line_info')
    data = {
        'Rbl_low_all': Rbl_low,
        'Rbl_high_all': Rbl_high,
        'high_low_ratio_all': high_low_ratio,
        'bias_line_resistance': np.nanmedian(Rbl_low),
        'high_current_mode_resistance': np.nanmedian(Rbl_high),
        'high_low_ratio': np.nanmedian(high_low_ratio),
        'sid': sid,
        'vstep': vstep,
        'bgs': bgs,
        'meta': sdl.get_metadata(S, cfg),
        'segs': segs,
        'sigs': sigs,
        'path': path,
    }
    np.save(path, data, allow_pickle=True)
    S.pub.register_file(path, 'bias_line_resistances', format='npy')

    return am, data

def setup_fixed_tones(S, cfg, tones_per_band=256, bands=None, jitter=0.5,
                      tone_power=None):
    """
    Enables many fixed tones across a selection of bands.

    Args
    ----------
    S : SmurfControl
        Pysmurf instance
    cfg : DetConfig
        Det config instance
    tones_per_band : int
        Number of fixed tones to create in each band
    bands : int, list[int]
        Bands to set fixed tones in. Defaults to all 8.
    jitter : float
        Noise [Mhz] to add to the center freq so that fixed tones are not
        equispaced.
    tone_power : int
        Tone power of fixed tones.
    """
    if bands is None:
        bands = np.arange(8)
    bands = np.atleast_1d(bands)

    chans_per_band = S.get_number_channels()
    for band in bands:
        S.log(f"Setting fixed tones for band {band}")
        if tone_power is None:
            tone_power = cfg.dev.bands[band]['tone_power']
        
        sbs = np.linspace(0, chans_per_band, tones_per_band, dtype=int, endpoint=False)
        asa = np.zeros_like(S.get_amplitude_scale_array(band))
        asa[sbs] = tone_power
        S.set_amplitude_scale_array(band, asa)
        S.set_center_frequency_array(band, np.random.uniform(-jitter/2, jitter/2, chans_per_band))
        S.set_feedback_enable_array(band, np.zeros(chans_per_band, dtype=int))

def get_noise_dBcHz(S, band, chan, nsamp=2**20, nperseg=2**16,
                    noise_freq=30e3, noise_bw=100):
    """
    Takes debug data and measures I/Q noise in dBc/Hz.

    Args
    -----
    S : SmurfControl
        Pysmurf instance
    band : int
        Smurf band
    chan : int
        Smurf chan
    nsamp : int
        Number of samples to take
    nperseg : int
        Nperseg to use when creating the psd
    noise_freq : float
        Freq to measure the readout noise at.
    noise_bw : float
        Frequency bandwidth over which the noise median will be taken

    Returns
    ---------
    noise_i : float
        Noise of the I data stream in dBc/Hz
    noise_q : float
        Noise of the Q data stream in dBc/Hz
    """
    fsamp = S.get_channel_frequency_mhz() * 1e6
    
    sig_i, sig_q, _ = S.take_debug_data(band, channel=chan, rf_iq=True, nsamp=nsamp)
    datfile = S.get_streamdatawriter_datafile()

    fs, pxxi = signal.welch(sig_i, fs=fsamp, nperseg=nperseg)
    fs, pxxq = signal.welch(sig_q, fs=fsamp, nperseg=nperseg)

    magfac = np.mean(sig_q)**2 + np.mean(sig_i)**2
    pxxi_dbc = 10. * np.log10(pxxi/magfac)
    pxxq_dbc = 10. * np.log10(pxxq/magfac)

    f0, f1 = noise_freq - noise_bw/2, noise_freq + noise_bw/2
    m = (f0 < fs) & (fs < f1)
    noise_i = np.nanmedian(pxxi_dbc[m])
    noise_q = np.nanmedian(pxxq_dbc[m])

    return noise_i, noise_q, datfile
    

def plot_fixed_tone_loopback(res):
    """Plot results from fixed_tone_loopback"""
    fig, ax = plt.subplots()
    fig.patch.set_facecolor('white')

    ax.plot(res['freqs'], res['noise_q'], '.', alpha=0.8)
    ax.plot(res['freqs'], res['noise_i'], '.', alpha=0.8)
    txt = '\n'.join([
        f"num tones: {len(res['ft_freqs_all'])}",
        f"tone power: {res['tone_power']}",
        f"UC att: {res['att_uc']}",
        f"DC att: {res['att_dc']}",
    ])
    ax.text(0.05, 0.8, txt, transform=ax.transAxes,
            bbox=dict(fc='white', alpha=0.6))
    ax.set_xlabel("Freq [MHz]")
    ax.set_ylabel("Noise [dBc/Hz]")
    crate_id = res['meta']['crate_id']
    slot = res['meta']['slot']
    ax.set_title(f"Crate {crate_id}, Slot {slot}")

    return fig, ax


def fixed_tone_loopback(
        S, cfg, bands=None, tones_per_band=256, meas_chans_per_band=5,
        setup_tones=False, tone_power=12, show_pb=True, noise_freq=30e3,
        noise_bw=100, att_uc=None, att_dc=None):
    """
    Runs QC test to check noise levels across band with many fixed tones enabled.
    
    Args
    ------
    S : SmurfControl
        Pysmurf instance
    cfg : DetConfig
        Det config instance
    bands : int, list[int]
        Bands to run on. Default is all 8
    tones_per_band : int
        Number of fixed tones to enable per band. This defaults to 256, which
        is 2048 tones total with all 8 bands enabled.
    meas_chans_per_band : int
        Number of channels per band to measure readout noise
    setup_fixed_tones : bool
        If true, will set up fixed tones across specified bands. If false,
        this will skip the setup and assume tones are already set up.
    tone_power : int
        Tone power to use
    show_pb : bool
        If True will show a progress bar
    noise_freq : float
        Target frequency to measure the readout noise
    noise_bw : float
        Frequency bandwidth over which the noise median will be taken
    att_uc : int
        UC atten. If not set, will use current uc atten.
    att_dc : int
        DC atten. If not set, will use current dc atten
    """
    if bands is None:
        bands = np.arange(8)
    bands = np.atleast_1d(bands)

    if att_uc is None:
        att_uc = S.get_att_uc(bands[0])
    else:
        for b in bands:
            S.set_att_uc(b, att_uc)

    if att_dc is None:
        att_dc = S.get_att_dc(bands[0])
    else:
        for b in bands:
            S.set_att_dc(b, att_dc)

    if setup_tones:
        setup_fixed_tones(S, cfg, tones_per_band=tones_per_band, bands=bands,
                          tone_power=tone_power)
    else: # Just update the tone power
        S.log(f"Setting tone power to {tone_power} for bands {bands}...")
        for band in bands:
            asa = S.get_amplitude_scale_array(band)
            if tone_power in np.unique(asa):
                continue
            asa[asa != 0] = tone_power
            S.set_amplitude_scale_array(band, asa)
    
    meas_bands = []
    meas_chans = []
    meas_freqs = []
    ft_chans_all = []
    ft_freqs_all = []
    ft_bands_all = []
    S.log("Finding fixed tones and meas_channels")
    for band in bands:
        freqs = S.get_center_frequency_array(band) \
                 + S.get_tone_frequency_offset_mhz(band) \
                 + S.get_band_center_mhz(band)

        ft_chans = np.where(S.get_amplitude_scale_array(band))[0]
        freqs = freqs[ft_chans]
        sort_idx = np.argsort(freqs)
        meas_idx = np.unique(np.round(np.linspace(
            0, len(ft_chans) - 1, meas_chans_per_band)).astype(int))
        # We want to sort first so meas_chans are evenly distributed across
        # freq space
        meas_chans.append(ft_chans[sort_idx][meas_idx])
        meas_bands.append([band for _ in meas_idx])
        meas_freqs.append(freqs[sort_idx][meas_idx])
        ft_chans_all.append(ft_chans)
        ft_bands_all.append([band for _ in ft_chans])
        ft_freqs_all.append(freqs)
    meas_bands = np.hstack(meas_bands)
    meas_chans = np.hstack(meas_chans)
    meas_freqs = np.hstack(meas_freqs)
    ft_chans_all = np.hstack(ft_chans_all)
    ft_bands_all = np.hstack(ft_freqs_all)
    ft_freqs_all = np.hstack(ft_freqs_all)

    noise_i = np.full_like(meas_freqs, np.nan)
    noise_q = np.full_like(meas_freqs, np.nan)
    datfiles = []
    for i in trange(len(meas_bands), disable=not(show_pb)):
        b, c = meas_bands[i], meas_chans[i]
        S.log(f"Band {b}, Chan {c}")
        try:
            noise_i[i], noise_q[i], _ = get_noise_dBcHz(
                S, b, c, noise_freq=noise_freq,
                noise_bw=noise_bw)
        except IndexError as e:
            S.log(f"Take Data failed...\n{e}")
            S.log("Skipping channel")
        datfiles.append(S.get_streamdatawriter_datafile())

    fname = sdl.make_filename(S, 'fixed_tone_loopback.npy')
    res = dict(
        meta=sdl.get_metadata(S, cfg),
        bands=meas_bands, channels=meas_chans, freqs=meas_freqs,
        noise_i=noise_i, noise_q=noise_q,
        ft_bands_all=ft_bands_all, ft_channels_all=ft_chans_all,
        ft_freqs_all=ft_freqs_all, noise_freq=noise_freq,
        att_uc=att_uc, att_dc=att_dc, tone_power=tone_power,
        datfiles=datfiles
    )
    np.save(fname, res, allow_pickle=True)
    S.pub.register_file(fname, 'loopback', format='npy')


    fname = sdl.make_filename(S, 'fixed_tone_loopback.png', plot=True)
    fig, _ = plot_fixed_tone_loopback(res)
    fig.savefig(fname)
    S.pub.register_file(fname, 'loopback', format='png', plot=True)

    return res

