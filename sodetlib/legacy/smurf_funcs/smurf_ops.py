"""
Module for general smurf operations.
"""
from sodetlib.util import cprint, TermColors, make_filename, \
                          get_tracking_kwargs, Registers

import numpy as np
import os
import time
import sys
import matplotlib
from scipy import signal

from tqdm.auto import tqdm

try:
    import matplotlib.pyplot as plt
except Exception:
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
from pysmurf.client.util.pub import set_action

try:
    from spt3g import core
    from sotodlib.io import load_smurf
except Exception:
    print("Could not import spt3g-software! Update your sodetlib docker")

@set_action()
def take_squid_open_loop(S,cfg,bands,wait_time,Npts,NPhi0s,Nsteps,relock,
                         frac_pp=None,lms_freq=None,reset_rate_khz=None,
                         lms_gain=None,channels=None):
    """
    Adapted from script by SWH: shawn@slac.stanford.edu (can still see original
    in pysmurf/scratch/shawn/measureFluxRampFvsV.py) by MSF: msilvafe@ucsd.edu.

    Takes data in open loop (only slow integral tracking) and steps through flux
    values to trace out a SQUID curve. This can be compared against the tracked
    SQUID curve which might not perfectly replicate this if these curves are
    poorly approximated by a sine wave (or ~3 harmonics of a fourier expansion).

    Parameters
    ----------
    bands: (int list)
        list of bands to take dc SQUID curves on
    wait_time: (float)
        how long you wait between flux step point in seconds
    Npts: (int)
        number of points you take at each flux bias step to average
    Nphi0s: (int)
        number of phi0's or periods of your squid curve you want to take at
        least 3 is recommended and more than 5 just takes longer without much
        benefit.
    Nsteps: (int)
        Number of flux points you will take total.
    relock: (bool)
        Whether or not to relock before starting flux stepping
    channels: (dict)
        default is None and will run on all channels that are on
        otherwise pass a dictionary with a key for each band
        with values equal to the list of channels to run in each band.

    Returns
    -------
    raw_data : (dict)
        This contains the flux bias array, channel array, and frequency
        shift at each bias value for each channel in each band.
    """
    cur_mode = S.get_cryo_card_ac_dc_mode()
    if cur_mode == 'AC':
        S.set_mode_dc()
    ctime = S.get_timestamp()
    fn_raw_data = f'{S.output_dir}/{ctime}_fr_sweep_data.npy'
    print(bands)

    #This calculates the amount of flux ramp amplitude you need for 1 phi0
    #and then sets the range of flux bias to be enough to achieve the Nphi0s
    #specified in the fucnction call.
    band_cfg = cfg.dev.bands[bands[0]]
    if frac_pp is None:
        frac_pp = band_cfg['frac_pp']
    if lms_freq is None:
        lms_freq = band_cfg['lms_freq_hz']
    if reset_rate_khz is None:
        reset_rate_khz = band_cfg['flux_ramp_rate_khz']
    frac_pp_per_phi0 = frac_pp/(lms_freq/(reset_rate_khz*1e3))
    bias_low=-frac_pp_per_phi0*NPhi0s
    bias_high=frac_pp_per_phi0*NPhi0s

    #This is the step size calculated from range and number of steps
    bias_step=np.abs(bias_high-bias_low)/float(Nsteps)
    if channels is None:
        channels = {}
        for band in bands:
            channels[band] = S.which_on(band)

    bias = np.arange(bias_low, bias_high, bias_step)

    # final output data dictionary
    raw_data = {}
    raw_data['bias'] = bias
    bands_with_channels_on=[]
    for band in bands:
        band_cfg = cfg.dev.bands[band]
        if lms_gain is None:
            lms_gain = band_cfg['lms_gain']
        if len(channels[band])>0:
            S.log(f'{len(channels[band])} channels on in band {band}, configuring band for simple, integral tracking')
            S.log(f'-> Setting lmsEnable[1-3] and lmsGain to 0 for band {band}.', S.LOG_USER)
            prev_lms_enable1 = S.get_lms_enable1(band)
            prev_lms_enable2 = S.get_lms_enable2(band)
            prev_lms_enable3 = S.get_lms_enable3(band)
            prev_lms_gain = S.get_lms_gain(band)
            S.set_lms_enable1(band, 0)
            S.set_lms_enable2(band, 0)
            S.set_lms_enable3(band, 0)
            S.set_lms_gain(band, lms_gain)

            raw_data[band]={}

            bands_with_channels_on.append(band)

    bands=bands_with_channels_on
    fs = {}

    sys.stdout.write('\rSetting flux ramp bias to 0 V\033[K before tune'.format(bias_low))
    S.set_fixed_flux_ramp_bias(0.)

    ### begin retune on all bands with tones
    for band in bands:
        fs[band] = []
        S.log('Retuning')
        if relock:
            S.relock(band)
        for i in range(2):
            S.run_serial_gradient_descent(band)
            S.run_serial_eta_scan(band)

        # toggle feedback if functionality exists in this version of pysmurf
        time.sleep(5)
        S.toggle_feedback(band)

        ### end retune

    small_steps_to_starting_bias=None
    if bias_low<0:
        small_steps_to_starting_bias=np.arange(bias_low,0,bias_step)[::-1]
    else:
        small_steps_to_starting_bias=np.arange(0,bias_low,bias_step)

    # step from zero (where we tuned) down to starting bias
    S.log('Slowly shift flux ramp voltage to place where we begin.', S.LOG_USER)
    for b in small_steps_to_starting_bias:
        sys.stdout.write('\rFlux ramp bias at {b} V')
        sys.stdout.flush()
        S.set_fixed_flux_ramp_bias(b,do_config=False)
        time.sleep(wait_time)

    ## make sure we start at bias_low
    sys.stdout.write(f'\rSetting flux ramp bias low at {bias_low} V')
    S.set_fixed_flux_ramp_bias(bias_low,do_config=False)
    time.sleep(wait_time)

    S.log('Starting to take flux ramp.', S.LOG_USER)

    for b in bias:
        sys.stdout.write(f'\rFlux ramp bias at {b} V')
        sys.stdout.flush()
        S.set_fixed_flux_ramp_bias(b,do_config=False)
        time.sleep(wait_time)
        for band in bands:
            fsamp=np.zeros(shape=(Npts,len(channels[band])))
            for i in range(Npts):
                fsamp[i,:]=S.get_loop_filter_output_array(band)[channels[band]]
            fsampmean=np.nanmean(fsamp,axis=0)
            fs[band].append(fsampmean)

    sys.stdout.write('\n')

    S.log('Done taking flux ramp data.', S.LOG_USER)

    for band in bands:
        fres = []
        for ch in channels[band]:
            fres.append(S.channel_to_freq(band,ch))
        fres=[S.channel_to_freq(band, ch) for ch in channels[band]]
        raw_data[band]['fres']=np.array(fres)
        raw_data[band]['channels']=channels[band]

        #stack
        lfovsfr=np.dstack(fs[band])[0]
        raw_data[band]['lfovsfr']=lfovsfr
        raw_data[band]['fvsfr']=np.array([arr+fres for (arr,fres) in zip(lfovsfr,fres)])

    # save dataset for each iteration, just to make sure it gets
    # written to disk
    print(f'Writing SQUID Curve Data to: {fn_raw_data}')
    np.save(fn_raw_data, raw_data)
    S.pub.register_file(fn_raw_data,'dc_squid_curve',format='npy')

    # done - zero and unset
    S.set_fixed_flux_ramp_bias(0,do_config=False)
    S.unset_fixed_flux_ramp_bias()
    S.set_lms_enable1(band, prev_lms_enable1)
    S.set_lms_enable2(band, prev_lms_enable2)
    S.set_lms_enable3(band, prev_lms_enable3)
    S.set_lms_gain(band, lms_gain)
    if cur_mode == 'AC':
        S.set_mode_ac()
    return raw_data


def find_and_tune_freq(S, cfg, bands, new_master_assignment=True,
                       grad_cut=0.01,amp_cut=0.01, show_plot=True,
                       dump_cfg=True):
    """
    Find_freqs to identify resonance, measure eta parameters + setup channels
    using setup_notches, run serial gradient + eta to refine

    Parameters
    ----------
    S: pysmurf.client.SmurfControl
        Pysmurf control instance
    cfg: DetConfig
        Detector config object
    bands : List[int]
        bands to find tuned frequencies on. In range [0,7].
    new_master_assignment : bool, optional
        Whether to create a new master assignment (tuning)
        file. This file defines the mapping between resonator frequency
        and channel number. Default True.
    amp_cut : float
        The fractiona distance from the median value to decide whether there
        is a resonance.
    grad_cut : float
        The value of the gradient of phase to look for. Default is 0.01
    show_plot: bool
        If True will show the find-freq plots
    dump_cfg: bool
        If True, will dump updated dev cfg (with new tunefile) to disk.
    """
    bands = np.atleast_1d(bands)
    num_resonators_on = 0
    for band in bands:
        band_cfg = cfg.dev.bands[band]
        cprint(f"Tuning band {band}...")
        S.find_freq(band, tone_power=band_cfg['drive'],
                    make_plot=True, save_plot=True, show_plot=show_plot,
                    amp_cut=amp_cut, grad_cut=grad_cut)
        if len(S.freq_resp[band]['find_freq']['resonance']) == 0:
            cprint(f'Find freqs could not find resonators in  band {band}',
                   False)
            continue
        S.setup_notches(band, tone_power=band_cfg['drive'],
                        new_master_assignment=new_master_assignment)
        S.run_serial_gradient_descent(band)
        S.run_serial_eta_scan(band)

        num_resonators_on += len(S.which_on(band))

    tune_file = S.tune_file
    if not tune_file:
        cprint("Find freqs was unsuccessful! could not find resonators in the\
                specified bands", False)
        return False
    print(f"Total num resonators on: {num_resonators_on}")
    print(f"Tune file: {tune_file}")

    print("Updating config tunefile...")
    cfg.dev.update_experiment({'tunefile': tune_file})
    if dump_cfg:
        cfg.dev.dump(cfg.dev_file, clobber=True)

    return num_resonators_on, tune_file


@set_action()
def tracking_quality(S, cfg, band, tracking_kwargs=None,
                     make_channel_plots=False, r_thresh=0.9, show_plots=False,
                     nphi0=1):
    """
    Runs tracking setup and returns how good at tracking each channel is

    Args
    -----
        S : SmurfControl
            Pysmurf control object
        cfg : DetConfig
            Detconfig object
        band : int
            band number
        tracking_kwargs : dict
            Dictionary of additional custom args to pass to tracking setup
        r_thresh : float
            Threshold used to set color on plots
        nphi0 : optional(int)
            Number of segments to divide FluxRamp cycle into. Will default to
            1, which will use the entire flux ramp period to stack instead of
            dividing into NPHI0 segments, which is necessary if the FR period
            isn't an integer number of phi0.
    Returns
    --------
        rs : np.ndarray
            Array of size (512) containing values between 0 and 1 which tells
            you how good a channel is at tracking. If close to 1, the channel
            is tracking well and if close to 0 the channel is tracking poorly
        f : np.ndarray
            f as returned from tracking setup
        df : np.ndarray
            df as returned from tracking setup
        sync : np.ndarray
            sync as returned from tracking setup
    """
    band_cfg = cfg.dev.bands[band]
    tk = get_tracking_kwargs(S, cfg, band, kwargs=tracking_kwargs)
    tk['nsamp'] = 2**20  # moreee data
    tk['show_plot'] = False  # Override

    f, df, sync = S.tracking_setup(band, **tk)
    si = S.make_sync_flag(sync)

    if nphi0 is None:
        nphi0 = int(round(tk['lms_freq_hz'] / S.get_flux_ramp_freq()/1000))

    # Average cycles to get single period estimate
    seg_size = (si[1] - si[0]) // nphi0
    nstacks = (len(si) - 1) * nphi0

    active_chans = np.zeros_like(f[0], dtype=bool)
    active_chans[S.which_on(band)] = True

    fstack = np.zeros((seg_size, len(f[0])))
    for i in range(len(si) - 1):
        s = si[i]
        for j in range(nphi0):
            a = s + seg_size * j
            fstack += f[a:a + seg_size, :]
    fstack /= nstacks

    # calculates quality of estimate wrt real data
    y_real = f[si[0]:si[-1], :]
    # Averaged cycle repeated nstack times

    with np.errstate(invalid='ignore'):
        y_est = np.vstack([fstack for _ in range(nstacks)])
        sstot = np.sum((y_real - np.nanmean(y_real, axis=0))**2, axis=0)
        ssres = np.sum((y_real - y_est)**2, axis=0)

        r = 1 - ssres/sstot

    # Probably means it's a bugged debug channels.
    r[np.isnan(r) & active_chans] = 1

    if show_plots:
        plt.ion()
    else:
        matplotlib.use("Agg")
        plt.ioff()

    fname = make_filename(S, f'tracking_quality_b{band}.png', plot=True)
    fig, ax = plt.subplots()
    fig.patch.set_facecolor('white')
    ax.hist(r[active_chans], bins=30, range=(0, 1))
    ax.axvline(r_thresh, linestyle=':', alpha=0.8)
    text_props = {
        'transform': ax.transAxes, 'fontsize': 11, 'verticalalignment': 'top',
        'bbox': {'facecolor': 'white'}
    }
    props = {'facecolor': 'white'}
    num_good = np.sum(r[active_chans] > r_thresh)
    num_active = np.sum(active_chans)
    s = f"{num_good}/{num_active} Channels above r={r_thresh}"
    ax.text(0.05, 0.95, s, **text_props)
    ax.set(xlabel="Tracking Quality", ylabel="Num Channels",
           title=f"Band {band} Tracking Quality")
    plt.savefig(fname)
    S.pub.register_file(fname, 'tracking_goodness', plot=True)
    if not show_plots:
        plt.close()

    if make_channel_plots:
        print("Making channel plots....")
        nramps = 2
        xs = np.arange(len(f))
        m = (si[1] - 20 < xs) & (xs < si[1 + nramps] + 20)
        for chan in np.where(active_chans)[0]:
            fig, ax = plt.subplots()
            fig.patch.set_facecolor('white')
            c = 'C1' if r[chan] > r_thresh else 'black'
            ax.plot(xs[m], f[m, chan], color=c)
            props = {'facecolor': 'white'}
            ax.text(0.05, 0.95, f"r={r[chan]:.3f}", transform=ax.transAxes,
                    fontsize=15, verticalalignment="top", bbox=props)
            ax.set_title(f"Band {band} Channel {chan}")
            fname = make_filename(S, f"tracking_b{band}_c{chan}.png",
                                  plot=True)
            fig.savefig(fname)
            if not show_plots:
                plt.close()

    return r, f, df, sync


def get_session_files(cfg, session_id, idx=None, stream_id=None):
    base_dir = cfg.sys['g3_dir']
    if stream_id is None:
        stream_id = cfg.sys['slots'][f'SLOT[{cfg.slot}]']['stream_id']
    subdir = os.path.join(base_dir, str(session_id)[:5], stream_id)
    files = sorted([
        os.path.join(subdir, f) for f in os.listdir(subdir)
        if str(session_id) in f
    ])

    if idx is None:
        return files
    elif isinstance(idx, int):
        return files[idx]
    else:  # list of indexes
        return [files[i] for i in idx]


def load_session(cfg, session_id, idx=None, stream_id=None, show_pb=False):
    """
    Loads a stream-session into an axis manager.

    Args
    ----
    cfg : DetConfig object
        DetConfig object
    session_id: int
        Session id corresonding with the stream session you wish to load
    idx: int, list(int), optional
    """
    files = get_session_files(cfg, session_id, idx, stream_id=stream_id)
    am = load_smurf.load_file(files, show_pb=show_pb)

    if 'ch_info' not in am._fields:
        am.wrap('ch_info', am.det_info.smurf)

    return am


@set_action()
def take_g3_data(S, dur, **stream_kw):
    """
    Takes data for some duration

    Args
    ----
    S : SmurfControl
        Pysmurf control object
    dur : float
        Duration to take data over

    Returns
    -------
    session_id : int
        Id used to read back stream data
    """
    stream_g3_on(S, **stream_kw)
    time.sleep(dur)
    sid = stream_g3_off(S, emulator=stream_kw.get('emulator', False))
    return sid


@set_action()
def stream_g3_on(S, make_freq_mask=True, emulator=False, tag='',
                 channel_mask=None, filter_wait_time=2):
    """
    Starts the G3 data-stream. Returns the session-id corresponding with the
    data stream.

    Args
    ----
    S : S
        Pysmurf control object
    make_freq_mask : bool, optional
        Tell pysmurf to write and register the current freq mask
    emulator : bool
        If True, will enable the emulator source data-generator. Defaults to
        False.

    Return
    -------
    session_id : int
        Id used to read back streamed data
    """
    reg = Registers(S)

    reg.pysmurf_action.set(S.pub._action)
    reg.pysmurf_action_timestamp.set(S.pub._action_ts)
    reg.stream_tag.set(tag)

    S.stream_data_on(make_freq_mask=make_freq_mask, channel_mask=channel_mask,
                     filter_wait_time=filter_wait_time)

    if emulator:
        reg.source_enable.set(1)
        S.set_stream_enable(1)

    reg.open_g3stream.set(1)

    # Sometimes it takes a bit for data to propogate through to the
    # streamer
    for _ in range(5):
        sess_id = reg.g3_session_id.get()
        if sess_id != 0:
            break
        time.sleep(0.3)
    return sess_id


@set_action()
def stream_g3_off(S, emulator=False):
    """
    Stops the G3 data-stream. Returns the session-id corresponding with the
    data stream.

    Args
    ----
    S : S
        Pysmurf control object
    emulator : bool
        If True, will enable the emulator source data-generator. Defaults to
        False.

    Return
    -------
    session_id : int
        Id used to read back streamed data
    """
    reg = Registers(S)
    sess_id = reg.g3_session_id.get()

    if emulator:
        reg.source_enable.set(0)

    S.set_stream_enable(0)
    S.stream_data_off()

    reg.open_g3stream.set(0)
    reg.pysmurf_action.set('')
    reg.pysmurf_action_timestamp.set(0)
    reg.stream_tag.set('')

    # Waits until file is closed out before returning
    S.log("Waiting for g3 file to close out")
    while reg.g3_session_id.get() != 0:
        time.sleep(0.5)

    return sess_id


@set_action()
def apply_dev_cfg(S, cfg, load_tune=True):
    cfg.dev.apply_to_pysmurf_instance(S, load_tune=load_tune)


def get_wls_from_am(am, nperseg=2**16, fmin=10., fmax=20., pA_per_phi0=9e6):
    """
    Gets white-noise levels for each channel from the axis manager returned
    by smurf_ops.load_session.

    Args
    ----
    am : AxisManager
        Smurf data returned by so.load_session or the G3tSmurf class
    nperseg : int
        nperseg to be passed to welch
    fmin : float
        Min frequency to use for white noise mask
    fmax : float
        Max freq to use for white noise mask
    pA_per_phi0 : float
        S.pA_per_phi0 unit conversion. This will eventually make its way
        into the axis manager, but for now I'm just hardcoding this here
        as a keyword argument until we get there.

    Returns
    --------
    wls : array of floats
        Array of the white-noise level for each channel, indexed by readout-
        channel number
    band_medians : array of floats
        Array of the median white noise level for each band.
    """
    fsamp = 1./np.median(np.diff(am.timestamps))
    fs, pxx = signal.welch(am.signal * pA_per_phi0 / (2*np.pi),
                           fs=fsamp, nperseg=nperseg)
    pxx = np.sqrt(pxx)
    fmask = (fmin < fs) & (fs < fmax)
    wls = np.median(pxx[:, fmask], axis=1)
    band_medians = np.zeros(8)
    for i in range(8):
        m = am.ch_info.band == i
        band_medians[i] = np.median(wls[m])
    return wls, band_medians


def plot_band_noise(am, nbins=40):
    bands = am.ch_info.band
    wls, _ = get_wls_from_am(am)

    fig, axes = plt.subplots(4, 2, figsize=(16, 8),
                             gridspec_kw={'hspace': 0})
    fig.patch.set_facecolor('white')
    bins = np.logspace(1, 4, nbins)
    max_bins = 0

    for b in range(8):
        ax = axes[b % 4, b // 4]
        m = bands == b
        x = ax.hist(wls[m], bins=bins)
        text  = f"Median: {np.median(wls[m]):0.2f}\n"
        text += f"Chans pictured: {np.sum(x[0]):0.0f}"
#         text += f"{}/{} channels < 100"
        ax.text(0.75, .7, text, transform=ax.transAxes)
        ax.axvline(np.median(wls[m]), color='red')
        max_bins = max(np.max(x[0]), max_bins)
        ax.set(xscale='log', ylabel=f'Band {b}')

    axes[0][0].set(title="AMC 0")
    axes[0][1].set(title="AMC 1")
    axes[-1][0].set(xlabel="White Noise (pA/rt(Hz))")
    axes[-1][1].set(xlabel="White Noise (pA/rt(Hz))")
    for _ax in axes:
        for ax in _ax:
            ax.set(ylim=(0, max_bins * 1.1))

    return fig, axes


@set_action()
def loopback_test(S, cfg, bands=None, attens=None, scans_per_band=1):
    """
    Runs loopback tests for smurf. For each of the specified bands loops
    through uc and dc attens taking the full band response at each atten.
    Can plot results for one AMC at a time using the plot_loopback_results
    function.

    Args:
        S (pysmurf.SmurfControl):
            Pysmurf instance
        cfg (DetConfig):
            config object
        bands (int, list):
            List of bands to take loopback over
        attens (list):
            List of attenuations to be looped over. This will be used for both
            UC and DC attens.
        scans_per_band (int):
            Number of band-responses to average over

    Returns:
        out (dict):
            Summary dictionary. Has structure::
                ``out['uc_sweep/dc_sweep'][band][atten]``
    """
    if bands is None:
        bands = S._bands
    else:
        bands = np.atleast_1d(bands)

    if attens is None:
        attens = [0, 1, 2, 4, 8, 16, 31]
    else:
        attens = np.atleast_1d(attens)

    out = {
        'uc_sweep': {b: {} for b in bands},
        'dc_sweep': {b: {} for b in bands},
        'band_center_mhz': {}
    }

    tot = 2*len(bands)*len(attens)
    pb = tqdm(total=tot)
    for b in bands:
        out['band_center_mhz'][b] = S.get_band_center_mhz(b)
        S.set_att_dc(b, 0)
        for att in attens:
            S.set_att_uc(b, att)
            out['uc_sweep'][b][att] = S.full_band_resp(
                band=b, make_plot=False, save_plot=False, show_plot=False,
                save_data=True, n_scan=scans_per_band,
                correct_att=False)
            pb.update()

        S.set_att_uc(b, 0)
        for att in attens:
            S.set_att_dc(b, att)
            out['dc_sweep'][b][att] = S.full_band_resp(
                band=b, make_plot=False, save_plot=False, show_plot=False,
                save_data=True, n_scan=scans_per_band,
                correct_att=False)
            pb.update()
    fname = make_filename(S, 'loopback_test.npy')
    np.save(fname, out, allow_pickle=True)
    S.pub.register_file(fname, 'loopback_test', format='npy')

    return out

def plot_loopback_results(summary, amc, band_width=200e6, S=None):
    """
    Plots loopback results for a single AMC.

    Args:
        summary (dict or str):
            Summary dict returned from loopback_test function
        amc (int):
            AMC to plot.
        band_width (float):
            Width of each band (Hz) to include in the response plot.
        S (SmurfControl, optional):
            If specified, will save plot to the plots directory and register
            with the smurf publisher.
    """
    if isinstance(summary, str):
        summary = np.load(summary, allow_pickle=True).item()

    fig, ax = plt.subplots(2, 2, figsize=(18, 10))
    bands = list(summary['uc_sweep'].keys())

    # Estimate attenuation for each band based on response
    est_attens = {
    'uc_sweep': {b: {} for b in bands},
    'dc_sweep': {b: {} for b in bands},
    }

    # Bandwidth to use to estimate atten
    est_bw = 200
    for sweep in est_attens.keys():
        x = summary[sweep]
        for b in bands:
            for i, (att, v) in enumerate(x[b].items()):
                fs, resp = v
                # Estimate attenuation
                m = np.abs(fs) < est_bw
                power = 20 * np.log10(np.nanmean(np.abs(resp[m])))
                if att==0:
                    p0 = power
                else:
                    est_attens[sweep][b][att] = -2*(power - p0)

    labeled = False
    for b in range(amc * 4, 4*amc + 4):
        if b not in bands:
            continue

        for i, (att, v) in enumerate(summary['uc_sweep'][b].items()):
            fs, resp = v
            m = np.abs(fs) < band_width
            if not labeled:
                label = f'att={att}'
            else:
                label = None

            bc = summary['band_center_mhz'][b] * 1e6
            ax[0][0].plot(fs[m] + bc, np.abs(resp[m]), color=f'C{i}', alpha=0.8, label=label)
            ax[0][0].set(title="UC Sweep Response")

        for i, (att, v) in enumerate(summary['dc_sweep'][b].items()):
            fs, resp = v
            m = np.abs(fs) < band_width
            if not labeled:
                label = f'att={att}'
            else:
                label = None

            bc = summary['band_center_mhz'][b] * 1e6
            ax[1][0].plot(fs[m] + bc, np.abs(resp[m]), color=f'C{i}', alpha=0.8, label=label)
            ax[1][0].set(title="DC Sweep Response")

        labeled = True

        xs = est_attens['uc_sweep'][b].keys()
        ys = est_attens['uc_sweep'][b].values()
        ax[0][1].plot(xs, ys, 'o-', label=f'Band {b}')

        xs = est_attens['dc_sweep'][b].keys()
        ys = est_attens['dc_sweep'][b].values()
        ax[1][1].plot(xs, ys, 'o-', label=f'Band {b}')

    ax[0][1].set(title="Estimated UC Atten")
    ax[1][1].set(title="Estimated DC Atten")

    ax[0][0].set(xlabel="Frequency (Hz)", ylabel="Response")
    ax[1][0].set(xlabel="Frequency (Hz)", ylabel="Response")
    ax[0][1].set(xlabel="Actual atten", ylabel="Estimated atten")
    ax[1][1].set(xlabel="Actual atten", ylabel="Estimated atten")

    for axis in ax.flatten():
        axis.legend(fontsize='small', loc='upper left')

    fig.suptitle(f"AMC {amc}", fontsize=20)
    if S is not None:
        plot_file = make_filename(S, f"amc_{amc}_loopback.png")
        fig.savefig(plot_file)
        S.pub.register_file(plot_file, 'loopback', plot=True)

    return fig, ax
