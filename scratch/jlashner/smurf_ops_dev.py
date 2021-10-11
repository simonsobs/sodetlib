"""
Module for general smurf operations.
"""
from sodetlib.util import cprint, TermColors, make_filename, \
                          get_tracking_kwargs

import sodetlib.smurf_funcs.optimize_params as op
import numpy as np
import os
import time
import sys
import matplotlib
from scipy import signal

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
            fsampmean=np.mean(fsamp,axis=0)
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
    if cur_mod == 'AC':
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
            # Sometimes the last sync segment is a bit shorter which breaks 
            # numpy broadcasting, so we just skip it if that happens
            if len(f[a:a+seg_size, 0]) != seg_size:
                continue
            fstack += f[a:a + seg_size, :]
    fstack /= nstacks

    # calculates quality of estimate wrt real data
    y_real = f[si[0]:si[-1], :]
    # Averaged cycle repeated nstack times

    with np.errstate(invalid='ignore'):
        y_est = np.vstack([fstack for _ in range(nstacks)])
        sstot = np.sum((y_real - np.mean(y_real, axis=0))**2, axis=0)
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


def get_stream_session_id(S):
    """
    Gets the current G3 stream session-id
    """
    reg = S.smurf_processor + "SOStream:SOFileWriter:session_id"
    return S._caget(reg)


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
    return load_smurf.load_file(files, show_pb=show_pb)


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
    so_root = S.epics_root + ":AMCc:SmurfProcessor:SOStream:"
    reg_em_enable = S.epics_root + ":AMCc:StreamDataSource:SourceEnable"
    reg_action = so_root + "pysmurf_action"
    reg_action_ts = so_root + "pysmurf_action_timestamp"
    reg_stream_tag = so_root + "stream_tag"

    S._caput(reg_action, S.pub._action)
    S._caput(reg_action_ts, S.pub._action_ts)
    S._caput(reg_stream_tag, tag)

    S.stream_data_on(make_freq_mask=make_freq_mask, channel_mask=channel_mask,
                     filter_wait_time=filter_wait_time)

    if emulator:
        S._caput(reg_em_enable, 1)
        S.set_stream_enable(1)

    # Sometimes it takes a bit for data to propogate through to the
    # streamer
    for _ in range(5):
        sess_id = get_stream_session_id(S)
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
    sess_id = get_stream_session_id(S)

    so_root = S.epics_root + ":AMCc:SmurfProcessor:SOStream:"
    reg_em_enable = S.epics_root + ":AMCc:StreamDataSource:SourceEnable"
    reg_action = so_root + "pysmurf_action"
    reg_action_ts = so_root + "pysmurf_action_timestamp"
    reg_stream_tag = so_root + "stream_tag"

    if emulator:
        S._caput(reg_em_enable, 0)

    S.set_stream_enable(0)
    S.stream_data_off()

    S._caput(reg_action, '')
    S._caput(reg_action_ts, 0)
    S._caput(reg_stream_tag, '')

    # Waits until file is closed out before returning
    S.log("Waiting for g3 file to close out")
    while get_stream_session_id(S) != 0:
        time.sleep(0.5)

    return sess_id


@set_action()
def apply_dev_cfg(S, cfg, load_tune=True):
    cfg.dev.apply_to_pysmurf_instance(S, load_tune=load_tune)

@set_action()
def cryo_amp_check(S, cfg):
    """
    Performs a system health check. This includes checking/adjusting amplifier
    biases, checking timing, checking the jesd connection, and checking that
    noise can be seen through the system.

    Parameters
    ----------
    S : pysmurf.client.SmurfControl
        Smurf control object
    cfg : DetConfig
        sodetlib config object

    Returns
    -------
    success: bool
        Returns true if all of the following checks were successful:
            - hemt and 50K are able to be biased
            - Id is in range for hemt and 50K
            - jesd_tx and jesd_rx connections are working on specified bays
            - response check for band 0
    """
    amp_hemt_Id = cfg.dev.exp['amp_hemt_Id']
    amp_50K_Id = cfg.dev.exp['amp_50k_Id']

    bays = S.which_bays()
    bay0 = 0 in bays
    bay1 = 1 in bays

    # Turns on both amplifiers and checks biasing.

    cprint("Checking biases", TermColors.HEADER)
    S.C.write_ps_en(11)
    amp_biases = S.get_amplifier_biases()
    biased_hemt = np.abs(amp_biases['hemt_Id']) > 0.2
    biased_50K = np.abs(amp_biases['50K_Id']) > 0.2
    if not biased_hemt:
        cprint("hemt amplifier could not be biased. Check for loose cable",
               False)
    if not biased_50K:
        cprint("50K amplifier could not be biased. Check for loose cable",
               False)

    # Optimize bias voltages
    if biased_hemt and biased_50K:
        cprint("Scanning hemt bias voltage", TermColors.HEADER)
        Id_hemt_in_range = op.optimize_bias(S, amp_hemt_Id, -1.2, -0.6, 'hemt')
        cprint("Scanning 50K bias voltage", TermColors.HEADER)
        Id_50K_in_range = op.optimize_bias(S, amp_50K_Id, -0.8, -0.3, '50K')
        time.sleep(0.2)
        amp_biases = S.get_amplifier_biases()
        Vg_hemt, Vg_50K = amp_biases['hemt_Vg'], amp_biases['50K_Vg']
        print(f"Final hemt current = {amp_biases['hemt_Id']}")
        print(f"Desired hemt current = {amp_hemt_Id}")
        cprint(f"hemt current within range of desired value: "
                            f" {Id_hemt_in_range}",Id_hemt_in_range)
        print(f"Final hemt gate voltage is {amp_biases['hemt_Vg']}")

        print(f"Final 50K current = {amp_biases['50K_Id']}")
        print(f"Desired 50K current = {amp_50K_Id}")
        cprint(f"50K current within range of desired value:"
                            f"{Id_50K_in_range}", Id_50K_in_range)
        print(f"Final 50K gate voltage is {amp_biases['50K_Vg']}")
    else:
        cprint("Both amplifiers could not be biased... skipping bias voltage "
               "scan", False)
        Id_hemt_in_range = False
        Id_50K_in_range = False

    # Check timing is active.
    # Waiting for smurf timing card to be defined
    # Ask if there is a way to add 122.8 MHz external clock check

    # Check JESD connection on bay 0 and bay 1
    # Return connections for both bays, or passes if bays not active
    cprint("Checking JESD Connections", TermColors.HEADER)
    if bay0:
        jesd_tx0, jesd_rx0, status = S.check_jesd(0)
        if jesd_tx0:
            cprint(f"bay 0 jesd_tx connection working", True)
        else:
            cprint(f"bay 0 jesd_tx connection NOT working. "
                    "Rest of script may not function", False)
        if jesd_rx0:
            cprint(f"bay 0 jesd_rx connection working", True)
        else:
            cprint(f"bay 0 jesd_rx connection NOT working. "
                    "Rest of script may not function", False)
    else:
        jesd_tx0, jesd_rx0 = False, False
        print("Bay 0 not enabled. Skipping connection check")

    if bay1:
        jesd_tx1, jesd_rx1, status = S.check_jesd(1)
        if jesd_tx1:
            cprint(f"bay 1 jesd_tx connection working", True)
        else:
            cprint(f"bay 1 jesd_tx connection NOT working. Rest of script may "
                   "not function", False)
        if jesd_rx1:
            cprint(f"bay 1 jesd_rx connection working", True)
        else:
            cprint(f"bay 1 jesd_rx connection NOT working. Rest of script may "
                    "not function", False)
    else:
        jesd_tx1, jesd_rx1 = False, False
        print("Bay 1 not enabled. Skipping connection check")

    # Full band response. This is a binary test to determine that things are
    # plugged in.  Typical in-band noise values are around ~2-7, so here check
    # that average value of noise through band 0 is above 1.  

    # Check limit makes sense when through system
    cprint("Checking full-band response for band 0", TermColors.HEADER)
    band_cfg = cfg.dev.bands[0]
    S.set_att_uc(0, band_cfg['uc_att'])

    freq, response = S.full_band_resp(band=0)
    # Get the response in-band
    resp_inband = []
    band_width = 500e6  # Each band is 500 MHz wide
    for f, r in zip(freq, np.abs(response)):
        if -band_width/2 < f < band_width/2:
            resp_inband.append(r)
    # If the mean is > 1, say response received
    if np.mean(resp_inband) > 1: #LESS THAN CHANGE
        resp_check = True
        cprint("Full band response check passed", True)
    else:
        resp_check = False
        cprint("Full band response check failed - maybe something isn't "
               "plugged in?", False)

    # Check if ADC is clipping. Probably should be a different script, after
    # characterizing system to know what peak data amplitude to simulate
    # Should result in ADC_clipping = T/F
    # Iterate through lowest to highest band, stop when no clipping.
    # Find max value of output of S.read_adc_data(0), compare to pre-set threshold
    # Probably should have a 'good' 'warning', and 'failed' output
    # Above functions are 'startup_check", this is a seperate function

    cfg.dev.update_experiment({
        'amp_hemt_Vg': Vg_hemt,
        'amp_50k_Vg': Vg_50K,
    })

    cprint("Health check finished! Final status", TermColors.HEADER)
    cprint(f" - Hemt biased: \t{biased_hemt}", biased_hemt)
    cprint(f" - Hemt Id in range: \t{Id_hemt_in_range}", Id_hemt_in_range)
    print(f" - Hemt (Id, Vg): \t{(amp_biases['hemt_Id'], amp_biases['hemt_Vg'])}\n")
    cprint(f" - 50K biased: \t\t{biased_50K}", biased_50K)
    cprint(f" - 50K Id in range: \t{Id_50K_in_range}", Id_50K_in_range)
    print(f" - 50K (Id, Vg): \t{(amp_biases['50K_Id'], amp_biases['50K_Vg'])}\n")
    cprint(f" - Response check: \t{resp_check}", resp_check)

    if bay0:
        cprint(f" - JESD[0] TX, RX: \t{(jesd_tx0, jesd_rx0)}",
               jesd_tx0 and jesd_rx0)
    if bay1:
        cprint(f" - JESD[1] TX, RX: \t{(jesd_tx1, jesd_rx1)}",
               jesd_tx1 and jesd_rx1)

    status_bools = [biased_hemt, biased_50K, Id_hemt_in_range, Id_50K_in_range,
                    resp_check]
    if bay0:
        status_bools.extend([jesd_tx0, jesd_rx0])
    if bay1:
        status_bools.extend([jesd_tx1, jesd_rx1])

    return all(status_bools)


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
