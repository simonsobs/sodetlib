"""
Module for general smurf operations.
"""
from sodetlib.util import cprint, TermColors, make_filename, \
                          get_tracking_kwargs
import numpy as np
import os
import time
import sys

import matplotlib
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
    return raw_data

@set_action()
def find_subbands(S, cfg, spur_width=5):
    """
    Do a noise sweep to find the coarse position of resonators.
    Return active bands and a dictionary of active subbands.

    Parameters
    ----------
    S : pysmurf.client.SmurfControl
        Smurf control object
    cfg : DetConfig
        sodetlib config object
    spur_width: float
        Will throw out any resonators which are within ``spur_width`` MHz
        from a multiple of 500 MHz to avoid picking up spurs.

    Returns
    -------
    bands : int array
        Active bands
    subband_dict : dict
        A dictionary containing the list of subbands in each band.
    """
    subband_dict = {}
    bands = []

    amc = S.which_bays()
    if 0 in amc:
        bands += [0, 1, 2, 3]
    if 1 in amc:
        bands += [4, 5, 6, 7]
    if not bands:
        print('No active AMC')
        return bands, subband_dict

    for band in bands:
        freq, resp = S.full_band_resp(band)
        peaks = S.find_peak(freq, resp, make_plot=True,show_plot=False, band=band)
        fs_ = np.array(peaks*1.0E-6) + S.get_band_center_mhz(band)

        # Drops channels that are too close to 500 MHz multiple
        fs = [f for f in fs_
              if (np.abs((f + 500/2) % 500 - 500/2) > spur_width)]
        bad_fs = list(set(fs_) - set(fs))
        bad_fs = [f for f in fs_
                  if np.abs((f + 500/2) % 500 - 500/2) <= spur_width]

        if bad_fs:
            cprint(f"Dropping frequencies {bad_fs} because they are too close "
                   "to a 500 MHz interval", style=TermColors.WARNING)

        subbands=sorted(list({S.freq_to_subband(band,f)[0] for f in fs}))
        subband_dict[band] = subbands

        subband_strings = []
        for i,b in enumerate(subbands):
            subband_strings.append(f"{b} ({fs[i]:.2f}MHz)")

        cprint(f"Subbands detected for band {band}:\n{subband_strings}",
                style=TermColors.OKBLUE)
        cfg.dev.update_band(band, {'active_subbands': subbands})

    return bands, subband_dict


def find_and_tune_freq(S, cfg, bands, new_master_assignment=True, amp_cut=0.1):
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
    """
    num_resonators_on = 0
    default_subbands = np.arange(13, 115)
    for band in bands:
        band_cfg = cfg.dev.bands[band]
        subband = band_cfg.get('active_subbands', default_subbands)
        if subband is True:
            subband = default_subbands
        elif not subband:
            continue
        S.find_freq(band, tone_power=band_cfg['drive'],
                    make_plot=True,
                    save_plot=True,
                    subband=subband, amp_cut=amp_cut)
        if len(S.freq_resp[band]['find_freq']['resonance']) == 0:
            cprint(f'Find freqs could not find resonators in '
            f'band : {band} and subbands : {subband}', False)
            continue
        S.setup_notches(band, tone_power=band_cfg['drive'],
                    new_master_assignment=new_master_assignment)
        S.run_serial_gradient_descent(band)
        S.run_serial_eta_scan(band)

        num_resonators_on += len(S.which_on(band))

    tune_file = S.tune_file
    if not tune_file:
        cprint("Find freqs was unsuccessful! could not find resonators in the\
                specified bands + subbands", False)
        return False
    print(f"Total num resonators on: {num_resonators_on}")
    print(f"Tune file: {tune_file}")

    print("Updating config tunefile...")
    cfg.dev.update_experiment({'tunefile': tune_file})

    return num_resonators_on, tune_file

def res_shift(S, bands):
    """
    Calculates the resonance frequency from serial gradient descent before and 
    after setup_notches is run. Typicaly paired w/ uc_att or flux steps.

    Parameters
    ----------
    S: pysmurf.client.SmurfControl
        Pysmurf control instance
    bands : List[int]
        bands to perform operation on.
    """
    out_dict = {}
    for band in bands:
        for band in bands:
            out_dict[band] = {}
            #For all other steps run serial algs after changing flux bias but before 
            #running setup notches to see how much eta and freq shift
            print(f'Running serial algorithms on band {band}')
            S.run_serial_gradient_descent(band)
            S.run_serial_eta_scan(band)
            out_dict[band]['fs_sg_b'] = S.channel_to_freq(band)
            out_dict[band]['eta_mags_sg_b'] = S.get_eta_mag_array(band)
            out_dict[band]['eta_ps_sg_b'] = S.get_eta_phase_array(band)
            #Now run setup notches and get the new freqs and etas
            print(f'Running setup_notches on band {band}')
            S.setup_notches(band,new_master_assignment = True)
            out_dict[band]['fs_sn'] = S.channel_to_freq(band)
            out_dict[band]['eta_mags_sn'] = S.get_eta_mag_array(band)
            out_dict[band]['eta_ps_sn'] = S.get_eta_phase_array(band)
            #Run serial algs after and get the new freqs and etas
            print(f'Running serial algorithms on band {band}')
            S.run_serial_gradient_descent(band)
            S.run_serial_eta_scan(band)
            out_dict[band]['fs_sg_a'] = S.channel_to_freq(band)
            out_dict[band]['eta_mags_sg_a'] = S.get_eta_mag_array(band)
            out_dict[band]['eta_ps_sg_a'] = S.get_eta_phase_array(band)
            out_dict[band]['channels'] = S.which_on(band)
        #For each flux step write out a tunefile that contains both band freq_resp info
        #this can be used for fitting.
        out_dict['tunefile'] = S.tune_file
        return out_dict

def res_shift_vs_uc_att(S, uc_atts, bands, tunefile):
    """
    Calculates the resonance frequency from serial gradient descent before and 
    after setup_notches is run over a range of uc attenuator steps.

    Parameters
    ----------
    S: pysmurf.client.SmurfControl
        Pysmurf control instance
    bands : List[int]
        bands to perform operation on.
    uc_atts : list[int]
        uc attenuator values to step over.
    tunefile : str
        tunefile to use for retuning at each step.
    """
    #Initialize output dictionary
    out_dict = {}
    initial_uc_att = {}
    #Step over uc attenuator values
    for band in bands:
        initial_uc_att[band] = S.get_att_uc(band)
        S.set_att_uc(band,uc_atts[0])
        #For the first step load the tunefile and relock then run serial
        #algs to get freq and eta before setup_notches
        S.load_tune(tunefile)
        S.relock(band)
    for uc_att in uc_atts:
        #Loop over bands since serial algs and setup notches are per band operations
        for band in bands:
            print(f'Setting uc att in band {band} to {uc_att}')
            S.set_att_uc(band,uc_att)
        out_dict[uc_att] = res_shift(S, bands)
    for band in bands:
        S.set_att_uc(band,initial_uc_att[band])
    return out_dict

def res_shift_vs_flux_bias(S, frac_pp_steps, bands, tunefile):
    """
    Calculates the resonance frequency from serial gradient descent before and 
    after setup_notches is run over a range of squid flux bias steps.

    Parameters
    ----------
    S: pysmurf.client.SmurfControl
        Pysmurf control instance
    bands : List[int]
        bands to perform operation on.
    frac_pp_steps : list[float]
        list of flux bias steps to take in units of fraction full scale
        of the flux ramp dac.
    tunefile : str
        tunefile to use for retuning at each step.
    """
    S.set_mode_dc()
    #Initialize output dictionary
    out_dict = {}
    #Step over dc flux bias values
    S.set_fixed_flux_ramp_bias(frac_pp_steps[0],do_config=False)
    for band in bands:
        #For the first step load the tunefile and relock then run serial
        #algs to get freq and eta before setup_notches
        S.load_tune(tunefile)
        S.relock(band)
    for frac_pp in frac_pp_steps:
        print(f'Setting flux bias to {frac_pp} fraction full scale')
        S.set_fixed_flux_ramp_bias(frac_pp,do_config=False)
        out_dict[frac_pp] = res_shift(S, bands)
    #Set flux bias back to 0
    S.set_fixed_flux_ramp_bias(0,do_config = False)
    return out_dict


@set_action()
def tracking_quality(S, cfg, band, tracking_kwargs=None,
                     make_channel_plots=False, r_thresh=0.9, show_plots=False):
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
    nphi0 = int(round(tk['lms_freq_hz'] / S.get_flux_ramp_freq()/1000))

    active_chans = np.zeros_like(f[0], dtype=bool)
    active_chans[S.which_on(band)] = True

    # Average cycles to get single period estimate
    seg_size = (si[1] - si[0]) // nphi0
    fstack = np.zeros((seg_size, len(f[0])))
    nstacks = (len(si)-1) * nphi0
    for i in range(len(si) - 1):
        s = si[i]
        for j in range(nphi0):
            a = s + seg_size * j
            fstack += f[a:a + seg_size, :]
    fstack /= nstacks

    # calculates quality of estimate wrt real data
    y_real = f[si[0]:si[-1], :]
    # Averaged cycle repeated nstack times
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


def get_session_files(cfg, session_id, idx=None):
    base_dir = cfg.sys['g3_dir']
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
    else:  #list of indexes
        return [files[i] for i in idx]


def load_session(cfg, session_id, idx=None):
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
    files = get_session_files(cfg, session_id, idx)
    return load_smurf.load_file(files)


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
def stream_g3_on(S, make_freq_mask=True, emulator=False, tag=''):
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

    S.stream_data_on(make_freq_mask=make_freq_mask)

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


def apply_dev_cfg(S, cfg, load_tune=True):
    """
    Applies basic device config params (amplifier biases, attens, tone_powers)
    to a pysmurf instance based on the device cfg values. Note that this does
    not set any of the tracking-related params since this shouldn't replace
    tracking_setup.
    """
    S.set_amplifier_bias(
        bias_hemt=cfg.dev.exp['amp_hemt_Vg'],
        bias_50k=cfg.dev.exp['amp_50k_Vg']
    )

    if load_tune:
        S.load_tune(cfg.dev.exp['tunefile'])

    for b in S._bands:
        band_cfg = cfg.dev.bands[b]
        if 'uc_att' in band_cfg:
            S.set_att_uc(b, band_cfg['uc_att'])
        if 'dc_att' in band_cfg:
            S.set_att_dc(b, band_cfg['dc_att'])
        if 'drive' in band_cfg:
            # Sets the tone power of all enabled channels to `drive`
            amp_scales = S.get_amplitude_scale_array(b)
            amp_scales[amp_scales != 0] = band_cfg['drive']
            S.set_amplitude_scale_array(b, amp_scales)
