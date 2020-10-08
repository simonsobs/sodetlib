import numpy as np
import os
import time
from scipy import signal
import scipy.optimize as opt
from scipy import interpolate
import pickle as pkl

try:
    import matplotlib.pyplot as plt
except Exception:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

class TermColors:
    HEADER = '\n\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def cprint(msg, style=TermColors.OKBLUE):
    if style == True:
        style = TermColors.OKGREEN
    elif style == False:
        style = TermColors.FAIL
    print(style + str(msg) + TermColors.ENDC)

def lowpass_fit(x,scale,cutoff):
    fs = 4e3
    b,a = signal.butter(1,2*cutoff/fs)
    w,h = signal.freqz(b,a)
    x_fit = (fs*0.5/np.pi)*w
    y_fit = scale*abs(h)
    splrep = interpolate.splrep(x_fit,y_fit,s=0)
    return interpolate.splev(x,splrep,der=0)

def optimize_lms_gain(S, cfg, band, BW_target,tunefile=None,
                        reset_rate_khz=None, frac_pp=None,
                        lms_freq=None, meas_time=None,
                        make_plot = True):
    """
    Finds the drive power and uc attenuator value that minimizes the median noise within a band.

    Parameters
    ----------
    band: (int)
        band to optimize noise on
    tunefile: (str)
        filepath to the tunefile for the band to be optimized
    reset_rate: (float)
        flux ramp reset rate in kHz used for tracking setup
    frac_pp: (float)
        fraction full scale of the FR DAC used for tracking_setup
    lms_freq: (float)
        tracking frequency used for tracking_setup
    make_plot: (bool)
        If true will make plots

    Returns
    -------

    """
    #Set the timestamp for plots and data outputs
    ctime = S.get_timestamp()

    #Initialize params from device config
    band_cfg = cfg.dev.bands[band]
    if tunefile is 'devcfg':
        tunefile = cfg.dev.exp['tunefile']
        S.load_tune(tunefile)
    if tunefile is None:
        S.load_tune()
    if frac_pp is None:
        frac_pp = band_cfg['frac_pp']
    if lms_freq is None:
        lms_freq = band_cfg['lms_freq_hz']
    if reset_rate_khz is None:
        reset_rate_khz = band_cfg['flux_ramp_rate_khz']
    
    #Retune and setup tracking on band to optimize
    S.relock(band)
    for i in range(2):
        S.run_serial_gradient_descent(band)
        S.run_serial_eta_scan(band)

    f,df,sync = S.tracking_setup(band,
                    reset_rate_khz=reset_rate_khz,
                    lms_freq_hz = lms_freq,
                    fraction_full_scale = frac_pp,
                    make_plot=False, show_plot=False,
                    channel = S.which_on(band), nsamp = 2**18,
                    feedback_start_frac = 0.02,
                    feedback_end_frac = 0.94,lms_gain=7,
                    return_data = True
                    )
    
    #Identify channel that has flux ramp modulation depth between
    #40-120kHz and has the minimum noise and frequency tracking
    #error. We will use this channel to optimize the lms_gain.
    df_std = np.std(df, 0)
    f_span = np.max(f,0) - np.min(f,0)
    chans_to_consider = np.where((f_span>.04) & (f_span<0.12))[0]
    datfile = S.take_stream_data(10)
    _, outdict = analyze_noise_psd(S,band,datfile)
    chan_noise = []
    chan_df = []
    for ch in list(chans_to_consider):
        chan_noise.append(outdict[ch]['white noise'])
        chan_df.append(df_std[ch])
    best_chan = chans_to_consider[np.argmin(
                np.asarray(chan_df)*np.asarray(chan_noise))]
    print(f'Channel chosen for lms_gain optimization: {best_chan}')
    
    #Store old downsample factor and filter parameters to reset at end
    prev_ds_factor = S.get_downsample_factor()
    prev_filt_param = S.get_filter_disable()
    #Turn off downsampling + filtering to measure tracking BW
    S.set_downsample_factor(1)
    S.set_filter_disable(1)

    #Setup parameters for PSDs
    nperseg = 2**12
    fs = S.get_flux_ramp_freq()*1e3
    detrend = 'constant'

    #Initialize output parameters
    outdict = {}
    f3dBs = []

    #Sweep over lms gain from 8 to 2 and calculate f3dB at each.
    lms_gain_sweep = [8,7,6,5,4,3,2]
    if make_plot == True:
        fig, (ax1,ax2) = plt.subplots(1,2,figsize = (20,10))
        fig.suptitle('$f_{3dB}$ vs lms_gain',fontsize = 32)
        alpha = 0.8
    for lms_gain in lms_gain_sweep:
        outdict[lms_gain] = {}
        S.set_lms_gain(band,lms_gain)
        #for i in range(2):
        #    S.run_serial_gradient_descent(band)
        #    S.run_serial_eta_scan(band)
        S.tracking_setup(band,
                    reset_rate_khz=reset_rate_khz,
                    lms_freq_hz = lms_freq,
                    fraction_full_scale = frac_pp,
                    make_plot=True, show_plot=False,
                    channel = best_chan, nsamp = 2**18,
                    feedback_start_frac = 0.02,
                    feedback_end_frac = 0.94,
                    lms_gain=lms_gain,
                    )
        datafile = S.take_stream_data(20)
        timestamp,phase,mask = S.read_stream_data(datafile)
        phase *= S.pA_per_phi0/(2*np.pi)
        ch_idx = mask[band,best_chan]
        f,Pxx = signal.welch(phase[ch_idx],detrend = detrend,
                            nperseg = nperseg, fs=fs)
        Pxx = np.sqrt(Pxx)
        outdict[lms_gain]['f']=f
        outdict[lms_gain]['Pxx']=Pxx
        pars,covs = opt.curve_fit(lowpass_fit,f,Pxx,bounds = ([0,0],[1e3,1999]))
        outdict[lms_gain]['fit_params'] = pars
        f3dBs.append(pars[1])
        if make_plot == True:
            ax1.loglog(f,Pxx,alpha = alpha,label = f'lms_gain: {lms_gain}')
            ax1.loglog(f,lowpass_fit(f,pars[0],pars[1]),'--',label = f'fit f3dB = {np.round(pars[1],2)} Hz')
            alpha = alpha*0.9
        print(f'lms_gain = {lms_gain}: f_3dB fit = {pars[1]} Hz')
    
    #Identify the lms_gain that produces a f3dB closest to the target
    f3dBs = np.asarray(f3dBs)
    idx_min = np.argmin(np.abs(f3dBs - BW_target))
    if f3dBs[idx_min] < BW_target:
        opt_lms_gain = lms_gain_sweep[idx_min-1]
    else:
        opt_lms_gain = lms_gain_sweep[idx_min]
    print(f'Optimum lms_gain is: {opt_lms_gain}')

    #Save plots and data and register them with the ocs publisher
    if make_plot == True:
        ax1.set_ylim([10,100])
        ax1.legend(fontsize = 14)
        ax1.set_xlabel('Frequency [Hz]',fontsize = 18)
        ax1.set_ylabel('PSD',fontsize = 18)
        ax2.plot(lms_gain_sweep,f3dBs,'o--',label = 'Data')
        ax2.set_xlabel('lms_gain',fontsize = 18)
        ax2.set_ylabel('$f_{3dB}$ [Hz]',fontsize = 18)
        ax2.axvline(opt_lms_gain,ls = '--',color = 'r',label = 'Optimized LMS Gain')
        ax2.axhline(BW_target,ls = '--',color = 'k',label = 'Target $f_{3dB}$')
        ax2.legend(fontsize = 14)
        plotpath = f'{S.plot_dir}/{ctime}_f3dB_vs_lms_gain_b{band}.png'
        plt.savefig(plotpath)
        plt.show()
        S.pub.register_file(plotpath,'opt_lms_gain',plot=True)
    datpath = f'{S.output_dir}/{ctime}_f3dB_vs_lms_gain_b{band}.pkl'
    pkl.dump(outdict, open(datpath,'wb'))
    S.pub.register_file(datpath,'opt_lms_gain_data',format='pkl')
    
    #Reset the downsampling filtering parameters
    S.set_downsample_factor(prev_ds_factor)
    S.set_filter_disable(prev_filt_param)
    return opt_lms_gain, outdict

def optimize_bias(S, target_Id, vg_min, vg_max, amp_name, max_iter=30):
    """
    Scans through bias voltage for hemt or 50K amplifier to get the correct
    gate voltage for a target current.

    Parameters
    -----------
    S (pysmurf.client.SmurfControl):
        PySmurf control object
    target_Id (float):
        Target amplifier current
    vg_min (float):
        Minimum allowable gate voltage
    vg_max (float):
        Maximum allowable gate voltage
    amp_name (str):
        Name of amplifier. Must be either "hemt" or "50K'.
    max_iter (int):
        Maximum number of iterations to find voltage. Defaults to 30.

    Returns
    --------
    success (bool):
        Returns a boolean signaling whether voltage scan has been successful.
        The set voltages can be read with S.get_amplifier_biases().
    """
    if amp_name not in ['hemt', '50K']:
        raise ValueError(cprint(f"amp_name must be either 'hemt' or '50K'", False))

    for _ in range(max_iter):
        amp_biases = S.get_amplifier_biases(write_log=True)
        Vg = amp_biases[f"{amp_name}_Vg"]
        Id = amp_biases[f"{amp_name}_Id"]
        delta = target_Id - Id
        # Id should be within 0.5 from target without going over.
        if 0 <= delta < 0.5:
            return True

        if amp_name=='hemt':
            step = np.sign(delta) * (0.1 if np.abs(delta) > 1.5 else 0.01)
        else:
            step = np.sign(delta) * (0.01 if np.abs(delta) > 1.5 else 0.001)

        Vg_next = Vg + step
        if not (vg_min < Vg_next < vg_max):
            cprint(f"Vg adjustment would go out of range ({vg_min}, {vg_max}). "
                         f"Unable to change {amp_name}_Id to desired value", False)
            return False

        if amp_name == 'hemt':
            S.set_hemt_gate_voltage(Vg_next)
        else:
            S.set_50k_amp_gate_voltage(Vg_next)
        time.sleep(0.2)
    cprint(f"Max allowed Vg iterations ({max_iter}) has been reached. "
           f"Unable to get target Id for {amp_name}.", False)
    return False


def health_check(S, cfg, bay0, bay1):
    """
    Performs a system health check. This includes checking/adjusting amplifier biases,
    checking timing, checking the jesd connection, and checking that noise can
    be seen through the system.

    Parameters
    ----------
    bay0 : bool
        Whether or not bay 0 is active
    bay1 : bool
        Whether or not bay 1 is active

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

    # Turns on both amplifiers and checks biasing.

    cprint("Checking biases", TermColors.HEADER)
    S.C.write_ps_en(11)
    amp_biases = S.get_amplifier_biases()
    biased_hemt = np.abs(amp_biases['hemt_Id']) > 0.2
    biased_50K = np.abs(amp_biases['50K_Id']) > 0.2
    if not biased_hemt:
        cprint("hemt amplifier could not be biased. Check for loose cable", False)
    if not biased_50K:
        cprint("50K amplifier could not be biased. Check for loose cable", False)

    # Optimize bias voltages
    if biased_hemt and biased_50K:
        cprint("Scanning hemt bias voltage", TermColors.HEADER)
        Id_hemt_in_range = optimize_bias(S, amp_hemt_Id, -1.2, -0.6, 'hemt')
        cprint("Scanning 50K bias voltage", TermColors.HEADER)
        Id_50K_in_range = optimize_bias(S, amp_50K_Id, -0.8, -0.3, '50K')
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
        cprint("Both amplifiers could not be biased... skipping bias voltage scan", False)
        Id_hemt_in_range = False
        Id_50K_in_range = False

    # Check timing is active.
    # Waiting for smurf timing card to be defined
    # Ask if there is a way to add 122.8 MHz external clock check

    # Check JESD connection on bay 0 and bay 1
    # Return connections for both bays, or passes if bays not active
    cprint("Checking JESD Connections", TermColors.HEADER)
    if bay0:
        jesd_tx0, jesd_rx0 = S.check_jesd(0)
        if jesd_tx0:
            cprint(f"bay 0 jesd_tx connection working", True)
        else:
            cprint(f"bay 0 jesd_tx connection NOT working. Rest of script may not function", False)
        if jesd_rx0:
            cprint(f"bay 0 jesd_rx connection working", True)
        else:
            cprint(f"bay 0 jesd_rx connection NOT working. Rest of script may not function", False)
    if not bay0:
        jesd_tx0, jesd_rx0 = False, False
        print("Bay 0 not enabled. Skipping connection check")

    if bay1:
        jesd_tx1, jesd_rx1 = S.check_jesd(1)
        if jesd_tx1:
            cprint(f"bay 1 jesd_tx connection working", True)
        else:
            cprint(f"bay 1 jesd_tx connection NOT working. Rest of script may not function", False)
        if jesd_rx1:
            cprint(f"bay 1 jesd_rx connection working", True)
        else:
            cprint(f"bay 1 jesd_rx connection NOT working. Rest of script may not function", False)
    if not bay1:
        jesd_tx1, jesd_rx1 = False, False
        print("Bay 1 not enabled. Skipping connection check")

    # Full band response. This is a binary test to determine that things are plugged in.
    # Typical in-band noise values are around ~2-7, so here check that average value of
    # noise through band 0 is above 1.
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
        cprint("Full band response check failed - maybe something isn't plugged in?", False)

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
        cprint(f" - JESD[0] TX, RX: \t{(jesd_tx0, jesd_rx0)}", jesd_tx0 and jesd_rx0)
    if bay1:
        cprint(f" - JESD[1] TX, RX: \t{(jesd_tx1, jesd_rx1)}", jesd_tx1 and jesd_rx1)

    status_bools = [biased_hemt, biased_50K, Id_hemt_in_range, Id_50K_in_range, resp_check]
    if bay0:
        status_bools.extend([jesd_tx0, jesd_rx0])
    if bay1:
        status_bools.extend([jesd_tx1, jesd_rx1])

    return all(status_bools)


def find_and_tune_freq(S, cfg, bands, new_master_assignment=True):
    """
    Find_freqs to identify resonance, measure eta parameters + setup channels
    using setup_notches, run serial gradient + eta to refine
    Parameters
    ----------
    S:  (pysmurf.client.SmurfControl)
        Pysmurf control instance
    cfg: (DetConfig)
        Detector config object
    bands : [int]
        bands to find tuned frequencies on. In range [0,7].

    Optional parameters
    ----------
    new_master_assignment : bool
        Whether to create a new master assignment (tuning)
        file. This file defines the mapping between resonator frequency
        and channel number. Default True.

    Optional parameters from cfg file
    ----------
    drive : int
        The drive amplitude.  If none given, takes from cfg.
    make_plot : bool
        make the plot frequency sweep. Default False.
    save_plot : bool
        save the plot. Default True.
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
        S.find_freq(band, drive_power=band_cfg['drive'],
                    make_plot=band_cfg['make_plot'],
                    save_plot=band_cfg['save_plot'],
                    subband=subband)
        if len(S.freq_resp[band]['find_freq']['resonance']) == 0:
            cprint(f'Find freqs could not find resonators in '
            f'band : {band} and subbands : {subband}', False)
            continue
        S.setup_notches(band, drive=band_cfg['drive'],
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


def get_median_noise(S, cfg, band, meas_time=30):
    """
    Takes PSD and returns the median noise of all active channels.

    Parameters
    ------------
    S:  (pysmurf.client.SmurfControl)
        Pysmurf control instance
    cfg: (DetConfig)
        Detector config object
    band : (int)
        band to get median noise for.

    Returns
    ---------
    median_noise: (float)
        Median noise for the specified band.
    """
    band_cfg = cfg.dev.bands[band]
    S.run_serial_gradient_descent(band)
    S.run_serial_eta_scan(band)
    S.tracking_setup(
        band, reset_rate_khz=4, fraction_full_scale=band_cfg['frac_pp'],
        make_plot=band_cfg.get('make_plot', False),
        save_plot=band_cfg.get('save_plot', False),
        channel=S.which_on(band), nsamp=2**18, lms_gain=band_cfg['lms_gain'],
        lms_freq_hz=band_cfg['lms_freq_hz'], feedback_start_frac=1/12,
        feedback_end_frac=0.98, show_plot=False
    )
    datafile = S.take_stream_data(meas_time)
    median_noise, _ = analyze_noise_psd(S, band, datafile)
    return median_noise


def analyze_noise_psd(S, band, dat_file):
    """
    Finds the white noise level, 1/f knee, and 1/f polynomial exponent of a noise timestream and returns the median white noise level of all channels and a dictionary of fitted values per channel.

    Parameters
    ----------
    band : int
        band to optimize noise on
    dat_file : str
        filepath to timestream data to analyze
    ctime : str
        ctime used for saved data/plot titles

    Returns
    -------
    median_noise : float
        median white noise level of all channels analyzed in pA/rtHz

    outdict : dict of{int:dict of{str:float}}
        dictionary with each key a channel number and each channel number another dictionary containing the fitted 1/f knee, 1/f exponent, and white noise level in pA/rtHz
    """

    outdict = {}
    datafile = dat_file
    nperseg = 2**16
    detrend = 'constant'
    timestamp, phase, mask = S.read_stream_data(datafile)
    phase *= S.pA_per_phi0/(2.*np.pi)
    num_averages = S.get_downsample_factor()
    fs = S.get_flux_ramp_freq()*1.0E3/num_averages
    wls = []
    for chan in S.which_on(band):
        if chan < 0:
            continue
        ch_idx = mask[band, chan]
        f, Pxx = signal.welch(phase[ch_idx], nperseg=nperseg,fs=fs, detrend=detrend)
        Pxx = np.sqrt(Pxx)
        popt, pcov, f_fit, Pxx_fit = S.analyze_psd(f, Pxx)
        wl,n,f_knee = popt
        wls.append(wl)
        outdict[chan] = {}
        outdict[chan]['fknee']=f_knee
        outdict[chan]['noise index']=n
        outdict[chan]['white noise']=wl
    median_noise = np.median(np.asarray(wls))
    return median_noise, outdict


def optimize_power_per_band(S, cfg, band, tunefile=None, dr_start=None,
                            frac_pp=None, lms_freq=None, make_plots=True,
                            meas_time=None, fixed_drive=False):
    """
    Finds the drive power and uc attenuator value that minimizes the median noise within a band.

    Parameters
    ----------
    band: (int)
        band to optimize noise on
    tunefile: (str)
        filepath to the tunefile for the band to be optimized
    dr_start: (int)
        drive power to start all channels with, default is 12
    frac_pp: (float)
        fraction full scale of the FR DAC used for tracking_setup
    lms_freq: (float)
        tracking frequency used for tracking_setup
    make_plots: (bool)
        If true, will make median noise plots

    Returns
    -------
    min_med_noise : float
        The median noise at the optimized drive power
    atten : int
        Optimized uc attenuator value
    cur_dr : int
        Optimized dr value
    meas_time : float
        Measurement time for noise PSD in seconds.
    fixed_drive: bool
        If true, will not try to vary drive to search for global minimum.
    """
    band_cfg = cfg.dev.bands[band]
    if tunefile is None:
        tunefile = cfg.dev.exp['tunefile']
    if dr_start is None:
        dr_start = band_cfg['drive']
    if frac_pp is None:
        frac_pp = band_cfg['frac_pp']
    if lms_freq is None:
        lms_freq = band_cfg['lms_freq_hz']

    S.load_tune(tunefile)
    drive = dr_start
    attens = np.arange(30, -2, -2)
    checked_drives = []
    found_min = False
    while not found_min:
        cprint(f"Setting Drive to {drive}")
        ctime = S.get_timestamp()
        S.set_att_uc(band, 30)
        S.relock(band=band, drive=drive)

        medians = []
        initial_median = None
        for atten in attens:
            cprint(f'Setting UC atten to: {atten}')
            S.set_att_uc(band, atten)

            kwargs = {}
            if meas_time is not None:
                kwargs['meas_time'] = meas_time
            m = get_median_noise(S, cfg, band, **kwargs)
            medians.append(m)
            # Checks to make sure noise doesn't go too far over original median
            if initial_median is None:
                initial_median = m
            if m > 4 * initial_median:
                cprint(f"Median noise is now 4 times what it was at atten=30, "
                       f"so exiting loop at uc_atten = {atten}", style=False)
                break

        if make_plots:
            plt.figure()
            plt.plot(attens[:len(medians)], medians)
            plt.title(f'Drive = {drive} in Band {band}', fontsize=18)
            plt.xlabel('UC Attenuator Value', fontsize=14)
            plt.ylabel('Median Channel Noise [pA/rtHz]', fontsize=14)
            plotname = os.path.join(S.plot_dir,
                                    f'{ctime}_noise_vs_uc_atten_b{band}.png')
            plt.savefig(plotname)
            S.pub.register_file(plotname, 'noise_vs_atten', plot=True)
            plt.close()

        medians= np.asarray(medians)
        min_arg = np.argmin(medians)
        checked_drives.append(drive)
        if (0 < min_arg < len(medians)-1) or fixed_drive:
            found_min = True
            min_median = medians[min_arg]
            min_atten = attens[min_arg]
            min_drive = drive
            if not (0 < min_arg < len(medians) - 1):
               cprint("Minimum is on the boundary! May not be a global minimum!",
                      style=TermColors.WARNING)
        else:
            drive += 1 if min_arg == 0 else -1
            if drive in checked_drives:
                cprint(f"Drive {drive} has already been checked!! "
                       f"Exiting loop unsuccessfully", False)
                found_min = False
                break

    if found_min:
        cprint(f'found optimum dr = {drive}, and optimum uc_att = {min_atten}',
               style=True)
        S.set_att_uc(band, min_atten)
        S.load_tune(tunefile)
        S.relock(band=band,drive=drive)
        S.run_serial_gradient_descent(band)
        S.run_serial_eta_scan(band)

        cfg.dev.update_band(band, {
            'uc_att': min_atten, 'drive': drive
        })
        return min_median, min_atten, drive
