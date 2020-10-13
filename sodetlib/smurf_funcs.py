import numpy as np
import os
import time
from scipy import signal
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scipy.optimize as opt
import pickle as pkl
pi = np.pi

from pysmurf.client.util.pub import set_action

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


@set_action()
def find_subbands(S, cfg, spur_width=5):
    """
    Do a noise sweep to find the coarse position of resonators. 
    Return active bands and a dictionary of active subbands.
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
        A dictionary of {band:[list of subbands]} for each resonator in MHz.
    """
    bands=np.array([])
    subband_dict = {}

    AMC=S.which_bays()
    bands = []
    if 0 in AMC:
        bands += [0, 1, 2, 3]
    if 1 in AMC:
        bands += [4, 5, 6, 7]
    if not bands:
        print('No active AMC')
        return bands, subband_dict

    for band in bands:
        band_cfg = cfg.dev.bands[band]
        freq, resp = S.full_band_resp(band)
        peaks = S.find_peak(freq, resp, make_plot=True,show_plot=False, band=band)
        fs_ = np.array(peaks*1.0E-6) + S.get_band_center_mhz(band)  

        # Drops channels that are too close to 500 MHz multiple
        fs = [f for f in fs_ if (np.abs((f + 500/2) % 500 - 500/2) > spur_width)]
        bad_fs = list(set(fs_) - set(fs))
        bad_fs = [f for f in fs_ if np.abs((f + 500/2) % 500 - 500/2) <= spur_width]

        if bad_fs:
            cprint(f"Dropping frequencies {bad_fs} because they are too close to a " 
                    "500 MHz interval", style=TermColors.WARNING)

        subbands=sorted(list(set([S.freq_to_subband(band,f)[0] for f in fs])))
        subband_dict[band] = subbands
        
        subband_strings = []
        for i,b in enumerate(subbands):
            subband_strings.append(f"{b} ({fs[i]:.2f}MHz)")

        cprint(f"Subbands detected for band {band}:\n{subband_strings}", 
                style=TermColors.OKBLUE)
        cfg.dev.update_band(band, {'active_subbands': subbands})

    return bands, subband_dict


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
    num_averages = S.config.get('smurf_to_mce')['num_averages']
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

def serial_corr(wave, lag=1):
	"""
	Adapted from SWH squid fitting code. Calculates the correlation
	coefficient normalized between 0 to 1 at a fixed lag.

	Parameters
	----------
	wave : array of float
		time stream to autocorrelate
	lag : int
		number of samples delay between the two timestreams at which you
		calculate the coefficient at.

	Returns
	-------
	corr : float
		Normalized correlation coefficient between 0 & 1.
	"""

	n = len(wave)
	y1 = wave[lag:]
	y2 = wave[:n-lag]
	corr = np.corrcoef(y1, y2)[0, 1]
	return corr

def autocorr(wave):
	"""
	Adapted from SWH squid fitting code. Calculates the autocorrelation.

	Parameters
	----------
	wave : array of float
		time stream to autocorrelate

	Returns
	-------
	lags : tuple of ints
		The lags at which the autocorrelation is calculated
	corr : array of floats
		Array of normalized correlation coefficients for each lag
	"""
	lags = range(len(wave)//2)
	corrs = np.array([serial_corr(wave, lag) for lag in lags])
	return lags, corrs

def sine_fit(x,a0,a1,ph1,f):
	"""
	Sine wave.

	Parameters
	----------
	x : array of float
		points to evaluate the sine function at
	a0 : float
		sine offset from 0
	a1 : float
		sine amplitude
	ph1 : float
		sine phase
	f : float
		sine frequency

	Returns
	-------
	sine : array of float
		Sine wave with given parameters evaluated at all x points.
	"""
	return a0 + a1*np.sin(2*np.pi*f*x + ph1)

def take_tickle(S,band,bias_group, high_current, tickle_voltage, over_bias):
	"""
	Takes a tickle measurement on a particular bias_group at a specified amplitude.

	Parameters
	----------
	band : int
		band to optimize noise on
	bias_group : int
		tes bias [0,11] to apply tickle on
	tickle_voltage : float
		voltage amplitude of tickle
	high_current : bool
		whether or not to take the tickle in high current mode
	over_bias : bool
		whether or not to overbias in high current mode before taking tickle

	Returns
	-------
	data_file : filepath
		Path to tickle data file.
	"""
	#Setting bias in low current mode
	if high_current:
		S.set_tes_bias_high_current(bias_group)
	else:
		S.set_tes_bias_low_current(bias_group)
	if over_bias:
		cur_bias = S.get_tes_bias_bipolar(bias_group)
		S.set_tes_bias_high_current(bias_group)
		S.set_tes_bias_bipolar(bias_group,19)
		time.sleep(15)
		S.set_tes_bias_bipolar(bias_group,cur_bias)
		if not(high_current):
			S.set_tes_bias_low_current(bias_group)
		time.sleep(90)
	#Starting data stream
	data_file = S.stream_data_on()
	#Wait 5 seconds before playing tickle
	time.sleep(5)
	#This is the DAC full scale in bits
	scale = 2**18
	#Multiplier is the fraction of the DAC scale
	multiplier = tickle_voltage/10.
	#Get current DC bias level
	cur_dc = S.get_tes_bias_bipolar(bias_group)
	print(f'Current bias: {S.get_tes_bias_bipolar(bias_group)}')
	#This defines the signal we'll play on the DAC
	sig   = multiplier*scale*np.cos(2*np.pi*np.array(range(2048))/(2048)) + (cur_dc/10.)*scale
	S.play_tes_bipolar_waveform(bias_group,sig)
	#Play sine wave for 15 sec
	time.sleep(15)
	S.set_rtm_arb_waveform_enable(0x0)
	print(f'Current bias: {S.get_tes_bias_bipolar(bias_group)}')
	S.set_tes_bias_bipolar(bias_group,cur_dc)
	print(f'Current bias: {S.get_tes_bias_bipolar(bias_group)}')
	#wait 5 seconds after sine wave stops to stop stream
	time.sleep(5)
	S.stream_data_off()
	return data_file, cur_dc

def analyze_tickle(S, band, data_file, dc_level, tickle_voltage, high_current, channels = None,
					make_channel_plots = False):
	"""
	Analyzes tickle measurement and writes a pickle file and returns a dictionary
	with the channels that show a tickle response and the resistance of those
	channels.

	Parameters
	----------
	band : int
		band to optimize noise on
	data_file : filepath
		path to data to be analyzed
	dc_level : float
		DC voltage bias at which tickle taken at.
	tickle_voltage : float
		voltage amplitude of tickle
	channels : int list
		If None will analyze all channels that are on in the specified band.
	make_channel_plots : bool
		Whether or not to show the individual channel plots with fits, defaults
		to false.

	Returns
	-------
	tick_dict : dictionary
		Dictionary with T/F for each channel whether its a detector or not and
		for detector channels calculated resistance.
	"""
	#Read back in stream data to analyze
	timestamp,phase,mask = S.read_stream_data(datafile=data_file)
	#ctime for plot labeling
	ctime = S.get_timestamp()
	#Convert voltage amplitude of sine wave to the current sent down bl.
	I_command = (tickle_voltage)/S.bias_line_resistance
	if high_current:
		I_command = I_command*S.high_low_current_ratio
	print(f'I commanded = {I_command}')
	if channels == None:
		channels = S.which_on(band)

	#Initialize some output variables
	tick_dict = {}
	det_freqs = []
	det_Rs = []

	for c in channels:
		tick_dict[c] = {}
		ch_idx = mask[band,c]
		pnew = phase[ch_idx]
		#Extract only the range of the phase data where we have the tickle
		pnew = pnew[1500:4250]
		p_mean = np.mean(pnew)
		I_amp_check = (S.pA_per_phi0*(pnew-p_mean)/(2*np.pi))*1e-12
		amp_check = (np.max(I_amp_check)-np.min(I_amp_check))/2
		print(f'Channel {c}, amp_check = {amp_check}')
		tick_dict[c]['detector_chan'] = True
		if 1.1*amp_check > I_command:
			#This checks if you are not tracking properly and you're getting
			#more phase shift than you would get in the superconducting state
			#when all commanded current goes through the TES
			tick_dict[c]['detector_chan'] = False
			plot_style = 'ro'
			if make_channel_plots:
				V_bias = (I_command - amp_check)*S.R_sh
				R_bolo = V_bias/(amp_check)
				plt.figure(1)
				plt.plot(np.arange(0,len(pnew))/S.fs, I_amp_check)
				plt.title(f'Band {band}, Channel {c}, Freq {S.channel_to_freq(band,c)}')
				props = dict(boxstyle = 'round',facecolor = 'wheat', alpha = 0.5)
				txtstr = '\n'.join((
				'Non-detector channel:',
				'Not tracking',
				f'R_pp = {np.round(R_bolo/1e-3,2)}'))
				ax = plt.gca()
				ax.text(0.05, 0.95, txtstr, transform=ax.transAxes, fontsize = 14,
						verticalalignment='top',bbox=props)
				plt.savefig(f'{S.plot_dir}/{ctime}_Tickle_Response_b{band}c{c}.png')
				plt.close()
			#Makes scatter plot to show which channels we identified as detectors
			plt.figure(2)
			plt.semilogx(amp_check/I_command,S.channel_to_freq(band,c),plot_style)
		if amp_check < I_command*(S.R_sh/(S.R_sh + 1000e-3)):
			#This checks if you see no response (i.e. just noise), it rejects
			#all responses whose peak to peak shift of its timestream is less
			#than what would be present w/ a 500 mOhm bolo which is 50 times
			#our bolometer designed normal reistance.
			tick_dict[c]['detector_chan'] = False
			plot_style = 'ro'
			if make_channel_plots:
				V_bias = (I_command - amp_check)*S.R_sh
				R_bolo = V_bias/(amp_check)
				plt.figure(1)
				plt.plot(np.arange(0,len(pnew))/S.fs, I_amp_check)
				plt.title(f'Band {band}, Channel {c}, Freq {S.channel_to_freq(band,c)}')
				props = dict(boxstyle = 'round',facecolor = 'wheat', alpha = 0.5)
				txtstr = '\n'.join((
				'Non-detector channel:',
				'No Response',
				f'R_pp = {np.round(R_bolo/1e-3,2)}'))
				ax = plt.gca()
				ax.text(0.05, 0.95, txtstr, transform=ax.transAxes, fontsize = 14,
						verticalalignment='top',bbox=props)
				plt.savefig(f'{S.plot_dir}/{ctime}_Tickle_Response_b{band}c{c}.png')
				plt.close()
			#Makes scatter plot to show which channels we identified as detectors
			plt.figure(2)
			plt.semilogx(amp_check/I_command,S.channel_to_freq(band,c),plot_style)
		if tick_dict[c]['detector_chan']:
			#This identifies the detector channels to calculate resistance
			#off of.
			plot_style = 'bo'
			det_freqs.append(S.channel_to_freq(band,c))
			#Makes scatter plot to show which channels we identified as detectors
			plt.figure(2)
			plt.semilogx(amp_check/I_command,S.channel_to_freq(band,c),plot_style)

			#Calculate the autocorrelation to find the tickle frequency
			lags,corrs = autocorr(pnew)
			pks,_ = signal.find_peaks(corrs,distance = 100)
			tau0 = np.median(np.diff(np.sort(pks)))/S.fs
			fguess = 1/tau0
			#Fit the tickle to a sine wave
			time_plot = np.arange(0,len(pnew))/S.fs
			popt,pcov = opt.curve_fit(f = sine_fit,xdata = time_plot, ydata = pnew,
								p0 = [np.mean(pnew),np.abs(np.max(pnew)-np.min(pnew))/2,
										0,fguess],
								bounds = ([np.min(pnew),0,-pi,0.95*fguess],
								[np.max(pnew),20*(np.max(pnew)-np.min(pnew)),
								pi,1.05*fguess]))
			#Use the offset of the sine wave to define the 0 phase reference
			ph_0 = popt[0]
			if make_channel_plots:
				#Plot the phase data + fit vs time w/ the fit parameters
				#displayed in a text box on the plot.
				plt.figure(3)
				plt.plot(time_plot,pnew,'bo')
				plt.plot(time_plot,sine_fit(time_plot,popt[0],popt[1],popt[2],popt[3]),'r-')
				plt.title(f'Band {band}, Channel {c}, Freq {S.channel_to_freq(band,c)}')
				txtstr = '\n'.join((
				f'$a_0 = {popt[0]}$',
				f'$a_1 = {popt[1]}$',
				f'$\phi_1= {popt[2]}$',
				f'$f = {popt[3]}$'))
				props = dict(boxstyle = 'round',facecolor = 'wheat', alpha = 0.5)
				ax = plt.gca()
				ax.text(0.05, 0.95, txtstr, transform=ax.transAxes, fontsize = 14,
						verticalalignment='top',bbox=props)
				plt.savefig(f'{S.plot_dir}/{ctime}_Tickle_Response_Fit_b{band}c{c}.png')
				plt.close()
			#Convert phase nparray to current through the SQUID
			I_SQ = (S.pA_per_phi0*(pnew-ph_0)/(2*np.pi))*1e-12
			#Convert the amplitude of the sine fit to current through the SQUID
			I_amp_SQ = (S.pA_per_phi0*(popt[1])/(2*np.pi))*1e-12
			#Calculate the voltage on the bolometer from the current division
			#between the shunt + TES branch
			V_bias = (I_command - I_amp_SQ)*S.R_sh
			#Calculate the resistance from V_bolo/I_bolo
			R_bolo = V_bias/(I_amp_SQ)
			det_Rs.append(R_bolo)
			if make_channel_plots:
				V_bias_pp = (I_command - amp_check)*S.R_sh
				R_bolo_pp = V_bias/(amp_check)
				plt.figure(1)
				plt.plot(time_plot,I_SQ)
				plt.xlabel('Time [s]',fontsize = 16)
				plt.ylabel('$I_{SQ/bolo}$ [A]',fontsize = 16)
				plt.title(f'Band {band}, Channel {c}, Freq {S.channel_to_freq(band,c)}, R = {np.round(R_bolo/1e-3,4)} mOhms')
				props = dict(boxstyle = 'round',facecolor = 'wheat', alpha = 0.5)
				txtstr = '\n'.join((
				'Detector channel:',
				f'R = {np.round(R_bolo/1e-3,4)} mOhms',
				f'R_pp = {np.round(R_bolo_pp/1e-3,2)}',
				f'Voltage Bias = {np.round(dc_level,2)} V'))
				ax = plt.gca()
				ax.text(0.05, 0.95, txtstr, transform=ax.transAxes, fontsize = 14,
						verticalalignment='top',bbox=props)
				plt.savefig(f'{S.plot_dir}/{ctime}_Tickle_Response_b{band}c{c}.png')
				plt.close()


			tick_dict[c]['I_commanded'] = I_command
			tick_dict[c]['I_amp_SQ'] = I_amp_SQ
			tick_dict[c]['V_bias'] = V_bias
			tick_dict[c]['R_bolo'] = R_bolo
	plt.figure(2)
	plt.xlabel('Max-Min Current Through SQUID [A]')
	plt.ylabel('Channel number')
	props = dict(boxstyle = 'round',facecolor = 'wheat', alpha = 0.5)
	txtstr = '\n'.join((
	'Red = Non response channels',
	'Blue = Detectors'))
	ax = plt.gca()
	ax.text(0.05, 0.95, txtstr, transform=ax.transAxes, fontsize = 14,
			verticalalignment='top',bbox=props)
	plt.savefig(f'{S.plot_dir}/{ctime}_identified_detectors_b{band}.png')
	plt.close()

	plt.figure(4)
	plt.semilogy(det_freqs,np.asarray(det_Rs)/1e-3,'ko')
	#plt.ylim(0,400)
	plt.xlabel(f'Detector SMuRF Channel Number for Band {band}')
	plt.ylabel('Resistance from Tickle [mOhms]')
	plt.title(f'Bolo resistance at DC Bias = {np.round(dc_level,2)} V')
	plt.savefig(f'{S.plot_dir}/{ctime}_tickle_resistance_b{band}.png')
	plt.close()


	pkl.dump(tick_dict,open(f'{S.output_dir}/{ctime}_summary_data.pkl','wb'))
	return tick_dict
