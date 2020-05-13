import numpy as np
import os
import time
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


def get_median_noise(S, cfg, band):
    """
    Gets the median noise from the psd.

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
        make_plot=band_cfg.get('make_plot', False), channel=S.which_on(band),
        nsamp=2**18, lms_gain=band_cfg['lms_gain'],
        lms_freq_hz=band_cfg['lms_freq_hz'], feedback_start_frac=1/12,
        feedback_end_frac=0.98
    )
    datafile = S.take_noise_psd(30, nperseg=2**16, save_data=False,
                                make_channel_plot=False)
    median_noise, _ = analyze_noise_psd(S, band, datafile)
    return median_noise



def optimize_uc_att_fix_dr(band,frac_pp,lms_freq,ctime):
    """
    Finds the minimum median white noise level for all channels on in a band vs
    the uc attenuator value and tells you if you actually found a minimum vs uc
    attenuator within your range or not.

    Parameters
    ----------
    band : int
        band to optimize noise on
    frac_pp : float
        fraction full scale of the FR DAC used for tracking_setup
    lms_freq : float
        tracking frequency used for tracking_setup
    ctime : str
        ctime used for saved data/plot titles

    Returns
    -------
    delta_dr : int

    found_min : str
        tells you if the minimum of the noise is within your range, yes if
        true, low if its at uc_atten = 30, and high if its at uc_atten = 0
    median_noise_min : float
        minimum median white noise level at the optimized uc_atten value
    atten_min : int
        attenuator value which minizes the median noise
    """
    attens = np.arange(30, -2, -2) #This is the full range of the attenuators
    median_noise = []

    for atten in attens:
        print('Setting UC Atten to: ',atten)
        S.set_att_uc(band, atten)
        S.run_serial_gradient_descent(band)
        S.run_serial_eta_scan(band)

        #4kHz is the standard reset_rate, this can be changed but not to any arbitrary number. We can choose to expose this value as an argument later or not.
        S.tracking_setup(
            band, reset_rate_khz=4, fraction_full_scale=frac_pp, make_plot=False,
            save_plot=False, show_plot=False, channel=S.which_on(band),
            nsamp=2**18, lms_gain=6, lms_freq_hz = lms_freq,
            feedback_start_frac=1/12, feedback_end_frac=0.98
        )
        dat_file_temp = S.take_noise_psd(30, nperseg=2**16, save_data=False,
                                         make_channel_plot = False)
        mn, od = analyze_noise_psd(band,dat_file_temp,ctime)
        median_noise.append(mn)
       if atten == 30:
            mn_initial = mn
        if mn > 4*mn_initial:
            print('Median noise is now greater than 4 times what it was at UC atten = 30 so exiting loop at uc_atten = '+str(atten))
            attens = np.arange(30,atten-2,-2)
            break
    plt.figure()
    plt.plot(attens,median_noise)
    plt.title(f'Drive = {S.get_amplitude_scale_channel(band, S.which_on(band)[0])} in Band {band}',fontsize = 18)
    plt.xlabel('UC Attenuator Value',fontsize = 14)
    plt.ylabel('Median Channel Noise [pA/rtHz]',fontsize = 14)
    plotname = os.path.join(S.plot_dir, f'{ctime}_noise_vs_uc_atten_b{band}.png')
    plt.savefig(plotname)
    S.pub.register_file(plotname, 'noise_vs_atten', plot=True)
    plt.close()

    median_noise = np.asarray(median_noise)
    med_min_arg = np.argmin(median_noise)
    found_min = 'yes'
    if med_min_arg == 0:
        found_min = 'low'
    if med_min_arg == (len(median_noise) - 1):
        found_min = 'high'
    return found_min, np.min(median_noise), attens[med_min_arg]


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
                            frac_pp=None, lms_freq=None, make_plots=False):
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
    while True:
        print(f"Setting Drive to {drive}")
        ctime = S.get_timestamp()
        S.set_att_uc(band, 30)
        S.relock(band=band, drive=drive)

        medians = []
        initial_median = None
        for atten in attens:
            print(f'Setting UC atten to: {}')
            S.set_att_uc(band, atten)
            m = get_median_noise(S, cfg, band)
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
            plt.plot(attens, medians)
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
        if 0 < min_arg < len(medians)-1:
            found_min = True
            min_median = medians[min_arg]
            min_atten = attens[min_arg]
            min_drive = drive
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
