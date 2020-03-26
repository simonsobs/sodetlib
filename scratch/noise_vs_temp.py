import matplotlib
matplotlib.use('Agg')

import pysmurf
import argparse
import numpy as np
import os
import time

# Enumerate the channels that track poorly that we must kill after each time we run find_freq + setup_notches
bad_track_chans = [15, 23, 28, 29, 59, 63, 87, 91, 127, 151, 183, 187, 267, 287, 295, 303, 315,319, 335, 423]

#Initialize the frac_pp that corresponds to 5 Phi_0's for our mux chip
frac_pp = 0.14139048298824738

def get_power(dr, att):
    return -19 - 3*dr - 0.5*att

def get_median_noise(S, band, dr, att, run_find_freq=False,bad_track_channels=[]):
    plot_dir = S.plot_dir

    if run_find_freq: 
        S.find_freq(band, drive_power=dr, make_plot=True, show_plot=False, save_plot=True)
        S.setup_notches(band, drive=dr, new_master_assignment=False)
        S.plot_tune_summary(band, eta_scan=True)

        for chans in bad_track_channels:
            S.channel_off(band, chans)

    S.run_serial_gradient_descent(band)
    S.run_serial_eta_scan(band)

    # Rerun tracking_setup to make sure things are still tracking after power/temp steps and to track pp flux ramp shift as a function of power
    S.tracking_setup(band, 
        reset_rate_khz=4, lms_freq_hz=20000, fraction_full_scale=frac_pp, make_plot=True, 
        show_plot=False, channel=S.which_on(band), nsamp=2**18, feedback_start_frac=0.02,
        feedback_end_frac = 0.94
    )

    # Take the noise data
    low_freq = 1
    high_freq = 5
    datafile = S.take_noise_psd(
        band=band, meas_time=60, low_freq=[low_freq], high_freq=[high_freq], nperseg=2**16, save_data=True,
        show_plot=False, make_channel_plot=True
    ) 

    noise_per_chan = []
    
    timestamp, phase, mask = S.read_stream_data_gcp_save(datafile)
    phase *= S.pA_per_phi0/(2.*np.pi)
    num_averages = S.config.get('smurf_to_mce')['num_averages']
    fs = S.get_flux_ramp_freq()*1.0E3/num_averages
    nperseg = 2**16
    detrend = 'constant'
    for chan in S.which_on(band):
        if chan < 0:
            continue
        ch_idx = mask[band,chan]
        f, Pxx = signal.welch(phase[ch_idx], nperseg=nperseg,fs=fs, detrend=detrend)
        Pxx = np.sqrt(Pxx)
        popt, pcov, f_fit, Pxx_fit = S.analyze_psd(f, Pxx)
        wl,n,f_knee = popt
        noise_per_chan.append(wl)
    median_noise = np.median(np.asarray(noise_per_chan))
    
    return median_noise


if __name__=='__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--setup', action='store_true')
    parser.add_argument('--config-file', required=True)
    parser.add_argument('--epics-root', default='smurf_server_s2')

    parser.add_argument('--band', type=int, default=2)

    parser.add_argument('--bad-channels',type=int,nargs="*")
    
    args = parser.parse_args()
    
    S = pysmurf.SmurfControl(
            epics_root = args.epics_root,
            cfg_file = args.config_file,
            setup = args.setup,make_logfile=False
    )

    print("Plots in director: ",S.plot_dir)

    #Initialize the band for our mux chip
    band = args.band

    #Set the uc attenuator range to loop over at drive = 12
    uc_att_list = np.arange(10,32,2)

    #Set the drive levels to loop over at uc att = 30
    drive_list = np.arange(11,4,-1)

    #Initialize Power and Median Noise lists to load data into for plotting later
    powers = []
    median_noise = []

    #Loop over uc attenuators for drive = 12
    run_find_freq = True
    dr = 12
    for i, att in enumerate(uc_att_list):
        S.set_att_uc(band,att)

        median_noise.append(get_median_noise(S, band, dr, att, run_find_freq=run_find_freq,bad_track_channels=args.bad_channels))
        powers.append(get_power(dr, at))

        run_find_freq = False

    # Loop over drive array at fixed attenuator
    att = 30 
    starting_asa = S.get_amplitude_scale_array(band)
    asa_mask = (starting_asa != 0).astype(int)
    for i,dr in enumerate(drive_list):
        S.set_amplitude_scale_array(band, dr*asa_mask)

        median_noise.append(get_median_noise(S, band, dr, att))
        powers.append(get_power(dr, at))

    #Save Power + Median Noise data in text file
    powers = np.asarray(powers)
    median_noise = np.asarray(median_noise)

    # Data should probably be timestamped so we can distinguish between runs
    save_name = '{}_noise_vs_power_data.txt'.format(S.get_timestamp())
    outfn = os.path.join(S.plot_dir, save_name)
    np.savetxt(outfn, np.c_[powers, median_noise])

    # Registers file so its picked up by pysmurf archiver
    S.pub.register_file(outfn, 'noise_vs_power', format='txt')
