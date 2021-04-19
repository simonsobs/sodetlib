#!/usr/bin/env python3
import argparse
import numpy as np
import os
import sys
sys.path.append("../../../python")
import pysmurf.client
import time


def make_html(data_path):
    """
    Makes the HTML page.

    Args:
    -----
    data_path : str
        The full path to the data output file
    """
    import shutil
    import fileinput
    import datetime
    import glob

    # Make output directories
    script_path = os.path.dirname(os.path.realpath(__file__))
    template_path = os.path.join(script_path, 'page_template')
    html_path = os.path.join(data_path, "summary")

    print(f'Making HTML output in : {html_path}')

    # Copy template directory
    print(f"Copying {template_path} to {html_path}")
    shutil.copytree(template_path, html_path)

    # Load status dict
    status = np.load(os.path.join(data_path, 'outputs/status.npy'),
        allow_pickle=True).item()
    band = status["band"]

    def replace_str(filename, search_str, replace_str):
        with fileinput.FileInput(filename, inplace=True, backup='.bak') as file:
            for line in file:
                print(line.replace(search_str, replace_str), end='')

    index_path = os.path.join(html_path, "index.html")

    n_res_found = len(status['which_on_before_check']['output'])
    n_res_track = len(status['which_on_after_check']['output'])
    ivdict = np.load(status['slow_iv_all']['output'].split('_raw_data')[0]+'.npy',
        allow_pickle=True).item()
    n_tes = len(ivdict[band].keys())

    # Fill why
    replace_str(index_path, "[[WHY]]",
                status['why']['output'])

    # Fill in time
    replace_str(index_path, "[[DATETIME]]",
                datetime.datetime.fromtimestamp(status['why']['start']).strftime('%Y-%m-%d'))

    # Summary
    summary_str = '<table style=\"width:30%\" align=\"center\" border=\"1\">'
    summary_str += f"<tr><td>Band</td><td>{band}</td>"
    summary_str += f"<tr><td>Resonators found</td><td>{n_res_found}</td>"
    summary_str += f"<tr><td>Resonators tracking</td><td>{n_res_track}</td>"
    summary_str += f"<tr><td>IV curves</td><td>{n_tes}</td>"
    summary_str += '</table>'
    replace_str(index_path, "[[SUMMARY]]",
                summary_str)

    # Do timing calculations
    skip_keys = ["band", "subband"]
    timing_str = '<table style=\"width:30%\" align=\"center\" border=\"1\">'
    timing_str += '<tr><th>Function</th><th>Time [s]</th></tr>'
    for k in list(status.keys()):
        if k not in skip_keys:
            t = status[k]['end'] - status[k]['start']
            timing_str += f'<tr><td>{k}</td><td>{t}</td></tr>'
    timing_str += '</table>'
    replace_str(index_path, "[[TIMING]]",
                timing_str)

    # Fill in band number
    replace_str(index_path, "[[BAND]]",
                str(band))

    # Amplifier bias
    amp_str = '<table style=\"width:30%\" align=\"center\" border=\"1\">'
    amp_dict = status['get_amplifier_bias']['output']
    for k in amp_dict.keys():
        amp_str += f'<tr><td>{k}</td><td>{amp_dict[k]:4.3f}</td></tr>'
    amp_str += '</table>'
    replace_str(index_path, "[[AMPLIFIER_BIAS]]",
                amp_str)

    # Add full band response plot
    basename = os.path.split(status['full_band_resp']['output'][2])[1]
    replace_str(index_path, "[[FULL_BAND_RESP]]",
                os.path.join('../plots/',basename))

    # Load tuning
    try:
        tn = np.load(status['save_tune']['output'], allow_pickle=True).item()
    except FileNotFoundError:
        print("Tuning file not found")

    res = tn[band]['resonances']
    res_list = np.array([], dtype=str)
    res_name = ""
    res_to_chan = ""
    for k in list(res.keys()):
        res_list = np.append(res_list, f"{res[k]['freq']:4.3f}|{k}")
        res_name = res_name + "\'" + f"{int(k):03}|{int(k):03}" + "\', "
        chan = res[k]['channel']
        res_to_chan = res_to_chan + f'\"{int(k):03}\":\"{chan:03}\", '

    res_name = '[' + res_name + ']'
    replace_str(index_path, "[[FREQ_RESP_LIST]]",
                res_name)

    replace_str(index_path, "[[RES_DICT]]",
                res_to_chan)

    # Load eta scans
    basename = os.path.split(glob.glob(os.path.join(data_path,
        'plots/*eta*'))[0])[1].split("res")
    instr = f"\'{basename[0]}\' + \'res\' + p[\'res\'] + \'.png\'"
    replace_str(index_path, "[[ETA_PATH]]",
                instr)

    # Load tracking setup
    basename = os.path.split(glob.glob(os.path.join(data_path,
        'plots/*tracking*'))[0])[1].split("_FRtracking")
    instr = f"\'{basename[0]}\' + \'_FRtracking_b{band}_ch\' + res_to_chan(p[\'res\']) + \'.png\'"
    replace_str(index_path, "[[TRACKING_PATH]]",
                instr)

    # Load bias group data
    bias_group_list = ""
    for bg in np.arange(8):
        bias_group_list += f"\'{bg:02}|{bg:02}\', "
    bias_group_list = "[" + bias_group_list + "]"
    replace_str(index_path, "[[BIAS_GROUP_LIST]]",
                bias_group_list)

    # Bias group path
    basename = os.path.split(glob.glob(os.path.join(data_path,
        'plots/*_identify_bg*'))[0])[1].split("_identify_bg")
    instr = f"\'{basename[0]}\' + \'_identify_bg\' + p[\'bg\'] + \'.png\'"
    replace_str(index_path, "[[BIAS_GROUP_PATH]]",
                instr)

    # IV path
    basename = os.path.split(glob.glob(os.path.join(data_path,
        'plots/*_IV_curve*'))[0])[1].split("_IV_curve")
    instr = f"\'{basename[0]}\' + \'_IV_curve_b{band}_ch\' + res_to_chan(p[\'res\']) + \'.png\'"
    replace_str(index_path, "[[IV_PATH]]",
                instr)
    
    # Noise PSD path
    basename = os.path.split(glob.glob(os.path.join(data_path,
        'plots/*_noise_timestream*'))[0])[1].split("_noise_timestream")
    instr = f"\'{basename[0]}\' + \'_noise_timestream_b{band}_ch\' + res_to_chan(p[\'res\']) + \'.png\'"
    replace_str(index_path, "[[NOISE_PSD_PATH]]",
                instr)

    return html_path

def run(band, epics_root, config_file, shelf_manager, setup, no_band_off=False,
        no_find_freq=False, subband_low=52, subband_high=460, loopback=False,
        no_setup_notches=False, reset_rate_khz=4, n_phi0=4, threading_test=False,
        c03_hack=False):
    """
    """
    # Storage dictionary
    status = {}
    status["band"] = band

    # Initialize
    S = pysmurf.client.SmurfControl(epics_root=epics_root,
        cfg_file=config_file, shelf_manager=shelf_manager,
        setup=False)

    print("All outputs going to: ")
    print(S.output_dir)

    def execute(status_dict, func, label, save_dict=True):
        """ Convenienec function used to run and time any arbitrary pysmurf
        functions. Must pass func as a lambda.

        Args
        ----
        status_dict : dict
            The dictionary that stores all the start/end times and outputs of
            the functions.
        label : str
            The descriptor to label the function
        save_dict : bool
            Whether to save the dict. Default True.

        Returns
        -------
        status_dict : dict
            The updated status dict
        """
        # Add label to dict
        status_dict[label] = {}

        # Note start time
        status_dict[label]['start'] = S.get_timestamp(as_int=True)

        # Run function
        status_dict[label]['output'] = func()

        # Add end time
        status_dict[label]['end'] = S.get_timestamp(as_int=True)

        # Save the dict
        np.save(os.path.join(S.output_dir, "status"),
                status_dict)

        return status_dict

    # why
    status = execute(status, lambda: S.why(), 'why')

    time.sleep(30)
    
    # Setup
    if setup:
        status = execute(status, lambda: S.setup(set_defaults_max_timeout_sec=1200),
                         'setup')

        # Check if setup succeeded
        if not status['setup']['output']:
            raise RuntimeError('Setup failed. Aborting test')

    # Band off
    if not no_band_off:
        bands = S.config.get('init').get('bands')
        for b in bands:
            status = execute(status, lambda: S.band_off(b),
                             f'band_off_b{b}')
    
    # Hack in for C03 cryocard
    if c03_hack:
        S.log('-> Running cryocard_c03_setup.py ...')

        # have to explicitly enable and set the Vd control DACs
        S.log('-> Enabling RTM DAC32, which drives DAC_In_50k_d')
        S.set_rtm_slow_dac_enable(32,2)

        S.log('-> Setting DAC_In_50k_d to 0.65V for 50kVd=5.003')
        S.set_rtm_slow_dac_volt(32,0.65) #corresponds to 50kVd->5.003
        
        S.log('-> Enabling RTM DAC31, which drives DAC_In_H_d')
        S.set_rtm_slow_dac_enable(31,2)
        
        S.log('-> Setting DAC_In_H_d to 6V for HVd=0.53')
        S.set_rtm_slow_dac_volt(31,6) #corresponds to HVd->0.53 
            
    # amplifier biases
    status = execute(status, lambda: S.set_amplifier_bias(write_log=True),
        'set_amplifier_bias')
    #status = execute(status, lambda: S.set_cryo_card_ps_en(write_log=True),
    #    'amplifier_enable')
    status = execute(status, lambda: S.C.write_ps_en(11), 'amplifier_enable')
    status = execute(status, lambda: S.get_amplifier_bias(),
        'get_amplifier_bias')


    # full band response
    status = execute(status, lambda: S.full_band_resp(band, make_plot=True,
        save_plot=True, show_plot=False, return_plot_path=True),
        'full_band_resp')
    
    status = execute(status, lambda: S.estimate_phase_delay(band),
                     'estimate_phase_delay')
    
    # find_freq
    if not no_find_freq:
        subband = np.arange(52, 460)
        if subband_low is not None and subband_high is not None:
            subband = np.arange(subband_low, subband_high)
        status['subband'] = subband
        status = execute(status, lambda: S.find_freq(band, subband=subband,
            make_plot=True, save_plot=True), 'find_freq')

    # setup notches
    if not no_setup_notches:
        status = execute(status, lambda: S.setup_notches(band,
            new_master_assignment=True), 'setup_notches')

        status = execute(status, lambda: S.plot_tune_summary(band,
            eta_scan=True, show_plot=False, save_plot=True),
            'plot_tune_summary')

    # Actually take a tuning serial gradient descent using tune_band_serial
    status = execute(status, lambda: S.run_serial_gradient_descent(band),
        'serial_gradient_descent')

    status = execute(status, lambda: S.run_serial_eta_scan(band),
        'serial_eta_scan')

    # track
    channel = S.which_on(band)
    status = execute(status, lambda: S.tracking_setup(band, channel=channel,
        reset_rate_khz=reset_rate_khz, fraction_full_scale=.5,
        make_plot=True, show_plot=False, nsamp=2**18, lms_gain=2,
        lms_freq_hz=None, meas_lms_freq=False, meas_flux_ramp_amp=True,
        n_phi0=n_phi0, feedback_start_frac=.2, feedback_end_frac=.98),
        'tracking_setup')

    # See what's on
    status = execute(status, lambda: S.which_on(band), 'which_on_before_check')

    # now track and check
    status = execute(status, lambda: S.check_lock(band), 'check_lock')

    status = execute(status, lambda: S.which_on(band), 'which_on_after_check')

    # Identify bias groups
    status = execute(status, lambda: S.identify_bias_groups(bias_groups=np.arange(8),
        make_plot=True, show_plot=False, save_plot=True, update_channel_assignment=True),
        'identify_bias_groups')

    # Save tuning
    status = execute(status, lambda: S.save_tune(), 'save_tune')

    # now take data using take_noise_psd and plot stuff

    # IV.
    status = execute(status, lambda: S.run_iv(np.arange(8),
        overbias_voltage=19.9, bias_high=10, bias_step=.01, wait_time=.1,
        high_current_mode=False, overbias_wait=.5, cool_wait=60,
        make_plot=True), 'slow_iv_all')

    # Take noise for 60 seconds
    status = execute(status, lambda: S.take_noise_psd(60), 'take_noise_psd')

    # Make webpage
    html_path = make_html(os.path.split(S.output_dir)[0])

    if threading_test:
        import threading
        for t in threading.enumerate():
            print(t)
            S.log(t)

    return html_path

if __name__ == "__main__":
    #####################
    # Arg parse things
    #####################

    # Create argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--epics-root", type=str, required=True,
                        help="The epics root.")
    parser.add_argument("--config-file", type=str, required=True,
                        help="The configuration file to use for this test.")
    parser.add_argument("--shelf-manager", type=str, required=True,
                        help="The shelf manager root.")
    parser.add_argument("--setup", default=False,
                        action="store_true",
                        help="Whether to run setup.")
    parser.add_argument("--band", type=int, required=True,
                        help="The band to run the analysis on.")
    parser.add_argument("--reset-rate-khz", type=int, required=False,
                        default=4,
                        help="The flux ramp reset rate")
    parser.add_argument("--n-phi0", type=float, required=False, default=4,
                        help="The number of phi0 per flux ramp desired.")
    parser.add_argument("--no-find-freq", default=False,
                        action="store_true",
                        help="Skip the find_freq step")
    parser.add_argument("--no-setup-notches", default=False,
                        action="store_true",
                        help="Skip the setup_notches")
    parser.add_argument("--subband-low", type=int, required=False,
                        help="The starting subband for find_freq")
    parser.add_argument("--subband-high", type=int, required=False,
                        help="The end subband for find_freq")
    parser.add_argument("--no-band-off", default=False,
                        action="store_true",
                        help="Whether to skip turning off bands")
    parser.add_argument("--threading-test", default=False,
                       action="store_true",
                       help="Whether to run threading test")

    # Parse arguments
    args = parser.parse_args()

    # Run the generator script
    run(args.band, args.epics_root, args.config_file, args.shelf_manager,
        args.setup, no_band_off=args.no_band_off, no_find_freq=args.no_find_freq,
        subband_low=args.subband_low, subband_high=args.subband_high,
        no_setup_notches=args.no_setup_notches,
        reset_rate_khz=args.reset_rate_khz, n_phi0=args.n_phi0,
        threading_test=args.threading_test)
