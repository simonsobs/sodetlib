import numpy as np
import matplotlib
matplotlib.use('Agg')
from pprint import pprint
import argparse

from sodetlib.det_config import DetConfig
from sodetlib.smurf_funcs.smurf_ops import take_g3_data
import sodetlib.smurf_funcs.det_ops as do
import sodetlib.analysis.det_analysis as da

parser = argparse.ArgumentParser(description='Setup for a set of observations')
parser.add_argument('smurf_bands', help='Which SMuRF Bands to Use: format is 0,1,2,3')
parser.add_argument('bias_groups', help='Which Bias Groups to Use: format is 0,2,3,4')
parser.add_argument('--setup_notches', default=False, 
        help='Whether or not to do a setup notches')
parser.add_argument('--new_master_assignment', default=False, 
        help="Whether or not to create a new master channel assignment")
parser.add_argument('--stream_time', default=10, 
        help="Length of time in seconds to stream after tracking setup and biasing")
parser.add_argument('--take_tickle', default=False,
        help="Do bias tickle to determine bias groups")
parser.add_argument('--bias_map', help="bias map file, required if take_tickle is False")
parser.add_argument('--tune_file', help="load specific tune file, otherwise load most recent")

if __name__ == '__main__':
    args = parser.parse_args()

    try:
        bands = [int(x) for x in args.smurf_bands.split(',')]
    except: 
        print("Parsing failed for bands list. Correct format is x,y,z")
        raise
    try:
        bgs = [int(x) for x in args.bias_groups.split(',')]
    except:
        print("Parsing failed for bias group list. Correct format is x,y,z")
        raise
    
    print(f"Will setup on bands {bands} and bias groups {bgs}")
    if args.new_master_assignment:
        raise NotImplementedError("Creating new Master Assignment still TBD")
    print(f"Will stream data for {args.stream_time} during setup.")
    if not args.take_tickle and args.bias_map is None:
        raise ValueError("If take_tickle is False we require a bias map")
    if args.take_tickle:
        raise NotImplementedError("Creating new tickle info still TBD")
    print(f"Will use bias map file: {args.bias_map}")
     
    cfg = DetConfig()
    cfg.load_config_files(slot=2)
    S = cfg.get_smurf_control(dump_configs=True)
    
    ## make sure biases are off
    for bg in bgs:
        S.set_tes_bias_bipolar(bg,0)

    if args.tune_file is None:
        S.load_tune()
    else:
        S.load_tune(args.tune_file)
    
    for band in bands:
        band_cfg = cfg.dev.bands[band]
        S.set_att_uc( band, band_cfg['uc_att'] )
        S.set_att_dc( band, band_cfg['dc_att'] )

        if args.setup_notches:
            S.setup_notches(band, tone_power=band_cfg['drive'],
                            new_master_assignment=args.new_master_assignment)
        else:
            S.relock(band)

        for _ in range(3):
            S.run_serial_gradient_descent(band)
            S.run_serial_eta_scan(band)

        S.tracking_setup(band, reset_rate_khz=cfg.dev.bands[band]['flux_ramp_rate_khz'],
                     fraction_full_scale=cfg.dev.bands[band]['frac_pp'],
                     make_plot=False, save_plot=False, show_plot=False,
                     nsamp=2**18, lms_freq_hz=band_cfg['lms_freq_hz'],
                     meas_lms_freq=False,
                     feedback_start_frac=cfg.dev.bands[band]['feedback_start_frac'],
                     feedback_end_frac=cfg.dev.bands[band]['feedback_end_frac'],
                     lms_gain=cfg.dev.bands[band]['lms_gain'])
    
    take_g3_data(S, args.stream_time, tag='setup,noise,superconducting')
    if args.take_tickle:
        pass
    iv_info_fp = do.take_iv(S,bias_groups=bgs,bias_high=19.9,bias_low=0.0,
                            bias_step=0.1,cool_wait=20.,high_current_mode=False,
                            do_analysis=False,cool_voltage=10.0,overbias_voltage=19.9)
    iv_info = np.load(iv_info_fp,allow_pickle=True).item()
    timestamp,phase,mask,tes_biases = da.load_from_dat(S,iv_info['datafile'])
    iv_analyze_fp = da.analyze_iv_and_save(S,iv_info_fp,phase,tes_biases,mask)
    iv_analyze = np.load(iv_analyze_fp,allow_pickle=True).item()

    chosen_biases_fp = da.find_bias_points(S,iv_analyze_fp,args.bias_map,
                                            bias_point=0.5,bias_groups=bgs)
    do.bias_detectors_from_sc(S, chosen_biases_fp)

    take_g3_data(S, args.stream_time, tag='setup,noise,biased')
