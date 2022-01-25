# tes_yield.py
'''
Code written in Oct 2021 by Yuhan Wang.
Check TES yield by taking bias tickle (from sodetlib) and IV curves.
Display quality in biasability, 50% RN target V bias, Psat and Rn.
'''
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import numpy as np
import os
import time
import glob
import csv
import pysmurf.client
from sodetlib.det_config  import DetConfig
from sodetlib.smurf_funcs import det_ops
from sodetlib.analysis import det_analysis
from pysmurf.client.util.pub import set_action
import logging

logger = logging.getLogger(__name__)

def tickle_and_iv(
        S, target_bg, bias_high, bias_low, bias_step, bath_temp, start_time, current_mode
):
    overbias_voltage = 18
    target_bg = np.array(target_bg)
    save_name = '{}_tes_yield.csv'.format(start_time)
    tes_yield_data = os.path.join(S.output_dir, save_name)
    logger.info(f'Saving data to {tes_yield_data}')
    out_fn = os.path.join(S.output_dir, tes_yield_data) 

    fieldnames = ['bath_temp', 'bias_line', 'band', 'data_path','notes']
    with open(out_fn, 'w', newline = '') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

    logger.info(f'Taking tickle on bias line all band')

    if current_mode.lower() in ['high, hi']:
        high_current_mode = True
        bias_high /= S.high_low_current_ratio
        bias_low /= S.high_low_current_ratio
        bias_step /= S.high_low_current_ratio
    else:
        high_current_mode = False

    tickle_file = det_ops.take_tickle(
        S, cfg, target_bg, tickle_freq=5, tickle_voltage=0.005, high_current=True
    )
    tsum, tsum_fp = det_analysis.analyze_tickle_data(S, tickle_file, normal_thresh=0.002)

    row = {}
    row['bath_temp'] = str(bath_temp)
    row['bias_line'] = str(target_bg)
    row['band'] = 'all'
    row['data_path'] = tickle_file

    with open(out_fn, 'a', newline = '') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow(row)

    for bg in target_bg:
        row = {}
        row['bath_temp'] = str(bath_temp)
        row['bias_line'] = bg
        row['band'] = 'all'

        logger.info(f'Taking IV on bias line {bg}, all smurf bands.')

        iv_data = det_ops.take_iv(
            S, cfg,
            bias_groups = [bg], wait_time=0.01, bias_high=bias_high,
            bias_low=bias_low, bias_step=bias_step,
            overbias_voltage=12, cool_wait=30, high_current_mode=high_current_mode,
            make_channel_plots=False, save_plots=True,
        )
        dat_file = iv_data.replace('info','analyze')     
        row['data_path'] = dat_file
        with open(out_fn, 'a', newline = '') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow(row)

    return out_fn


@set_action()
def tes_yield(S, target_bg, out_fn, start_time):
    data_dict = np.genfromtxt(out_fn, delimiter=",",unpack=True, dtype=None, names=True, encoding=None)
    
    data = []
    for ind in np.arange(1,len(target_bg) + 1):
        file_path = str(data_dict['data_path'][ind])
        data.append(file_path)
    
    good_chans = 0
    all_data_IV = dict()

    for ind, bl in enumerate(target_bg):
        if bl not in all_data_IV.keys():
            all_data_IV[bl] = dict()
        now = np.load(data[bl], allow_pickle=True).item()
        now = now['data']
        for sb in now.keys():
            if len(now[sb].keys()) != 0:
                all_data_IV[bl][sb] = dict()
            for chan, d in now[sb].items():
                if (d['R'][-1] < 5e-3):
                    continue
                elif len(np.where(d['R'] > 15e-3)[0]) > 0:
                    continue
                all_data_IV[bl][sb][chan] = d

    S.pub.register_file(out_fn, "tes_yield", format='.csv')

    common_biases = set()
    now_bias = set()
    for bl in all_data_IV.keys():
        for sb in all_data_IV[bl].keys():
            if len(all_data_IV[bl][sb].keys()) != 0:
                first_chan = next(iter(all_data_IV[bl][sb]))
                now_bias = set(all_data_IV[bl][sb][first_chan]['v_bias'])
                if len(common_biases) == 0:
                    common_biases = now_bias
                else:
                    common_biases = common_biases.intersection(now_bias)
    common_biases = np.array(sorted([np.float("%0.3f" % i) for i in common_biases]))
    common_biases = np.array(common_biases)

    operating_r = dict()
    for bl in target_bg:
        operating_r[bl] = dict()
        for sb in all_data_IV[bl].keys():
            if len(all_data_IV[bl][sb].keys()) == 0:
                continue
            for v in common_biases:
                if v not in operating_r[bl].keys():
                    operating_r[bl][v] = []
                first_chan = next(iter(all_data_IV[bl][sb]))
                ind = np.where(
                    (np.abs(all_data_IV[bl][sb][first_chan]['v_bias'] - v)) < 3e-3
                )[0][0]
                for chan, d in all_data_IV[bl][sb].items():
                    operating_r[bl][v].append(d['R'][ind]/d['R_n']) 

    target_vbias_dict = {}
    RN = []
    v_bias_all = []
    for bl in target_bg:
        percent_rn = 0.5
        target_v_bias = []

        for sb in all_data_IV[bl].keys():
            for ch, d in all_data_IV[bl][sb].items():
                rn = d['R']/d['R_n']
                cross_idx = np.where(np.logical_and(rn - percent_rn >= 0, np.roll(rn - percent_rn, 1) < 0))[0]
                if len(cross_idx) == 0:
                    continue
                RN.append(d['R_n'])
                target_v_bias.append(d['v_bias'][cross_idx][0]) 
                v_bias_all.append(d['v_bias'][cross_idx][0])
        med_target_v_bias = np.nanmedian(np.array(target_v_bias))
        target_vbias_dict[bl] = round(med_target_v_bias,1)

    target_vbias_fp = os.path.join(S.output_dir, f"{start_time}_target_vbias.npy")
    np.save(target_vbias_fp, target_vbias_dict, allow_pickle=True)
    S.pub.register_file(target_vbias_fp, "tes_yield", format='npy')

    fig, axs = plt.subplots(6, 4,figsize=(25,30), gridspec_kw={'width_ratios': [2, 1,2,1]})
    for ind, bl in enumerate(target_bg):
        if np.isnan(target_vbias_dict[bl]):
            continue
        count_num = 0
        for sb in all_data_IV[bl].keys():
            for ch,d in all_data_IV[bl][sb].items():
                axs[bl//2,bl%2*2].plot(d['v_bias'], d['R'], alpha=0.6)
                count_num = count_num + 1
        axs[bl//2,bl%2*2].set_xlabel('V_bias [V]')
        axs[bl//2,bl%2*2].set_ylabel('R [Ohm]')
        axs[bl//2,bl%2*2].grid()
        axs[bl//2,bl%2*2].axhspan(2.6e-3, 5.8e-3, facecolor='gray', alpha=0.2)
        axs[bl//2,bl%2*2].axvline(target_vbias_dict[bl], linestyle='--', color='gray')
        axs[bl//2,bl%2*2].set_title('bl {}, yield {}'.format(bl,count_num))
        axs[bl//2,bl%2*2].set_ylim([-0.001,0.012])

        h = axs[bl//2,bl%2*2+1].hist(operating_r[bl][target_vbias_dict[bl]], range=(0,1), bins=40)
        axs[bl//2,bl%2*2+1].axvline(np.median(operating_r[bl][target_vbias_dict[bl]]),linestyle='--', color='gray')
        axs[bl//2,bl%2*2+1].set_xlabel("percentage Rn")
        axs[bl//2,bl%2*2+1].set_ylabel("{} TESs total".format(count_num))
        axs[bl//2,bl%2*2+1].set_title("optimal Vbias {}V for median {}Rn".format(
            target_vbias_dict[bl],np.round(np.median(operating_r[bl][target_vbias_dict[bl]]),2))
        )

    save_name = os.path.join(S.plot_dir, f'{start_time}_IV_yield.png')
    logger.info(f'Saving plot to {save_name}')
    plt.savefig(save_name)

    S.pub.register_file(save_name, "tes_yield", plot=True)

    fig, axs = plt.subplots(6, 4, figsize=(25,30))
    for bl in target_bg:
        count_num = 0
        Rn = []
        psat = []
        for sb in all_data_IV[bl].keys():
            for ch,d in all_data_IV[bl][sb].items():
                Rn.append(d['R_n'])
                psat.append(d['p_sat'].item())
                count_num += 1

        axs[bl//2,bl%2*2].set_xlabel('P_sat (pW)')
        axs[bl//2,bl%2*2].set_ylabel('count')
        axs[bl//2,bl%2*2].grid()
        axs[bl//2,bl%2*2].hist(psat, range=(0,15), bins=50,histtype= u'step',linewidth=2,color = 'r')
        axs[bl//2,bl%2*2].axvline(np.median(psat), linestyle='--', color='gray')
        axs[bl//2,bl%2*2].set_title('bl {}, yield {} median Psat {:.2f} pW'.format(
            bl,count_num,np.median(psat))
        )
        h = axs[bl//2,bl%2*2+1].hist(Rn, range=(0.005,0.01), bins=50, histtype= u'step',linewidth=2,color = 'k')
        axs[bl//2,bl%2*2+1].axvline(np.median(Rn),linestyle='--', color='gray')
        axs[bl//2,bl%2*2+1].set_xlabel("Rn (Ohm)")
        axs[bl//2,bl%2*2+1].set_ylabel('count')
        axs[bl//2,bl%2*2+1].set_title('bl {}, median Rn {:.4f} Ohm'.format(bl,np.median(Rn)))

    save_name = os.path.join(S.plot_dir, f'{start_time}_IV_psat.png')
    logger.info(f'Saving plot to {save_name}')
    plt.savefig(save_name)

    S.pub.register_file(save_name, "tes_yield", plot=True)

    return target_vbias_dict


def run(S, cfg, bias_high=20, bias_low=0, bias_step=0.025, bath_temp=100, current_mode='low'):
    start_time = S.get_timestamp()
    target_bg = np.arange(12)

    out_fn = tickle_and_iv(
        S, target_bg, bias_high, bias_low, bias_step, bath_temp, start_time, current_mode)
    target_vbias = tes_yield(S, target_bg, out_fn, start_time)
    logger.info(f'Saving data to {out_fn}')
    return target_vbias


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--temp', type=str,
                        help="For record-keeping, not controlling,"
    )
    parser.add_argument('--bias-high', type=float, default=20)
    parser.add_argument('--bias-low', type=float, default=0)
    parser.add_argument('--bias-step', type=float, default=0.025)
    parser.add_argument('--current-mode', type=str, default='low')
    parser.add_argument(
        "--loglevel",
        type=str.upper,
        default=None,
        choices=['DEBUG','INFO','WARNING','ERROR','CRITICAL'],
        help="Set the log level for printed messages. The default is pulled from "
        +"$LOGLEVEL, defaulting to INFO if not set.",
    )

    cfg = DetConfig()
    args = cfg.parse_args(parser)
    if args.loglevel is None:
        args.loglevel = os.environ.get("LOGLEVEL","INFO")
    numeric_level = getattr(logging, args.loglevel)
    logging.basicConfig(
        format="%(levelname)s: %(funcName)s: %(message)s", level=numeric_level
    )

    S = cfg.get_smurf_control(make_logfile=(numeric_level != 10))

    S.load_tune(cfg.dev.exp['tunefile'])

    bath_temp = args.temp
    bias_high = args.bias_high
    bias_low = args.bias_low
    bias_step = args.bias_step
    current_mode = args.current_mode

    run(S, cfg, bias_high=bias_high, bias_low=bias_low,
        bias_step=bias_step, bath_temp=bath_temp, current_mode=current_mode)
