'''
Code written in Oct 2021 by Yuhan Wang
check TES yield by taking bias tickle (from sodetlib) and IV curves
display quality in biasability, 50% RN target V bias, Psat and Rn
'''



import matplotlib
matplotlib.use('Agg')

import pysmurf.client
import argparse
import numpy as np
import os
import time
import glob
from sodetlib.det_config  import DetConfig
import numpy as np
from scipy.interpolate import interp1d
import argparse
import time
import csv
import sodetlib.smurf_funcs.det_ops as det_op
import sodetlib.analysis.det_analysis as det_analysis


start_time=S.get_timestamp()

target_BL = np.array([0,1,2,3,4,5,6,7,8,9,10,11])

#this is more for keeping track of bath temp
bath_temp = 100
bias_high_command=20
bias_low_command=0
bias_step_command = 0.025

save_name = '{}_tes_yield.csv'.format(start_time)
print(f'Saving data to {os.path.join(S.output_dir, save_name)}')
tes_yield_data = os.path.join(S.output_dir, save_name)
path = os.path.join(S.output_dir, tes_yield_data) 


out_fn = path



fieldnames = ['bath_temp', 'bias_line', 'band', 'data_path','notes']
with open(out_fn, 'w', newline = '') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
 
 
print(f'Taking tickle on bias line all band')

tickle_file = det_op.take_tickle(S, cfg, target_BL, tickle_freq=5, tickle_voltage=0.005,high_current=True)
det_analysis.analyze_tickle_data(S, tickle_file,normal_thresh=0.002)  

row = {}
row['bath_temp'] = bath_temp
row['bias_line'] = str(target_BL)
row['band'] = 'all'
     

row['data_path'] = tickle_file
  
with open(out_fn, 'a', newline = '') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writerow(row)

for bias_gp in target_BL:
        row = {}
        row['bath_temp'] = bath_temp
        row['bias_line'] = bias_gp
        row['band'] = 'all'
             
        print(f'Taking IV on bias line {bias_gp}, all band')
          
 
        iv_data = S.run_iv(bias_groups = [bias_gp], wait_time=0.001, bias_high=bias_high_command, bias_low=bias_low_command, bias_step = bias_step_command, overbias_voltage=18, cool_wait=0, high_current_mode=False, make_plot=False, save_plot=True, cool_voltage = 8)
        dat_file = iv_data[0:-13]+'.npy'     
        row['data_path'] = dat_file
        with open(out_fn, 'a', newline = '') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow(row)




def write_IV_into_dict(IV_csv):
    
    data_dict = np.genfromtxt(IV_csv, delimiter=",",unpack=True, dtype=None, names=True, encoding=None)
    psat_array = []
    
    data = []
    for ind in np.array([0,1,2,3,4,5,6,7,8,9,10,11])+1:
        file_path = str(data_dict['data_path'][ind])
        data.append(file_path)
    
    
    
    good_chans = 0
    psat_all = []
    all_data = dict()
    good = 0
    bad = 0
    

    for ind, bl in enumerate([0,1,2,3,4,5,6,7,8,9,10,11]):
        ch_psat = []
        if bl not in all_data.keys():
            all_data[bl] = dict()
        now = np.load(data[bl], allow_pickle=True).item()
        for sb in [0,1,2,3,4,5,6,7]:
            try:
                if len(now[sb].keys()) != 0:
                    all_data[bl][sb] = dict()
            except:
                continue
    #         print(now[sb].keys())
            for chan, d in now[sb].items():
#                 print(chan)
    #             print(d.keys())
                if (d['R'][-1] < 5e-3):
                    continue
                elif len(np.where(d['R'] > 10e-3)[0]) > 0:
                    continue
#                 elif len(np.where(d['R'] < -2e-4)[0]) > 0:
#                     continue
                all_data[bl][sb][chan] = d
                good_chans += 1
                try:
                    psat = np.float(get_psat(np.load(data[ind],allow_pickle=True).item(),sb,chan, level = 0.9, greedy = False))
                    if psat > 0.5:
                        ch_psat.append(psat)
                        good = good + 1
                        good_band.append(int(det_band))
                        good_chan.append(int(det_chan))
            
                except:
                    bad = bad + 1
        psat_all.append(ch_psat)
    psat_array.append(psat_all)
    return all_data,psat_array


all_data_IV,Psat_array = write_IV_into_dict(out_fn)

common_biases = set()
now_bias = set()
for bl in all_data_IV.keys():
    for sb in [0,1,2,3,4,5,6,7]:
        try:
            if len(all_data_IV[bl][sb].keys()) != 0:
                first_chan = next(iter(all_data_IV[bl][band]))
                now_bias = set(all_data_IV[bl][sb][first_chan]['v_bias'])
                if len(common_biases) == 0:
                    common_biases = now_bias
                else:
                    common_biases = common_biases.intersection(now_bias)
        except:
            continue
common_biases = np.array(
    sorted([np.float("%0.3f" % i) for i in common_biases])
)
common_biases = np.array(common_biases)

operating_r = dict()
for bl in [0,1,2,3,4,5,6,7,8,9,10,11]:
    operating_r[bl] = dict()
    for band in [0,1,2,3,4,5,6,7]:
        try:
            if len(all_data_IV[bl][band].keys()) == 0:
                continue
        except:
            continue
        for v in common_biases:
            if v not in operating_r[bl].keys():
                operating_r[bl][v] = []
            first_chan = next(iter(all_data_IV[bl][band]))
            ind = np.where(
                (np.abs(all_data_IV[bl][band][first_chan]['v_bias'] - v)) < 3e-3
            )[0][0]
            for chan, d in all_data_IV[bl][band].items():
                operating_r[bl][v].append(d['R'][ind]/d['R_n']) 




bias_groups = target_BL
target_vbias_list = []
RN = []
v_bias_all = []
for bl in bias_groups:
    percent_rn = 0.5
    target_v_bias = []

    for band in [0,1,2,3,4,5,6,7]:
        try:

            for ch,d in all_data_IV[bl][band].items():
                rn = d['R']/d['R_n']
                cross_idx = np.where(np.logical_and(rn - percent_rn >= 0, np.roll(rn - percent_rn, 1) < 0))[0]
                RN.append(d['R_n'])
                target_v_bias.append(d['v_bias'][cross_idx][0]) 
                v_bias_all.append(d['v_bias'][cross_idx][0])
        except:
            continue

# print(target_v_bias)
    med_target_v_bias = np.median(np.array(target_v_bias))
    target_vbias_list.append(round(med_target_v_bias,1))
target_vbias_list = np.append(target_vbias_list,[0,0,0])
print(np.array(target_vbias_list))

total_count = 0
fig, axs = plt.subplots(6, 4,figsize=(25,30), gridspec_kw={'width_ratios': [2, 1,2,1]})
for bl in [0,1,2,3,4,5,6,7,8,9,10,11]:
    count_num = 0
    for band in [0,1,2,3,4,5,6,7]:
        try:
            for ch,d in all_data_IV[bl][band].items():
                axs[bl//2,bl%2*2].plot(d['v_bias'], d['R'], alpha=0.6)
                count_num = count_num + 1
                total_count = total_count + 1
        except:
            continue
    axs[bl//2,bl%2*2].set_xlabel('V_bias [V]')
    axs[bl//2,bl%2*2].set_ylabel('R [Ohm]')
    axs[bl//2,bl%2*2].grid()
    axs[bl//2,bl%2*2].axhspan(2.6e-3, 5.8e-3, facecolor='gray', alpha=0.2)
    axs[bl//2,bl%2*2].axvline(target_vbias_list[bl], linestyle='--', color='gray')
    axs[bl//2,bl%2*2].set_title('bl {}, yield {}'.format(bl,count_num))
    axs[bl//2,bl%2*2].set_ylim([-0.001,0.012])
    # print(bl)
    try:
        h = axs[bl//2,bl%2*2+1].hist(operating_r[bl][target_vbias_list[bl]], range=(0,1), bins=40)
        axs[bl//2,bl%2*2+1].axvline(np.median(operating_r[bl][target_vbias_list[bl]]),linestyle='--', color='gray')
        axs[bl//2,bl%2*2+1].set_xlabel("percentage Rn")
        axs[bl//2,bl%2*2+1].set_ylabel("{} TESs total".format(count_num))
        axs[bl//2,bl%2*2+1].set_title("optimal Vbias {}V for median {}Rn".format(target_vbias_list[bl],np.round(np.median(operating_r[bl][target_vbias_list[bl]]),2)))
    except:
        continue

save_name = f'{start_time}_IV_yield.png'
print(f'Saving plot to {os.path.join(S.plot_dir, save_name)}')
plt.savefig(os.path.join(S.plot_dir, save_name))


fig, axs = plt.subplots(6, 4,figsize=(25,30), gridspec_kw={'width_ratios': [2, 2,2,2]})
for bl in [0,1,2,3,4,5,6,7,8,9,10,11]:
    count_num = 0
    Rn = []
    psat = []
    for band in [0,1,2,3,4,5,6,7]:
        try:
            for ch,d in all_data_IV[bl][band].items():
                Rn.append(d['R_n'])
                
                level = 0.9
                p = d['p_tes']
                rn = d['R']/d['R_n']
                cross_idx = np.where(np.logical_and(rn - level >= 0, np.roll(rn - level, 1) < 0))[0]
                try:
                    assert len(cross_idx) == 1
                except AssertionError:
                    continue
        
                cross_idx = cross_idx[:1]
                cross_idx = cross_idx[0]
                rn2p = interp1d(rn[cross_idx-1:cross_idx+1], p[cross_idx-1:cross_idx+1])
                psat.append(np.float(rn2p(level)))
                count_num = count_num + 1
                total_count = total_count + 1
        except:
            continue

    axs[bl//2,bl%2*2].set_xlabel('P_sat (pW)')
    axs[bl//2,bl%2*2].set_ylabel('count')
    axs[bl//2,bl%2*2].grid()
    axs[bl//2,bl%2*2].hist(psat, range=(0,15), bins=50,histtype= u'step',linewidth=2,color = 'r')
    axs[bl//2,bl%2*2].axvline(np.median(psat), linestyle='--', color='gray')
    axs[bl//2,bl%2*2].set_title('bl {}, yield {} median Psat {:.2f} pW'.format(bl,count_num,np.median(psat)))


    # print(bl)

    h = axs[bl//2,bl%2*2+1].hist(Rn, range=(0.005,0.01), bins=50,histtype= u'step',linewidth=2,color = 'k')
    axs[bl//2,bl%2*2+1].axvline(np.median(Rn),linestyle='--', color='gray')
    axs[bl//2,bl%2*2+1].set_xlabel("Rn (Ohm)")
    axs[bl//2,bl%2*2+1].set_ylabel('count')
    axs[bl//2,bl%2*2+1].set_title('bl {}, midian Rn {:.4f} Ohm'.format(bl,np.median(Rn)))


save_name = f'{start_time}_IV_psat.png'
print(f'Saving plot to {os.path.join(S.plot_dir, save_name)}')
plt.savefig(os.path.join(S.plot_dir, save_name))

print(f'Saving data to {out_fn}')




























 
