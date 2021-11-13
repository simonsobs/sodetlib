'''
Code written in Oct 2021 by Yuhan Wang
to be used through OCS
UFM testing script in Pton
takes SC noise, normal noise, IV, (noise and biasstep) at 30,50,70 percent Rn
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


parser = argparse.ArgumentParser()


parser.add_argument('--slot',type=int)
parser.add_argument('--temp',type=str)
parser.add_argument('--output_file',type=str)

args = parser.parse_args()
slot_num = args.slot
bath_temp = args.temp
out_fn = args.output_file



cfg = DetConfig()
cfg.load_config_files(slot=slot_num)
S = cfg.get_smurf_control()
if slot_num == 2:
    fav_tune_files = '/data/smurf_data/tune/1634501972_tune.npy'
if slot_num == 3:
    fav_tune_files = '/data/smurf_data/tune/1634492357_tune.npy'
if slot_num == 4:
    fav_tune_files = '/data/smurf_data/tune/1634507354_tune.npy'
if slot_num == 5:
    fav_tune_files = '/data/smurf_data/tune/1633652773_tune.npy'


S.all_off()
S.set_rtm_arb_waveform_enable(0)
S.set_filter_disable(0)
S.set_downsample_factor(20)
S.set_mode_dc()

S.load_tune(fav_tune_files)

bands = [0,1,2,3,4,5,6,7]


for band in bands:
    print('setting up band {}'.format(band))

    S.set_att_dc(band,cfg.dev.bands[band]['dc_att'])
    print('band {} dc_att {}'.format(band,S.get_att_dc(band)))

    S.set_att_uc(band,cfg.dev.bands[band]['uc_att'])
    print('band {} uc_att {}'.format(band,S.get_att_uc(band)))

    S.amplitude_scale[band] = cfg.dev.bands[band]['drive']
    print('band {} tone power {}'.format(band,S.amplitude_scale[band] ))

    print('setting synthesis scale')
    # hard coding it for the current fw
    S.set_synthesis_scale(band,1)

    print('running relock')
    S.relock(band,tone_power=cfg.dev.bands[band]['drive'])
    
    S.run_serial_gradient_descent(band);
    S.run_serial_eta_scan(band);
    
    print('running tracking setup')
    S.set_feedback_enable(band,1) 
    S.tracking_setup(band,reset_rate_khz=cfg.dev.bands[band]['flux_ramp_rate_khz'],fraction_full_scale=cfg.dev.bands[band]['frac_pp'], make_plot=False, save_plot=False, show_plot=False, channel=S.which_on(band), nsamp=2**18, lms_freq_hz=None, meas_lms_freq=True,feedback_start_frac=cfg.dev.bands[band]['feedback_start_frac'],feedback_end_frac=cfg.dev.bands[band]['feedback_end_frac'],lms_gain=cfg.dev.bands[band]['lms_gain'])
    print('checking tracking')
    S.check_lock(band,reset_rate_khz=cfg.dev.bands[band]['flux_ramp_rate_khz'],fraction_full_scale=cfg.dev.bands[band]['frac_pp'], lms_freq_hz=None, feedback_start_frac=cfg.dev.bands[band]['feedback_start_frac'],feedback_end_frac=cfg.dev.bands[band]['feedback_end_frac'],lms_gain=cfg.dev.bands[band]['lms_gain'])

bias_groups = [0,1,2,3,4,5,6,7,8,9,10,11] 
S.set_filter_disable(0)
S.set_rtm_arb_waveform_enable(0)
S.set_downsample_factor(20)
for bias_index, bias_g in enumerate(bias_groups):
    S.set_tes_bias_low_current(bias_g)

bias_v = 0

## SC noise
S.set_rtm_arb_waveform_enable(0)
S.set_filter_disable(0)
S.set_downsample_factor(20)
S.set_tes_bias_bipolar_array([bias_v,bias_v,bias_v,bias_v,bias_v,bias_v,bias_v,bias_v,bias_v,bias_v,bias_v,bias_v,0,0,0])

time.sleep(120)


datafile_self = S.stream_data_on()
time.sleep(120)
S.stream_data_off()

fieldnames = ['bath_temp','bias_voltage', 'bias_line', 'band', 'data_path','type','note'] 
row = {}
row['data_path'] = datafile_self
row['bias_voltage'] = bias_v
row['type'] = 'sc noise'
row['bias_line'] = 'all'
row['band'] = 'all'
row['bath_temp'] = bath_temp
with open(out_fn, 'a', newline = '') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writerow(row)


## Normal noise
S.set_rtm_arb_waveform_enable(0)
S.set_filter_disable(0)
S.set_downsample_factor(20)
S.overbias_tes_all(bias_groups = [0,1,2,3,4,5,6,7,8,9,10,11], overbias_wait=1, tes_bias= 15, cool_wait= 3, high_current_mode=False, overbias_voltage= 5)
## sleep 6 mins to get stablized 
for i in range(36):
    time.sleep(10)
datafile_self = S.stream_data_on()
time.sleep(120)
S.stream_data_off()
row = {}
row['data_path'] = datafile_self
row['bias_voltage'] = 20
row['type'] = 'normal noise'
row['bias_line'] = 'all'
row['band'] = 'all'
row['bath_temp'] = bath_temp
with open(out_fn, 'a', newline = '') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writerow(row)


##IV
bl_iv_list = []
for bias_gp in [0,1,2,3,4,5,6,7,8,9,10,11]:
    row = {}
    row['bath_temp'] = bath_temp
    row['bias_line'] = bias_gp
    row['band'] = 'all'
    row['bias_voltage'] = 'IV 20 to 0'
    row['type'] = 'IV'
    print(f'Taking IV on bias line {bias_gp}, all band')
      

    iv_data = S.run_iv(bias_groups = [bias_gp], wait_time=0.001, bias_high=20, bias_low=0, bias_step = 0.025, overbias_voltage=18, cool_wait=0, high_current_mode=False, make_plot=False, save_plot=True, cool_voltage = 18)
    dat_file = iv_data[0:-13]+'.npy'     
    row['data_path'] = dat_file
    bl_iv_list.append(dat_file)
    with open(out_fn, 'a', newline = '') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow(row)


#no wave form please
S.set_rtm_arb_waveform_enable(0)




##get target v bias from IV
good_chans = 0
all_data = dict()
for ind, bl in enumerate(bias_groups):
    if bl not in all_data.keys():
        all_data[bl] = dict()
    now = np.load(bl_iv_list[bl], allow_pickle=True).item()
#     print(now[3].keys())
#     print(now[0].keys())
#     print(now[0][0]['R'])
    
    for sb in [0,1,2,3,4,5,6,7]:
        try:
            if len(now[sb].keys()) != 0:
                all_data[bl][sb] = dict()
        except:
            continue
#         print(now[sb].keys())
        for chan, d in now[sb].items():
#             print(d.keys())
            if (d['R'][-1] < 5e-3):
                continue
            elif len(np.where(d['R'] > 10e-3)[0]) > 0:
                continue
            # elif len(np.where(d['R'] < -2e-4)[0]) > 10:
            #     continue
            now_chan = len(all_data[bl][sb].keys())
            all_data[bl][sb][now_chan] = d
            good_chans += 1



##70% Rn first
S.set_rtm_arb_waveform_enable(0)
S.set_filter_disable(0)
S.set_downsample_factor(20)
v_bias_all = [] 
RN = []
target_vbias_list = []
for bl in bias_groups:
    percent_rn = 0.7
    target_v_bias = []
    

    for band in [0,1,2,3,4,5,6,7]:
        try:

            for ch,d in all_data[bl][band].items():
                rn = d['R']/d['R_n']
                cross_idx = np.where(np.logical_and(rn - percent_rn >= 0, np.roll(rn - percent_rn, 1) < 0))[0]
                RN.append(d['R_n'])
                target_v_bias.append(d['v_bias'][cross_idx][0]) 
                v_bias_all.append(d['v_bias'][cross_idx][0])
        except:
            continue
# print(target_v_bias)
    med_target_v_bias = np.median(np.array(target_v_bias))
    if med_target_v_bias > 12:
        target_vbias_list.append(0)
    else:
        target_vbias_list.append(round(med_target_v_bias,1))
target_vbias_list = np.append(target_vbias_list,[0,0,0])



S.overbias_tes_all(bias_groups = [0,1,2,3,4,5,6,7,8,9,10,11], overbias_wait=1, tes_bias= 5, cool_wait= 3, high_current_mode=True, overbias_voltage= 5)
bias_array = np.array(target_vbias_list) / S.high_low_current_ratio
S.set_tes_bias_bipolar_array(bias_array)


print('waiting extra long for this heat to go away')
for i in range(36):
    time.sleep(10)

#switch to high current mode and diable all filters
print('preparing for bias step')
S.set_downsample_factor(1)
S.set_filter_disable(1)


step_size = 0.1 / S.high_low_current_ratio
bias_voltage = bias_array
dat_path = S.stream_data_on()
for k in [0,1]:
    S.set_tes_bias_bipolar_array(bias_array)
    time.sleep(2)
    S.set_tes_bias_bipolar_array(bias_array - step_size)
    time.sleep(2)
S.stream_data_off()

row = {}
row['bath_temp'] = bath_temp
row['data_path'] = dat_path
row['bias_voltage'] = str(S.get_tes_bias_bipolar_array())
row['type'] = 'bias step'
row['bias_line'] = 'all'
row['band'] = 'all'
row['note'] = '70 Rn step size {}'.format(step_size)
with open(out_fn, 'a', newline = '') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writerow(row)


step_size = 0.025 / S.high_low_current_ratio
bias_voltage = bias_array
dat_path = S.stream_data_on()
for k in [0,1]:
    S.set_tes_bias_bipolar_array(bias_array)
    time.sleep(2)
    S.set_tes_bias_bipolar_array(bias_array - step_size)
    time.sleep(2)
S.stream_data_off()

row = {}
row['bath_temp'] = bath_temp
row['data_path'] = dat_path
row['bias_voltage'] = str(S.get_tes_bias_bipolar_array())
row['type'] = 'bias step'
row['bias_line'] = 'all'
row['band'] = 'all'
row['note'] = '70 Rn step size {}'.format(step_size)
with open(out_fn, 'a', newline = '') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writerow(row)



step_size = 0.01 / S.high_low_current_ratio
bias_voltage = bias_array
dat_path = S.stream_data_on()
for k in [0,1]:
    S.set_tes_bias_bipolar_array(bias_array)
    time.sleep(2)
    S.set_tes_bias_bipolar_array(bias_array - step_size)
    time.sleep(2)
S.stream_data_off()

row = {}
row['bath_temp'] = bath_temp
row['data_path'] = dat_path
row['bias_voltage'] = str(S.get_tes_bias_bipolar_array())
row['type'] = 'bias step'
row['bias_line'] = 'all'
row['band'] = 'all'
row['note'] = '70 Rn step size {}'.format(step_size)
with open(out_fn, 'a', newline = '') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writerow(row)


S.set_rtm_arb_waveform_enable(0)
S.set_filter_disable(0)
S.set_downsample_factor(20)

#bias to low current mode target first
bias_groups = [0,1,2,3,4,5,6,7,8,9,10,11]
S.set_tes_bias_bipolar_array(target_vbias_list)
#immediately drop to low current
S.set_tes_bias_low_current(bias_groups)

# sleep for 1 mins
for i in range(6):
    time.sleep(10)


datafile_self = S.stream_data_on()
time.sleep(120)
S.stream_data_off()



row = {}
row['data_path'] = datafile_self
row['bias_voltage'] = str(S.get_tes_bias_bipolar_array())
row['type'] = '70 percent noise low current mode'
row['bias_line'] = 'all'
row['band'] = 'all'
row['bath_temp'] = bath_temp
with open(out_fn, 'a', newline = '') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writerow(row)


##50% Rn
S.set_rtm_arb_waveform_enable(0)
S.set_filter_disable(0)
S.set_downsample_factor(20)
v_bias_all = [] 
RN = []
target_vbias_list = []
for bl in bias_groups:
    percent_rn = 0.5
    target_v_bias = []
    

    for band in [0,1,2,3,4,5,6,7]:
        try:

            for ch,d in all_data[bl][band].items():
                rn = d['R']/d['R_n']
                cross_idx = np.where(np.logical_and(rn - percent_rn >= 0, np.roll(rn - percent_rn, 1) < 0))[0]
                RN.append(d['R_n'])
                target_v_bias.append(d['v_bias'][cross_idx][0]) 
                v_bias_all.append(d['v_bias'][cross_idx][0])
        except:
            continue
# print(target_v_bias)
    med_target_v_bias = np.median(np.array(target_v_bias))
    if med_target_v_bias > 12:
        target_vbias_list.append(0)
    else:
        target_vbias_list.append(round(med_target_v_bias,1))
target_vbias_list = np.append(target_vbias_list,[0,0,0])



S.overbias_tes_all(bias_groups = [0,1,2,3,4,5,6,7,8,9,10,11], overbias_wait=1, tes_bias= 5, cool_wait= 3, high_current_mode=True, overbias_voltage= 5)
bias_array = np.array(target_vbias_list) / S.high_low_current_ratio
S.set_tes_bias_bipolar_array(bias_array)


print('waiting extra long for this heat to go away')
for i in range(36):
    time.sleep(10)

#switch to high current mode and diable all filters
print('preparing for bias step')
S.set_downsample_factor(1)
S.set_filter_disable(1)



step_size = 0.1 / S.high_low_current_ratio
bias_voltage = bias_array
dat_path = S.stream_data_on()
for k in [0,1]:
    S.set_tes_bias_bipolar_array(bias_array)
    time.sleep(2)
    S.set_tes_bias_bipolar_array(bias_array - step_size)
    time.sleep(2)
S.stream_data_off()

row = {}
row['bath_temp'] = bath_temp
row['data_path'] = dat_path
row['bias_voltage'] = str(S.get_tes_bias_bipolar_array())
row['type'] = 'bias step'
row['bias_line'] = 'all'
row['band'] = 'all'
row['note'] = '50 Rn step size {}'.format(step_size)
with open(out_fn, 'a', newline = '') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writerow(row)




step_size = 0.025 / S.high_low_current_ratio
bias_voltage = bias_array
dat_path = S.stream_data_on()
for k in [0,1]:
    S.set_tes_bias_bipolar_array(bias_array)
    time.sleep(2)
    S.set_tes_bias_bipolar_array(bias_array - step_size)
    time.sleep(2)
S.stream_data_off()

row = {}
row['bath_temp'] = bath_temp
row['data_path'] = dat_path
row['bias_voltage'] = str(S.get_tes_bias_bipolar_array())
row['type'] = 'bias step'
row['bias_line'] = 'all'
row['band'] = 'all'
row['note'] = '50 Rn step size {}'.format(step_size)
with open(out_fn, 'a', newline = '') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writerow(row)



step_size = 0.01 / S.high_low_current_ratio
bias_voltage = bias_array
dat_path = S.stream_data_on()
for k in [0,1]:
    S.set_tes_bias_bipolar_array(bias_array)
    time.sleep(2)
    S.set_tes_bias_bipolar_array(bias_array - step_size)
    time.sleep(2)
S.stream_data_off()

row = {}
row['bath_temp'] = bath_temp
row['data_path'] = dat_path
row['bias_voltage'] = str(S.get_tes_bias_bipolar_array())
row['type'] = 'bias step'
row['bias_line'] = 'all'
row['band'] = 'all'
row['note'] = '50 Rn step size {}'.format(step_size)
with open(out_fn, 'a', newline = '') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writerow(row)


S.set_rtm_arb_waveform_enable(0)
S.set_filter_disable(0)
S.set_downsample_factor(20)

#bias to low current mode target first
bias_groups = [0,1,2,3,4,5,6,7,8,9,10,11]
S.set_tes_bias_bipolar_array(target_vbias_list)
#immediately drop to low current
S.set_tes_bias_low_current(bias_groups)

# sleep for 5 mins
for i in range(36):
    time.sleep(10)


datafile_self = S.stream_data_on()
time.sleep(120)
S.stream_data_off()



row = {}
row['data_path'] = datafile_self
row['bias_voltage'] = str(S.get_tes_bias_bipolar_array())
row['type'] = '50 percent noise low current mode'
row['bias_line'] = 'all'
row['band'] = 'all'
row['bath_temp'] = bath_temp
with open(out_fn, 'a', newline = '') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writerow(row)



##30% Rn
S.set_rtm_arb_waveform_enable(0)
S.set_filter_disable(0)
S.set_downsample_factor(20)
v_bias_all = [] 
RN = []
target_vbias_list = []
for bl in bias_groups:
    percent_rn = 0.3
    target_v_bias = []
    

    for band in [0,1,2,3,4,5,6,7]:
        try:

            for ch,d in all_data[bl][band].items():
                rn = d['R']/d['R_n']
                cross_idx = np.where(np.logical_and(rn - percent_rn >= 0, np.roll(rn - percent_rn, 1) < 0))[0]
                RN.append(d['R_n'])
                target_v_bias.append(d['v_bias'][cross_idx][0]) 
                v_bias_all.append(d['v_bias'][cross_idx][0])
        except:
            continue
# print(target_v_bias)
    med_target_v_bias = np.median(np.array(target_v_bias))
    if med_target_v_bias > 12:
        target_vbias_list.append(0)
    else:
        target_vbias_list.append(round(med_target_v_bias,1))
target_vbias_list = np.append(target_vbias_list,[0,0,0])



S.overbias_tes_all(bias_groups = [0,1,2,3,4,5,6,7,8,9,10,11], overbias_wait=1, tes_bias= 5, cool_wait= 3, high_current_mode=True, overbias_voltage= 5)
bias_array = np.array(target_vbias_list) / S.high_low_current_ratio
S.set_tes_bias_bipolar_array(bias_array)


print('waiting extra long for this heat to go away')
for i in range(30):
    time.sleep(10)

#switch to high current mode and diable all filters
print('preparing for bias step')
S.set_downsample_factor(1)
S.set_filter_disable(1)



step_size = 0.1 / S.high_low_current_ratio
bias_voltage = bias_array
dat_path = S.stream_data_on()
for k in [0,1]:
    S.set_tes_bias_bipolar_array(bias_array)
    time.sleep(1)
    S.set_tes_bias_bipolar_array(bias_array - step_size)
    time.sleep(1)
S.stream_data_off()

row = {}
row['bath_temp'] = bath_temp
row['data_path'] = dat_path
row['bias_voltage'] = str(S.get_tes_bias_bipolar_array())
row['type'] = 'bias step'
row['bias_line'] = 'all'
row['band'] = 'all'
row['note'] = '30 Rn step size {}'.format(step_size)
with open(out_fn, 'a', newline = '') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writerow(row)



step_size = 0.025 / S.high_low_current_ratio
bias_voltage = bias_array
dat_path = S.stream_data_on()
for k in [0,1]:
    S.set_tes_bias_bipolar_array(bias_array)
    time.sleep(2)
    S.set_tes_bias_bipolar_array(bias_array - step_size)
    time.sleep(2)
S.stream_data_off()

row = {}
row['bath_temp'] = bath_temp
row['data_path'] = dat_path
row['bias_voltage'] = str(S.get_tes_bias_bipolar_array())
row['type'] = 'bias step'
row['bias_line'] = 'all'
row['band'] = 'all'
row['note'] = '30 Rn step size {}'.format(step_size)
with open(out_fn, 'a', newline = '') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writerow(row)


step_size = 0.01 / S.high_low_current_ratio
bias_voltage = bias_array
dat_path = S.stream_data_on()
for k in [0,1]:
    S.set_tes_bias_bipolar_array(bias_array)
    time.sleep(2)
    S.set_tes_bias_bipolar_array(bias_array - step_size)
    time.sleep(2)
S.stream_data_off()

row = {}
row['bath_temp'] = bath_temp
row['data_path'] = dat_path
row['bias_voltage'] = str(S.get_tes_bias_bipolar_array())
row['type'] = 'bias step'
row['bias_line'] = 'all'
row['band'] = 'all'
row['note'] = '30 Rn step size {}'.format(step_size)
with open(out_fn, 'a', newline = '') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writerow(row)














S.set_rtm_arb_waveform_enable(0)
S.set_filter_disable(0)
S.set_downsample_factor(20)

#bias to low current mode target first
bias_groups = [0,1,2,3,4,5,6,7,8,9,10,11]
S.set_tes_bias_bipolar_array(target_vbias_list)
#immediately drop to low current
S.set_tes_bias_low_current(bias_groups)

# sleep for 1 mins
for i in range(30):
    time.sleep(10)


datafile_self = S.stream_data_on()
time.sleep(120)
S.stream_data_off()



row = {}
row['data_path'] = datafile_self
row['bias_voltage'] = str(S.get_tes_bias_bipolar_array())
row['type'] = '30 percent noise low current mode'
row['bias_line'] = 'all'
row['band'] = 'all'
row['bath_temp'] = bath_temp
with open(out_fn, 'a', newline = '') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writerow(row)










#turn filter back on and sample rate into 200Hz
S.set_rtm_arb_waveform_enable(0)
S.set_filter_disable(0)
S.set_downsample_factor(20)



