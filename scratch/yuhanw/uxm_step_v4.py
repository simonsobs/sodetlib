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
    fav_tune_files = '/data/smurf_data/tune/1626808976_tune.npy'
if slot_num == 4:
    fav_tune_files = '/data/smurf_data/tune/1626804769_tune.npy'


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
    # S.check_lock(band,reset_rate_khz=cfg.dev.bands[band]['flux_ramp_rate_khz'],fraction_full_scale=cfg.dev.bands[band]['frac_pp'], lms_freq_hz=None, feedback_start_frac=cfg.dev.bands[band]['feedback_start_frac'],feedback_end_frac=cfg.dev.bands[band]['feedback_end_frac'],lms_gain=cfg.dev.bands[band]['lms_gain'])


# for band in [0,1,2,3,4,5,6,7]:
#     S.run_serial_gradient_descent(band);
#     S.run_serial_eta_scan(band);
#     S.set_feedback_enable(band,1) 
#     S.tracking_setup(band,reset_rate_khz=cfg.dev.bands[band]['flux_ramp_rate_khz'],fraction_full_scale=cfg.dev.bands[band]['frac_pp'], make_plot=False, save_plot=False, show_plot=False, channel=S.which_on(band), nsamp=2**18, lms_freq_hz=None, meas_lms_freq=True,feedback_start_frac=cfg.dev.bands[band]['feedback_start_frac'],feedback_end_frac=cfg.dev.bands[band]['feedback_end_frac'],lms_gain=cfg.dev.bands[band]['lms_gain'])
    


fieldnames = ['bath_temp','bias_voltage', 'bias_line', 'band', 'data_path','note']

# with open(out_fn, 'w', newline = '') as csvfile:
#     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#     writer.writeheader()
 



print('SC nosie')

bias_v = 0
S.set_tes_bias_bipolar_array([bias_v,bias_v,bias_v,bias_v,bias_v,bias_v,bias_v,bias_v,bias_v,bias_v,bias_v,bias_v,0,0,0])
time.sleep(60)
datafile_self = S.stream_data_on()
time.sleep(120)
S.stream_data_off()


row = {}
row['data_path'] = datafile_self
row['bias_voltage'] = bias_v
row['note'] = 'noise'
row['bias_line'] = 'all'
row['band'] = 'all'
row['bath_temp'] = bath_temp
with open(out_fn, 'a', newline = '') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writerow(row)







    
bias_groups = [0,1,2,3,4,5,6,7,8,9,10,11]

bl_iv_list = []
for bias_gp in bias_groups:
        row = {}
        row['bath_temp'] = bath_temp
        row['bias_voltage'] = 'iv'
        row['bias_line'] = bias_gp
        row['band'] = 'all'
        row['note'] = 'iv'    
        print(f'Taking IV on bias line {bias_gp}, all band')
          
        
        iv_data = S.run_iv(bias_groups = [bias_gp], wait_time=0.001, bias_high=16, bias_low=0, bias_step = 0.025, overbias_voltage=18, cool_wait=10, high_current_mode=False, make_plot=False, save_plot=True, cool_voltage = 8)
        dat_file = iv_data[0:-13]+'.npy'
        row['data_path'] = dat_file
        bl_iv_list.append(dat_file)
        with open(out_fn, 'a', newline = '') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow(row)




 
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
            elif len(np.where(d['R'] < -2e-4)[0]) > 10:
                continue
            now_chan = len(all_data[bl][sb].keys())
            all_data[bl][sb][now_chan] = d
            good_chans += 1

print('overall good iv {}'.format(good_chans))

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
    target_vbias_list.append(round(med_target_v_bias,1))
    
target_vbias_list = np.append(target_vbias_list,[0,0,0])
target_50_list = target_vbias_list
row = {}
row['bath_temp'] = bath_temp
row['bias_voltage'] = '50 Rn'
row['bias_line'] = 'all'
row['band'] = 'all'
row['note'] = target_vbias_list
row['data_path'] = 'none'
with open(out_fn, 'a', newline = '') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writerow(row)


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
    target_vbias_list.append(round(med_target_v_bias,1))
target_vbias_list = np.append(target_vbias_list,[0,0,0])
target_30_list = target_vbias_list
row = {}
row['bath_temp'] = bath_temp
row['bias_voltage'] = '30 Rn'
row['bias_line'] = 'all'
row['band'] = 'all'
row['note'] = target_vbias_list
row['data_path'] = 'none'
with open(out_fn, 'a', newline = '') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writerow(row)




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
    target_vbias_list.append(round(med_target_v_bias,1))
target_vbias_list = np.append(target_vbias_list,[0,0,0])
target_70_list = target_vbias_list
row = {}
row['bath_temp'] = bath_temp
row['bias_voltage'] = '70 Rn'
row['bias_line'] = 'all'
row['band'] = 'all'
row['note'] = target_vbias_list
row['data_path'] = 'none'
with open(out_fn, 'a', newline = '') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writerow(row)

for band in [0,1,2,3,4,5,6,7]:
    S.run_serial_gradient_descent(band);
    S.run_serial_eta_scan(band);
    S.set_feedback_enable(band,1) 
    S.tracking_setup(band,reset_rate_khz=cfg.dev.bands[band]['flux_ramp_rate_khz'],fraction_full_scale=cfg.dev.bands[band]['frac_pp'], make_plot=False, save_plot=False, show_plot=False, channel=S.which_on(band), nsamp=2**18, lms_freq_hz=None, meas_lms_freq=True,feedback_start_frac=cfg.dev.bands[band]['feedback_start_frac'],feedback_end_frac=cfg.dev.bands[band]['feedback_end_frac'],lms_gain=cfg.dev.bands[band]['lms_gain'])
    


print('start taking biasstep 70')
S.overbias_tes_all(bias_groups = bias_groups, overbias_wait=1, tes_bias= 2, cool_wait= 3, high_current_mode=True, overbias_voltage= 12)
S.set_tes_bias_bipolar_array(np.array(target_70_list)/S.high_low_current_ratio)
print('waiting extra long for this heat to go away')
time.sleep(420)
for bias_index, bias_g in enumerate(bias_groups):
    bias_voltage = target_70_list[bias_index]
    step_size = 0.05

    print(S.get_tes_bias_bipolar_array())
    print('equivalent in low current mode:')
    print(S.get_tes_bias_bipolar_array()*S.high_low_current_ratio)
   
  
    S.set_downsample_factor(1)
    print(S.get_sample_frequency())
    S.set_filter_disable(1)
   
    # Defining wave form
    signal = np.ones(2048)
    
    signal *= bias_voltage / (2*S._rtm_slow_dac_bit_to_volt*S.high_low_current_ratio) #defining the bias step level (lower step)
    signal[1024:] += step_size / (2*S._rtm_slow_dac_bit_to_volt*S.high_low_current_ratio) #defining the bias step level (upper step)
    ts = int(2/(6.4e-9 * 2048))
    S.set_rtm_arb_waveform_timer_size(ts, wait_done = True)
   
   
    S.play_tes_bipolar_waveform(bias_g,signal)
    datafile_self = S.stream_data_on()
    time.sleep(4)
    S.stream_data_off()

    
     
    S.set_rtm_arb_waveform_enable(0)
    S.set_filter_disable(0)
    S.set_downsample_factor(20)

    row = {}
    row['bath_temp'] = bath_temp
    row['data_path'] = datafile_self
    row['bias_voltage'] = bias_voltage
    row['bias_line'] = bias_g
    row['band'] = 'all'
    row['note'] = '70 Rn'
    with open(out_fn, 'a', newline = '') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow(row)





print('start taking biasstep 50')
S.set_tes_bias_bipolar_array(np.array(target_50_list)/S.high_low_current_ratio)
print('waiting for this heat to go away')
time.sleep(20)
for bias_index, bias_g in enumerate(bias_groups):
    bias_voltage = target_50_list[bias_index]
    step_size = 0.05

    print(S.get_tes_bias_bipolar_array())
    print('equivalent in low current mode:')
    print(S.get_tes_bias_bipolar_array()*S.high_low_current_ratio)
   
  
    S.set_downsample_factor(1)
    print(S.get_sample_frequency())
    S.set_filter_disable(1)
   
    # Defining wave form
    signal = np.ones(2048)
    
    signal *= bias_voltage / (2*S._rtm_slow_dac_bit_to_volt*S.high_low_current_ratio) #defining the bias step level (lower step)
    signal[1024:] += step_size / (2*S._rtm_slow_dac_bit_to_volt*S.high_low_current_ratio) #defining the bias step level (upper step)
    ts = int(2/(6.4e-9 * 2048))
    S.set_rtm_arb_waveform_timer_size(ts, wait_done = True)
   
   
    S.play_tes_bipolar_waveform(bias_g,signal)
    datafile_self = S.stream_data_on()
    time.sleep(4)
    S.stream_data_off()

    
     
    S.set_rtm_arb_waveform_enable(0)
    S.set_filter_disable(0)
    S.set_downsample_factor(20)

    row = {}
    row['bath_temp'] = bath_temp
    row['data_path'] = datafile_self
    row['bias_voltage'] = bias_voltage
    row['bias_line'] = bias_g
    row['band'] = 'all'
    row['note'] = '50 Rn'
    with open(out_fn, 'a', newline = '') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow(row)



print('start taking biasstep 30')
S.set_tes_bias_bipolar_array(np.array(target_30_list)/S.high_low_current_ratio)
print('waiting for this heat to go away')
time.sleep(20)
for bias_index, bias_g in enumerate(bias_groups):
    bias_voltage = target_30_list[bias_index]
    step_size = 0.05

    print(S.get_tes_bias_bipolar_array())
    print('equivalent in low current mode:')
    print(S.get_tes_bias_bipolar_array()*S.high_low_current_ratio)
   
  
    S.set_downsample_factor(1)
    print(S.get_sample_frequency())
    S.set_filter_disable(1)
   
    # Defining wave form
    signal = np.ones(2048)
    
    signal *= bias_voltage / (2*S._rtm_slow_dac_bit_to_volt*S.high_low_current_ratio) #defining the bias step level (lower step)
    signal[1024:] += step_size / (2*S._rtm_slow_dac_bit_to_volt*S.high_low_current_ratio) #defining the bias step level (upper step)
    ts = int(2/(6.4e-9 * 2048))
    S.set_rtm_arb_waveform_timer_size(ts, wait_done = True)
   
   
    S.play_tes_bipolar_waveform(bias_g,signal)
    datafile_self = S.stream_data_on()
    time.sleep(4)
    S.stream_data_off()

    
     
    S.set_rtm_arb_waveform_enable(0)
    S.set_filter_disable(0)
    S.set_downsample_factor(20)

    row = {}
    row['bath_temp'] = bath_temp
    row['data_path'] = datafile_self
    row['bias_voltage'] = bias_voltage
    row['bias_line'] = bias_g
    row['band'] = 'all'
    row['note'] = '30 Rn'
    with open(out_fn, 'a', newline = '') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow(row)


#switch to low current mode to take noise data
S.set_filter_disable(0)
S.set_rtm_arb_waveform_enable(0)
S.set_downsample_factor(20)
for bias_index, bias_g in enumerate(bias_groups):
    S.set_tes_bias_low_current(bias_g)

S.overbias_tes_all(bias_groups = bias_groups, overbias_wait=1, tes_bias= 12, cool_wait= 6, high_current_mode=False, overbias_voltage= 12)
bias_v = 12
S.set_tes_bias_bipolar_array([bias_v,bias_v,bias_v,bias_v,bias_v,bias_v,bias_v,bias_v,bias_v,bias_v,bias_v,bias_v,0,0,0])
datafile_self = S.stream_data_on()
time.sleep(120)
S.stream_data_off()

row = {}
row['data_path'] = datafile_self
row['bias_voltage'] = bias_v
row['note'] = 'noise'
row['bias_line'] = 'all'
row['band'] = 'all'
row['bath_temp'] = bath_temp
with open(out_fn, 'a', newline = '') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writerow(row)


S.set_tes_bias_bipolar_array(np.array(target_70_list))

time.sleep(600)
datafile_self = S.stream_data_on()
time.sleep(300)
S.stream_data_off()

row = {}
row['data_path'] = datafile_self
row['bias_voltage'] = target_70_list
row['note'] = 'noise'
row['bias_line'] = 'all'
row['band'] = 'all'
row['bath_temp'] = bath_temp
with open(out_fn, 'a', newline = '') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writerow(row)


S.set_tes_bias_bipolar_array(np.array(target_50_list))

time.sleep(240)
datafile_self = S.stream_data_on()
time.sleep(300)
S.stream_data_off()

row = {}
row['data_path'] = datafile_self
row['bias_voltage'] = target_50_list
row['note'] = 'noise'
row['bias_line'] = 'all'
row['band'] = 'all'
row['bath_temp'] = bath_temp
with open(out_fn, 'a', newline = '') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writerow(row)


S.set_tes_bias_bipolar_array(np.array(target_30_list))

time.sleep(240)
datafile_self = S.stream_data_on()
time.sleep(300)
S.stream_data_off()

row = {}
row['data_path'] = datafile_self
row['bias_voltage'] = target_30_list
row['note'] = 'noise'
row['bias_line'] = 'all'
row['band'] = 'all'
row['bath_temp'] = bath_temp
with open(out_fn, 'a', newline = '') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writerow(row)


#sine wave in

S.overbias_tes_all(bias_groups = bias_groups, overbias_wait=1, tes_bias= 2, cool_wait= 3, high_current_mode=False, overbias_voltage= 12)
S.set_tes_bias_bipolar_array(np.array(target_50_list))
print('waiting extra long for this heat to go away')
time.sleep(120)
for bias_index, bias_g in enumerate(bias_groups):
    bias_voltage = target_50_list[bias_index]
    step_size = 0.05

    S.set_downsample_factor(1)
    print(S.get_sample_frequency())
    S.set_filter_disable(1)

    print(S.get_tes_bias_bipolar_array())
    freq = 50
    period = 1/freq
    step_size = 0.05
    signal = np.ones(2048)
    S.get_tes_bias_bipolar(bias_g)
    for i in range(len(signal)):
        signal[i] *= (step_size / (2*S._rtm_slow_dac_bit_to_volt))*np.sin(4*np.pi*i/(1024))+bias_voltage / (2*S._rtm_slow_dac_bit_to_volt) + (step_size / (2*S._rtm_slow_dac_bit_to_volt))*np.sin(4*np.pi*i/(1024)*2)+bias_voltage / (2*S._rtm_slow_dac_bit_to_volt) + + (step_size / (2*S._rtm_slow_dac_bit_to_volt))*np.sin(4*np.pi*i/(1024)*4)+bias_voltage / (2*S._rtm_slow_dac_bit_to_volt)
    ts = int(period*4/(6.4e-9 * 2048))
    S.set_rtm_arb_waveform_timer_size(ts, wait_done = True)
    S.set_rtm_arb_waveform_enable(1)
    S.play_tes_bipolar_waveform(bias_g,signal)
       
    datafile_self = S.stream_data_on()
    time.sleep(4)
    S.stream_data_off()

    
     
    S.set_rtm_arb_waveform_enable(0)
    S.set_filter_disable(0)
    S.set_downsample_factor(20)

    row = {}
    row['bath_temp'] = bath_temp
    row['data_path'] = datafile_self
    row['bias_voltage'] = bias_voltage
    row['bias_line'] = bias_g
    row['band'] = 'all'
    row['note'] = '50 Rn sine 50 100 200 Hz'
    with open(out_fn, 'a', newline = '') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow(row)















