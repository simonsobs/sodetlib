import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import scipy

# below from Kaiwen Zheng
#########################
def read_vna_data(filename):
	# Reads vna data in s2p or csv format
	# outputs frequency, real and imaginary parts
	# You should use the function below instead.
    if filename.endswith('S2P'):
        s2pdata = rf.Network(filename)
        freq=np.array(s2pdata.frequency.f)
        real=np.squeeze(s2pdata.s21.s_re)
        imag=np.squeeze(s2pdata.s21.s_im)
    elif filename.endswith('CSV'):
        csvdata=pd.read_csv(filename,header=2)
        freq=np.array(csvdata['Frequency'])
        real=np.array(csvdata[' Formatted Data'])
        imag=np.array(csvdata[' Formatted Data.1'])
    else:
        freq=0;real=0;imag=0
        print('invalid file type')
    return freq,real,imag

def read_vna_data_array(filenames):
	# Input an array of vna filenames or just one file
	# Outputs all data in the file, organized by frequency
    if np.array([filenames]).size==1:
        freq,real,imag=read_vna_data(filenames)
    elif np.array([filenames]).size>1:
            freq=np.array([])
            real=np.array([])
            imag=np.array([])
            for onefile in list(filenames):
                ft,rt,it=read_vna_data(onefile)
                freq=np.append(freq,ft)
                real=np.append(real,rt)
                imag=np.append(imag,it)
    L=sorted(zip(freq,real,imag))
    f,r,i=zip(*L)
    return np.array(f),np.array(r),np.array(i)
    #return freq,real,imag

def s21_find_baseline(fs, s21, avg_over=800):
    #freqsarr and s21arr are your frequency and transmission
    #average the data every avg_over points to find the baseline
    #of s21.
    #written by Heather, modified so that number of datapoints
    #doesn't have to be multiples of avg_over.
    num_points_all = s21.shape[0]
    num_2 = num_points_all%avg_over
    num_1= num_points_all-num_2
    s21_reshaped = s21[:num_1].reshape(num_1//avg_over, avg_over)
    fs_reshaped = fs[:num_1].reshape(num_1//avg_over, avg_over)
    #s21_avg = s21_reshaped.mean(1)
    #fs_avg = fs_reshaped.mean(1)
    x = np.squeeze(np.median(fs_reshaped, axis=1))
    y = np.squeeze(np.amax(s21_reshaped, axis=1))
    if (num_2 !=0):
        x2=np.median(fs[num_1:num_points_all])
        y2=np.amax(s21[num_1:num_points_all])
        x=np.append(x,x2)
        y=np.append(y,y2)
    tck = scipy.interpolate.splrep(x, y, s=0)
    ynew = scipy.interpolate.splev(fs, tck, der=0)
    return ynew

def correct_trend(freq,real,imag,avg_over=800):
	#Input the real and imaginary part of a s21
	#Out put the s21 in db, without the trend. 
    s21=real+1j*imag
    s21_db=20*np.log10(np.abs(s21))
    baseline=s21_find_baseline(freq, s21, avg_over)
    bl_db=20*np.log10(baseline)
    s21_corrected=s21_db-bl_db
    return s21_corrected

def read_smurf_tuning_data(filename):
	# Reads the smurf file and extract frequency,
	# complex s21 and resonator index.
	# You should use the function below instead.
    dres = {'frequency':[], 'response':[],'index':[]}
    dfres =pd.DataFrame(dres)
    data=np.load(filename,allow_pickle=True).item()
    for band in list(data.keys()):
        if 'resonances' in list(data[band].keys()):
            for idx in list(data[band]['resonances'].keys()):
                scan=data[band]['resonances'][idx]
                f=np.array(scan['freq_eta_scan'])
                s21=np.array(scan['resp_eta_scan'])
                res_index = scan['channel']+band*512
                dfres = dfres.append({'frequency':f, 'response':s21,'index':res_index},ignore_index=True)
    return dfres

def read_smurf_tuning_data_array(filenames):
	# Reads one or an array of smurf files.
    if np.array([filenames]).size==1:
        frame=read_smurf_tuning_data(filenames)
    elif np.array([filenames]).size>1:
            frame=pd.DataFrame({'frequency':[], 'response':[],'index':[]})
            for onefile in list(filenames):
                f=read_smurf_tuning_data(onefile)
                frame=frame.append(f,ignore_index=True)
    return frame
##############################################################
plt.figure(figsize=(32,8))
plt.xlim(5.03e9,5.15e9)
offset = -7.5e6

fb_smurf = '/home/zatkins/repos/stslab/testbed_optics/deteff/data/assemblies/Sv5/tune_south/'
smurf_fns = ['1605204576_tune.npy']
s_fns = [fb_smurf + fn for fn in smurf_fns]

fb_vna = '/home/zatkins/so/data/vna_data/v5b_data.000/'
vna_fns = [
    '1595872591.84-F_4.00_TO_4.20-BW_100.0-ATTEN_15.0-VOLTS_0.000.CSV',
    '1595872800.63-F_4.20_TO_4.40-BW_100.0-ATTEN_15.0-VOLTS_0.000.CSV',
    '1595873009.44-F_4.40_TO_4.60-BW_100.0-ATTEN_15.0-VOLTS_0.000.CSV',
    '1595873218.25-F_4.60_TO_4.80-BW_100.0-ATTEN_15.0-VOLTS_0.000.CSV',
    '1595873427.05-F_4.80_TO_5.00-BW_100.0-ATTEN_15.0-VOLTS_0.000.CSV',
    '1595873635.82-F_5.00_TO_5.20-BW_100.0-ATTEN_10.0-VOLTS_0.000.CSV'
]
v_fns = [fb_vna + fn for fn in vna_fns]

fv,r,i = read_vna_data_array(v_fns)
vna = 10*np.log(np.abs(r**2 + i**2))
plt.plot(fv+offset,vna-32, label = f'UMM {offset/1e6} MHz')

frame = read_smurf_tuning_data_array(s_fns[0])
for i in np.arange(len(frame)):
    plt.plot(frame['frequency'][i]*1e6, 20*np.log10(np.abs(frame['response'][i])), 'C1')

plt.legend()