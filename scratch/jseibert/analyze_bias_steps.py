import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pysmurf.client
import argparse
import numpy as np
import os
import time
import glob
from scipy import signal
import scipy.optimize as opt
pi = np.pi

if __name__=='__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--setup', action='store_true')
    parser.add_argument('--config-file', required=True)
    parser.add_argument('--epics-root', default='smurf_server_s2')

    parser.add_argument('--out-file')
    parser.add_argument('--plot-dir')
    parser.add_argument('--out_dir')

    parser.add_argument('--band',type=int, required=True)
    parser.add_argument('--chans',required=True,type=int,nargs = '+')

    args = parser.parse_args() 

    out_file = args.out_file
    plot_dir = args.plot_dir
    out_dir = args.out_dir
    
    band = args.band
    chans = args.chans
    
    # Initialize pysmurf object
    S = pysmurf.client.SmurfControl(
            epics_root = args.epics_root,
            cfg_file = args.config_file,
            setup = args.setup,make_logfile=False

    # Use metadata from when bias steps were taken
    metadata = np.genfromtxt(out_file,dtype='str',delimiter=',',skipheader=1)

    bias_points = metadata[:,0]
    bias_points = bias_points.astype('float')
    datfiles = metadata[:,1]
    output_dirs = metadata[:,2]
    sample_rate = metadata[:,3]

    # Sample rate for all steps should've been the same, so just pull the first one
    fs = sample_rate[0].astype('float')

    ctime = S.get_timestamp()

    tau_dict = {}

    # Do this analysis for all channels that were called
    for ch in chans:
        tau_dict[ch] = {}
        for i,bias in enumerate(bias_points):
            datfile = datfiles[i]
            timestamp, phase, mask = S.read_stream_data(datafile=datfile)
            
            #Identify each of the falling edges

            #Initialize phase array
            p = phase[mask[band,ch]]
            p = p - np.min(p)
            t = np.arange(0,len(p),1)/fs
            min_idx_tot_array = []

            #Find how long after data taking starts the first step occurs
            chng_idx = np.where(abs(np.diff(p)) > 0.02)[0][0]

            #Take the location of the first step and add 0.1 sec 
            start_sec = t[chng_idx] + 0.1
            print(start_sec)

            #Make sure that the step is always a positive step, for analysis purposes
            if p[int(start_sec*fs)] < p[0]:
                p*= -1

            n_steps = 8
            
            plt.figure()

            for step in range(n_steps):
                if step == 0:
                    idx_step = int(start_sec*fs)
                    loc_phase_first_step = p[idx_step:idx_step+int(0.375*fs)]
                    idx_diff_min = np.argmin(np.diff(
                        signal.savgol_filter(loc_phase_first_step,11,3)))
                    p_target = loc_phase_first_step[idx_diff_min]
                    plt.plot(t[idx_step:idx_step+int(0.375*fs)],
                            p[idx_step:idx_step+int(0.375*fs)])
                    min_idx_tot_array.append(idx_step + idx_diff_min)
                else:
                    idx_step = int((start_sec + (step)*0.75)*fs)
                    loc_phase_first_step = p[idx_step:idx_step+int(0.375*fs)]
                    idx_diff_min = np.argmin(np.diff(
                        signal.savgol_filter(loc_phase_first_step,11,3)))
                    p_target = loc_phase_first_step[idx_diff_min]
                    plt.plot(t[idx_step:idx_step+int(0.375*fs)],
                            p[idx_step:idx_step+int(0.375*fs)])
                    min_idx_tot_array.append(idx_step + idx_diff_min)
            
            plt.plot(t,p,alpha = 0.5)
            plt.plot(t[min_idx_tot_array],p[min_idx_tot_array],'ro')
            plt.savefig(os.path.join(plot_dir,f'{ctime}_b{band}_c{ch}_bias{bias}_steps.png'))
            plt.close()
            
            plt.figure()
            #grab all of the steps and average them
            p_chunks = []
            print(n_steps)
            step_dur = 0.75
            for step in range(n_steps):
                p_step = p[min_idx_tot_array[step]-int((step_dur/5)*fs):min_idx_tot_array[step]+int((step_dur/5)*fs)]
                p_step = p_step - p_step[-1]
                if p_step[0] < p_step[-1]:
                    p_step*= -1
                p_chunks.append(p_step)
                plt.plot(np.arange(0,len(p_step),1)*fs,p_step,'--',alpha = 0.6)
            p_sum = np.sum(np.asarray(p_chunks),axis = 0)/n_steps
            print(np.shape(np.asarray(p_chunks)))
            plt.plot(np.arange(0,len(p_sum),1)*fs,p_sum,'r-',linewidth = 2)
            plt.xlabel('samples',fontsize = 16)
            plt.ylabel('Response',fontsize = 16)
            fig = plt.gcf()
            fig.set_size_inches(30,15)

            plt.savefig(os.path.join(plot_dir,f'{ctime}_b{band}_c{ch}_bias{bias}_averaged_steps.png'))
            plt.close()
            
            #Try to fit each chunk instead of the averaged step
            def fall_fn(t, a,tau):
                return a*np.exp(-t/tau)
            taus_fit = []
            pcovs = []
            plot_num = 0
            for p_c in p_chunks:
                p_c = p_c - np.mean(p_c[-400::])
                t_c = np.arange(0,len(p_c),1)/fs
                #p_temp = p_c[len(p_c)//2+15:len(p_c)//2+100]
                p_temp = p_c[len(p_c)//2-4:-50]
                p_temp = p_temp - p_temp[-1]
                t_temp = np.arange(0,len(p_temp),1)/fs

                #Mask out the overshoot
                idx_overshoot = np.argmin(p_temp)
                r = 1
                p_temp = np.concatenate((p_temp[0:idx_overshoot-r],p_temp[idx_overshoot+r:-1]))
                t_temp = np.concatenate((t_temp[0:idx_overshoot-r],t_temp[idx_overshoot+r:-1]))

                popt,pcov = opt.curve_fit(
                fall_fn, t_temp, p_temp, maxfev = 3000,p0 = [p_temp[0],.005]
                )
                print(popt)
                pcovs.append(pcov[1,1])
                taus_fit.append(popt[1])
                t_plot = np.arange(t_temp[0],t_temp[-1],1/fs)
                plt.figure()
                plt.semilogx(t_temp,p_temp,'o',label = f'data, {pcov[1,1]}')
                plt.semilogx(t_plot,fall_fn(t_plot,popt[0],popt[1]),'r--',
                            label = f'tau = {np.round(popt[1]/1e-3,2)} ms, $\sigma$ = {np.round(np.sqrt(pcov[1,1])/1e-3,3)}ms')
                plt.legend()
                plt.savefig(os.path.join(plot_dir,f'{ctime}_b{band}_c{ch}_bias{bias}_fitted_step_{plot_num}.png'))
                plt.close()

                plot_num += 1 


            taus_fit = np.asarray(taus_fit)
            pcovs = np.asarray(pcovs)
            idx_avg = np.argsort(pcovs)

            tau_dict[ch][bias] = taus_fit

    np.save(os.path.join(out_dir,f'{ctime}_taus_fit.npy'),tau_dict)