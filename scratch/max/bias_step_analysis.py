import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import pysmurf.client
import argparse
import numpy as np
import os
import time
import glob
import scipy.signal as sig
import scipy.optimize as opt
import pickle as pkl
pi = np.pi

#Exponential fall function to fit
def fall_fn(t, a,tau):
    return a*np.exp(-t/tau)

def analyze_tau_data(datatxtfile, bands = [2],channels = None,step_size = 0.004,
                     step_dur = 0.2, noise_dur = 1, n_steps = 15,fs = None,
                     make_debug_plots=False,debug = False):
    ctime_plots = S.get_timestamp()
    #Parse text file for temp + voltage bias points here
    f_read = open(datatxtfile)
    lines = f_read.readlines()
    datfile = {}
    temps = []
    biases = []
    for line in lines:
        temp_line = float(line.split(' mK,')[0].split('T = ')[-1])
        if not (temp_line in np.fromiter(datfile.keys(), dtype = float)):
            datfile[temp_line] = {}
            temps.append(temp_line)
        bias_line = float(line.split('Point: ')[-1].split(',')[0])
        if not (bias_line in biases):
            biases.append(bias_line)
            datfile[temp_line][bias_line] = line.split('Datafile: ')[-1].split(',')[0]
    print(temps)
    print(biases)
    if fs == None:
        fs = S.get_flux_ramp_freq()*1e3/S.get_downsample_factor()
    #Analyze the step response for each temp + voltage bias
    out_dict = {}
    for temp in temps:
        out_dict[temp] = {}
        for bias in biases:
            out_dict[temp][bias] = {}
            first_int = int(divmod(bias,1)[0])
            second_int = int(divmod(divmod(bias,1)[1]*10,1)[0])
            for band in bands:
                out_dict[temp][bias][band] = {}
                if channels == None:
                    chans = S.which_on(band)
                else:
                    chans = channels
                for chan in chans:
                    out_dict[temp][bias][band][chan] = {}
                    invert = False
                    timestamp, phase, mask = S.read_stream_data(datafile=datfile[temp][bias])

                    #Identify each of the falling edges
                    p = phase[mask[band,chan]]
                    p = p - np.min(p)
                    t = np.arange(0,len(p),1)/fs
                    min_idx_tot_array = []

                    for step in range(n_steps):
                        if debug:
                            print(f'Chunking Step Number {step}')
                        if step == 0:
                            idx_step = int((noise_dur + (step*2+1)*step_dur)*fs)
                            if p[idx_step-int(0.1*fs)] < p[0]:
                                p*= -1
                                invert = True
                            loc_phase_first_step = p[idx_step-int((step_dur/2)*fs):
                                                         idx_step+int((step_dur/2)*fs)]
                            idx_diff_min = np.argmin(np.diff(
                                    sig.savgol_filter(loc_phase_first_step,11,3)))
                            #idx_diff_min = np.argmin(np.diff(loc_phase_first_step))  
                            p_target = loc_phase_first_step[idx_diff_min]
                            if debug:
                                print(f'Target phase location in step = {p_target}')
                            min_idx_tot_array.append(idx_step - int((step_dur/2)*fs) + idx_diff_min)
                        else:
                            idx_step = int((noise_dur + (step*2+1)*step_dur)*fs)
                            loc_phase_first_step = p[idx_step-int((step_dur/2)*fs):
                                                     idx_step+int((step_dur/2)*fs)]
                            idx_diff_min = np.argmin(np.abs(loc_phase_first_step-p_target))
                            min_idx_tot_array.append(idx_step - int((step_dur/2)*fs) + idx_diff_min)
                    if make_debug_plots:
                        fig2 = plt.figure()
                        ax2 = fig2.gca()
                        ax2.plot(t,p,alpha = 0.5)
                        ax2.plot(t[min_idx_tot_array],p[min_idx_tot_array],'ro')
                        plt.xlim(noise_dur,noise_dur+2*n_steps*step_dur)
                        plt.xlabel('Time [s]',fontsize = 14)
                        plt.ylabel('Phase [Radians]',fontsize = 14)
                        plt.title(f'Band {band} Ch {chan}, {bias} V, {temp} mK')
                        plt.savefig(f'{S.plot_dir}/{ctime_plots}_Find_Edges_b{band}_c{chan}_{first_int}p{second_int}V_{temp}_mK.png')
                        plt.close()

                    #Find the amplitude of the steps and calculate R from this
                    I_step_comm = step_size*S.high_low_current_ratio/S.bias_line_resistance
                    I_height = []
                    Rs = []
                    for idx in min_idx_tot_array:
                        temp_height = 1e-12*S.pA_per_phi0*np.abs(np.mean(p[int(idx-0.5*step_dur*fs):
                                                                           int(idx-0.25*step_dur*fs)]) - 
                                                                 np.mean(p[int(idx+0.5*step_dur*fs):
                                                                           int(idx+0.75*step_dur*fs)]))/(2*pi)
                        I_height.append(temp_height)
                        I_sh = I_step_comm - temp_height
                        V_bias = I_sh*S.R_sh
                        R = V_bias/temp_height
                        Rs.append(R)
                        if debug:
                            print(f'Height [A] = {temp_height} Resistance = {R/1e-3}')

                    if debug:
                        print(f'Mean Height [uA] = {np.mean(np.asarray(I_height))/1e-6}')
                        print(f'I_command = {I_step_comm/1e-6} [uA]')
                        print(f'Mean R = {np.mean(np.asarray(Rs))/1e-3} [mOhm]')

                    out_dict[temp][bias][band][chan]['R_mOhm'] = np.mean(np.asarray(Rs))/1e-3
                    out_dict[temp][bias][band][chan]['I_Response_uA'] = -np.mean(np.asarray(I_height))/1e-6
                    out_dict[temp][bias][band][chan]['I_ratio'] = -np.mean(np.asarray(I_height))/I_step_comm
                    if invert:
                        out_dict[temp][bias][band][chan]['I_Response_uA'] *= -1
                        out_dict[temp][bias][band][chan]['I_ratio'] *= -1

                    #grab all of the steps
                    p_chunks = []
                    if make_debug_plots:
                        fig3 = plt.figure()
                        ax3 = plt.gca()
                    for step in range(n_steps):
                        p_step = p[min_idx_tot_array[step]-int((step_dur/5)*fs):
                                   min_idx_tot_array[step]+int((step_dur/5)*fs)]
                        p_step = p_step - p_step[-1]
                        if p_step[0] < p_step[-1]:
                            p_step*= -1
                        p_chunks.append(p_step)
                        if make_debug_plots:
                            ax3.plot(np.arange(0,len(p_step),1)*fs,p_step,'--',alpha = 0.6)
                    if make_debug_plots:
                        plt.xlabel('samples',fontsize = 16)
                        plt.ylabel('Response',fontsize = 16)
                        fig3.set_size_inches(16,9)
                        plt.savefig(f'{S.plot_dir}/{ctime_plots}_all_steps_b{band}_c{chan}_{first_int}p{second_int}V_{temp}_mK.png')
                        plt.close()

                    #Chunk the data around each step
                    taus_fit = []
                    pcovs = []
                    for i,p_c in enumerate(p_chunks):
                        p_c = p_c - np.mean(p_c[-100::])
                        t_c = np.arange(0,len(p_c),1)/fs
                        p_temp = p_c[len(p_c)//2-4:-50]
                        p_temp = p_temp - p_temp[-1]
                        t_temp = np.arange(0,len(p_temp),1)/fs
                        #Fit the data
                        popt,pcov = opt.curve_fit(
                            fall_fn, t_temp, p_temp, maxfev = 3000,p0 = [p_temp[0],.005])
                        if debug:
                            print(f'temp {temp} bias {bias} step {i} tau = {popt[1]}')
                        pcovs.append(pcov[1,1])
                        taus_fit.append(popt[1])
                        if make_debug_plots:
                            t_plot = np.arange(t_temp[0],t_temp[-1],1/fs)
                            fig4 = plt.figure()
                            ax4 = plt.gca()
                            ax4.semilogx(t_temp,p_temp,'o',label = f'data, {pcov[1,1]}')
                            ax4.semilogx(t_plot,fall_fn(t_plot,popt[0],popt[1]),'r--',
                                          label = f'tau = {np.round(popt[1]/1e-3,2)} ms,'+
                                          f'$\sigma$ = {np.round(np.sqrt(pcov[1,1])/1e-3,3)}ms')
                            ax4.legend()
                            plt.xlabel('Time [s]')
                            plt.ylabel('Phase [Radians]')
                            plt.title(f'Band : {band} Chan : {chan}, Bias : {bias} V, Temp : {temp} mK')
                            plt.savefig(f'{S.plot_dir}/{ctime_plots}_tau_fit_b{band}_c{chan}_step{i}_{first_int}p{second_int}V_{temp}_mK.png')
                            plt.close()

                    taus_fit = np.asarray(taus_fit)
                    pcovs = np.asarray(pcovs)
                    sum_inv_var = np.sum(1/pcovs)
                    weights = (1/pcovs)/sum_inv_var
                    weighted_avg = np.mean(taus_fit*weights)
                    print(f'Weighted average of {n_steps} for tau = {weighted_avg}')
                    out_dict[temp][bias][band][chan]['tau_avg'] = weighted_avg
                    out_dict[temp][bias][band][chan]['taus'] = taus_fit
                    out_dict[temp][bias][band][chan]['vars'] = pcovs

    for band in bands:
        for chan in chans:
            temps = np.fromiter(out_dict.keys(),dtype = float)
            biases = np.fromiter(out_dict[temps[0]].keys(),dtype = float)
            fig1, (ax1,ax2,ax3) = plt.subplots(3,1,sharex = True)
            colors = pl.cm.viridis(np.linspace(0,1,len(temps)))
            for col, temp in enumerate(temps):
                taus = []
                Rs = []
                Ratios = []
                for bias in biases:
                    taus.append(out_dict[temp][bias][band][chan]['tau_avg'])
                    Rs.append(out_dict[temp][bias][band][chan]['R_mOhm'])
                    Ratios.append(out_dict[temp][bias][band][chan]['I_ratio'])
                ax1.plot(biases,taus,'o-',color = colors[col])
                ax1.set_ylabel('Tau [ms]',fontsize = 14)
                ax2.plot(biases,Rs,'o-',color = colors[col])
                ax2.set_ylabel('$R_{bolo}$ [m$\Omega$]',fontsize = 14)
                ax3.plot(biases,Ratios,'o-',color = colors[col])
                ax3.set_ylabel('$I_{bolo}$/$I_{commanded}$',fontsize = 14)
                fig1.suptitle(f'Band {band} Chan {chan} Bias Step Response Fit Results',fontsize = 24)
                fig1.set_size_inches(9,16)
                fig1.savefig(f'{S.plot_dir}/{ctime_plots}_bias_step_analysis_summary_b{band}_c{chan}.png')
                plt.close()

    pkl.dump(out_dict,open(f'{S.output_dir}/{ctime_plots}_bias_step_analysis.pkl','wb'))
    return out_dict

if __name__=='__main__':
        parser = argparse.ArgumentParser()

        parser.add_argument('--setup', action='store_true')
        parser.add_argument('--config-file', required=True)
        parser.add_argument('--epics-root', default='smurf_server_s2')
        parser.add_argument('--bands', type=int,nargs='+', default=[2])
        parser.add_argument('--debug',action='store_true')
        parser.add_argument('--make-debug-plots',action='store_true')
        parser.add_argument('--step-size',type=float,default=0.004)
        parser.add_argument('--step-dur',type=float,default = 0.2)
        parser.add_argument('--noise-dur',type=float,default = 1.0)
        parser.add_argument('--n-steps',type = int,default = 15)
        parser.add_argument('--channels',type = int,nargs = '+',default = None)
        parser.add_argument('--datatxtfile')
        parser.add_argument('--fs',type = float,default = None)
        
        args = parser.parse_args()

        S = pysmurf.client.SmurfControl(
                        epics_root = args.epics_root,
                        cfg_file = args.config_file,
                        setup = args.setup,make_logfile=False
        )

        bands = args.bands
        debug = args.debug
        make_debug_plots = args.make_debug_plots
        step_size = args.step_size
        step_dur = args.step_dur
        noise_dur = args.noise_dur
        n_steps = args.n_steps
        channels = args.channels
        datatxtfile = args.datatxtfile
        fs = args.fs
        
        analyze_tau_data(datatxtfile= datatxtfile, bands = bands,channels = channels,fs = fs,
                         step_size = step_size,step_dur = step_dur, noise_dur = noise_dur,
                         n_steps = n_steps,make_debug_plots=make_debug_plots,debug = debug)
