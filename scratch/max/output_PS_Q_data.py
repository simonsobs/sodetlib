import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import pickle as pkl
plt.ion()

bands = [3]
single_channel_readout = 2
nsamp = 2**25

new_chans = True
out_dict = {}
ctime_plots = S.get_timestamp()

def etaPhaseModDegree(etaPhase):
        return (etaPhase+180)%360-180

#For resonator I/Q high sampled data use eta_mag + eta_phase found in eta scans for Q and +/- 90 deg fo\
r I, for off resonance data to look at HEMT, etc set eta_mag = 1 and eta_phase = 0 & 90 or the eta_phas\
e from the closest resonator for "Q" and that +/- 90 for "I"                                            

#In single_channel_readout mode 2 you take data at 2.4MHz and don't need to worry about decimation & fi\
lter_alpha, for single_channel_reaout = 1 600 kHz data you do, see confluence page https://confluence.s\
lac.stanford.edu/display/SMuRF/SMuRF+firmware#SMuRFfirmware-Datamodes                                   

if new_chans == True:
        chans = {}
        freqs = {}
        sbs = {}
        eta_mags_scaled = {}
        eta_phases = {}
        for band in bands:
                chans[band] = S.which_on(band)
                freqs[band] = []
                sbs[band] = []
                eta_mags_scaled[band] = []
                eta_phases[band] = []
                for chan in chans[band]:
                        freqs[band].append(S.channel_to_freq(band,chan))
                        sbs[band].append(S.freq_to_subband(band,S.channel_to_freq(band,chan))[0])
                        eta_mags_scaled[band].append(S.get_eta_mag_scaled_channel(band,chan))
                        eta_phases[band].append(S.get_eta_phase_degree_channel(band,chan))
                        S.channel_off(band,chan)
                freqs[band] = np.asarray(freqs[band])
                sbs[band] = np.asarray(sbs[band])
                eta_mags_scaled[band] = np.asarray(eta_mags_scaled[band])
                eta_phases[band] = np.asarray(eta_phases[band])

for band in bands:
        out_dict[band] = {}
        for i,chan in enumerate(chans[band]):
                out_dict[band][chan] = {}
                plt.figure()
                S.set_fixed_tone(freqs[band][i],12)
                #S.set_fixed_tone(freqs[band][i]+0.2,12)                                                
                S.set_feedback_enable(band,0)
                S.flux_ramp_off()
                channel = S.which_on(band)[0]
                qEtaPhaseDegree = eta_phases[band][i]
                S.set_eta_phase_degree_channel(band,channel,qEtaPhaseDegree)
                #qEtaPhaseDegree = 0                                                                    
                EtaMag = eta_mags_scaled[band][i]
                S.set_eta_mag_scaled_channel(band,channel,EtaMag)
                #EtaMag = 1                                                                             
                #S.run_serial_gradient_descent(band)                                                    
                #S.run_serial_eta_scan(band)                                                            
                print(f'Center Freq {S.get_center_frequency_mhz_channel(band,channel)}')
                print(f'Eta Phase {S.get_eta_phase_degree_channel(band,channel)}')
                print(f'Eta Mag Scaled {S.get_eta_mag_scaled_channel(band,channel)}')
                #Etamag = S.get_eta_mag_scaled_channel(band,channel)                                    
                #S.set_eta_mag_scaled_channel(band,channel,Etamag)                                      
                #qEtaPhaseDegree = S.get_eta_phase_degree_channel(band,channel)                         
                alpha = 1.0
                for IorQ in ['Q0','Q+','I+','I-']:
                        if IorQ is 'Q0':
                                S.set_eta_phase_degree_channel(band,channel,qEtaPhaseDegree)
                        if IorQ is 'Q+':
                                S.set_eta_phase_degree_channel(band,channel,etaPhaseModDegree(qEtaPhase\
Degree+180))
                        if IorQ is 'I+':
                                S.set_eta_phase_degree_channel(band,channel,etaPhaseModDegree(qEtaPhase\
Degree+90))
                        if IorQ is 'I-':
                                S.set_eta_phase_degree_channel(band,channel,etaPhaseModDegree(qEtaPhase\
Degree-90))
                        ctime1=int(S.get_timestamp())
                        filename='%d.dat'%ctime1
                        # take ~56 sec of data (18750 Hz)^-1 * (2^20) ~ 55.9sec.  Have to set kludge_se\
c=60.                                                                                                   
                        f, df, sync = S.take_debug_data(band, channel=channel, IQstream=False, single_c\
hannel_readout=single_channel_readout, nsamp=nsamp,filename=str(ctime1));
                        f,Pxx = signal.welch(df,nperseg = 2**16,fs=2.4e6)
                        Pxx = np.sqrt(Pxx)
                        if IorQ == 'Q0':
                                out_dict[band][chan][IorQ] = {}
                                out_dict[band][chan][IorQ]['f'] = f
                                out_dict[band][chan][IorQ]['Pxx'] = Pxx
                        if IorQ == 'I+':
                                out_dict[band][chan][IorQ] = {}
                                out_dict[band][chan][IorQ]['f'] = f
                                out_dict[band][chan][IorQ]['Pxx'] = Pxx
                        plt.loglog(f,Pxx,alpha=alpha,label = IorQ+': '+str(ctime1))
                        alpha = alpha*0.8
                        #dfs.append(df)                                                                 
                        #data=fmt.format([str(ctime1),'%0.6f'%(S.channel_to_freq(band,channel)),filenam\
e,IorQ])                                                                                                
                        #of.write(data)
                        #of.flush()                                                                     
                plt.xlabel('Frequency [Hz]',fontsize = 16)
                plt.ylabel('I/Q Noise',fontsize = 16)
                plt.title('Resonator at '+str(np.round(freqs[band][i],1))+ 'MHz')
                plt.legend()
                plt.show()
                plt.savefig(S.plot_dir+'/'+str(ctime_plots)+'_band_'+str(band)+'_chan_'+str(chan)+'.png\
')
                plt.close()
                S.channel_off(band,channel)
pkl.dump(out_dict,open(S.plot_dir+'/'+str(ctime_plots)+'_fast_sample_data_psd.pkl','wb'))
S.flux_ramp_on()
