import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
plt.ion()

bands = [2,3]
single_channel_readout = 2
nsamp = 2**25

new_chans = False

def etaPhaseModDegree(etaPhase):
    return (etaPhase+180)%360-180

#For resonator I/Q high sampled data use eta_mag + eta_phase found in eta scans for Q and +/- 90 deg for I, for off resonance data to look at HEMT, etc set eta_mag = 1 and eta_phase = 0 & 90 or the eta_phase from the closest resonator for "Q" and that +/- 90 for "I"

#In single_channel_readout mode 2 you take data at 2.4MHz and don't need to worry about decimation & filter_alpha, for single_channel_reaout = 1 600 kHz data you do, see confluence page https://confluence.slac.stanford.edu/display/SMuRF/SMuRF+firmware#SMuRFfirmware-Datamodes

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
    for i,chan in enumerate(chans[band]):
        plt.figure()
        S.set_fixed_tone(freqs[band][i],12)
        S.set_feedback_enable(band,0)
        #S.run_serial_gradient_descent(band)
        #S.run_serial_eta_scan(band)
        S.flux_ramp_off()
        #qEtaPhaseDegree = eta_phases[band][i]
        qEtaPhaseDegree = 0
        #EtaMag = eta_mags_scaled[band][i]
        EtaMag = 1
        channel = S.which_on(band)[0]
        S.set_eta_mag_scaled_channel(band,channel,EtaMag)
        alpha = 1.0
        for IorQ in ['Q0','Q+','I+','I-']:
            if IorQ is 'Q0':
                S.set_eta_phase_degree_channel(band,channel,qEtaPhaseDegree)
            if IorQ is 'Q+':
                S.set_eta_phase_degree_channel(band,channel,etaPhaseModDegree(qEtaPhaseDegree+180))
            if IorQ is 'I+':
                S.set_eta_phase_degree_channel(band,channel,etaPhaseModDegree(qEtaPhaseDegree+90))
            if IorQ is 'I-':
                S.set_eta_phase_degree_channel(band,channel,etaPhaseModDegree(qEtaPhaseDegree-90))
            ctime1=int(S.get_timestamp())
            filename='%d.dat'%ctime1
            # take ~56 sec of data (18750 Hz)^-1 * (2^20) ~ 55.9sec.  Have to set kludge_sec=60.
            f, df, sync = S.take_debug_data(band, channel=channel, IQstream=False, single_channel_readout=single_channel_readout, nsamp=nsamp,filename=str(ctime1));
            f,Pxx = signal.welch(df,nperseg = 2**16,fs=2.4e6)
            Pxx = np.sqrt(Pxx)
            plt.loglog(f,Pxx,alpha=alpha,label = IorQ+': '+str(ctime1))
            alpha = alpha*0.8
            #dfs.append(df)
            #data=fmt.format([str(ctime1),'%0.6f'%(S.channel_to_freq(band,channel)),filename,IorQ])
            #of.write(data)
            #of.flush()
        plt.xlabel('Frequency [Hz]',fontsize = 16)
        plt.ylabel('I/Q Noise',fontsize = 16)
        plt.title('Resonator at '+str(np.round(freqs[band][i],1))+ 'MHz')
        plt.legend()
        plt.show()
        plt.savefig(S.plot_dir+'/'+str(ctime1)+'_band_'+str(band)+'_chan_'+str(chan)+'.png')
        plt.close()
        S.channel_off(band,channel)
        
S.flux_ramp_on()



