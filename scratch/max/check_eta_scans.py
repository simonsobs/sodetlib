import matplotlib.pylab as pl
import matplotlib.pyplot as plt
S.find_freq(band, subband = [43,51,52,58], drive_power = 12,make_plot=True,show_plot=False,save_plot = True)
S.setup_notches(band,drive=12,new_master_assignment=True)
c_freqs = {}
eta_mags = {}
eta_phase = {}
nist_freqs = [5193.45,5154.53,5194.99,5228.04]
nist_chans = []
freqs_on = []
for ch in S.which_on(band):
    freqs_on.append(S.channel_to_freq(band,ch))
 
freqs_on = np.asarray(freqs_on)
 
for fr in nist_freqs:
    nist_chans.append(S.which_on(band)[np.argmin(np.abs(freqs_on-fr))])
for ch in S.which_on(band):
    if ch in nist_chans:
        continue
    else:
        S.channel_off(band,ch)

f = {}
resp = {}
eta = {}
for ch in S.which_on(band):
    S.channel_off(band,ch)
    f[ch],resp[ch],eta[ch] = S.eta_estimator(band,freq=S.channel_to_freq(band,ch))

S.setup_notches(band,drive=12,new_master_assignment=False)
for ch in S.which_on(band):
    if ch in nist_chans:
        continue
    else:
        S.channel_off(band,ch)

for chan in S.which_on(band):
    c_freqs[chan] = []
    eta_mags[chan] = []
    eta_phase[chan] = []
    c_freqs[chan].append(S.get_center_frequency_mhz_channel(band,chan))
    eta_mags[chan].append(S.get_eta_mag_scaled_channel(band,chan))
    eta_phase[chan].append(S.get_eta_phase_degree_channel(band,chan))
 
for i in range(5):
    S.run_serial_gradient_descent(band)
    S.run_serial_eta_scan(band)
 
    for chan in S.which_on(band):
        c_freqs[chan].append(S.get_center_frequency_mhz_channel(band,chan))
        eta_mags[chan].append(S.get_eta_mag_scaled_channel(band,chan))
        eta_phase[chan].append(S.get_eta_phase_degree_channel(band,chan))

col = pl.cm.jet(np.linspace(0,1,len(c_freqs)))
_,sc = S.get_subband_centers(band)
bc = S.get_band_center_mhz(band)
for ch in S.which_on(band):
    plt.figure()
    plt.plot(f[ch],np.abs(resp[ch]),'k-',label = 'Mag')
    plt.plot(f[ch],np.real(resp[ch]),'k--',label = 'Real')
    plt.plot(f[ch],np.imag(resp[ch]),'k.-',label = 'Imag')
    center_subband_mhz = bc + sc[S.get_subband_from_channel(band,ch)]
    for i,c in enumerate(col):
        plt.plot([c_freqs[ch][i]+center_subband_mhz,c_freqs[ch][i]+center_subband_mhz],[-0.5,0.5],color = c)

    
