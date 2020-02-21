import numpy as np
import matplotlib.pylab as plt

band = 2

chans_on = np.asarray([ 27,  31,  39,  55,  79,  95, 119, 123, 159, 191, 247, 251, 263, 271, 299, 311, 327, 359, 363, 367, 391, 399, 427, 463, 491])
freqs_on = np.asarray([5153.41176643, 5228.9300209 , 5184.51375923, 5201.70179367,5213.25170193, 5230.95227146, 5202.3697876 , 5164.00579262,5231.91998386, 5240.7478857 , 5204.12164459, 5165.96993752,5176.57430077, 5214.13700218, 5147.10797939, 5205.38774242,5178.60579815, 5188.56198559, 5148.43400345, 5226.34989414,5179.76771507, 5217.72999268, 5150.21187859, 5218.76221828,5151.04597893])

for c in chans_on:
    S.channel_off(band,c)

for i, chan in enumerate(chans_on):
    S.set_fixed_tone(freqs_on[i],12)
    channel = S.which_on(band)
    S.set_feedback_enable(band,0)
    S.set_lms_enable1(band,False)
    S.set_lms_enable2(band,False)
    S.set_lms_enable3(band,False)
    of=open(S.plot_dir + '/'+'%s_fsnt_b%d_ch%d.dat'%(S.get_timestamp(),band,chan),'w+')
    fmt='{0[0]:<15}{0[1]:<15}{0[2]:<15}{0[3]:<15}\n'
    columns=['ctime','ff','fres','filename']
    hdr=fmt.format(columns)
    
    of.write(hdr)
    of.flush()
    print(hdr.rstrip())

    ffs=[]
    dfs=[]

    # the script will run a lot faster if you turn off all other channels.
    channel=chan

    # Shawn measured a phi0 to be 0.253 in the units that the S.set_fixed_flux_ramp_bias
    # function takes at Pole in band 3.
    ffphi0=0.1413904829882473*(2/3)
    # take this many points between ff=0 and ff=ffphi0
    nffsteps=40

    # settings for taking fast(ish) data
    #S.set_decimation(band,5) # decimate by 2^5=32.  Sample rate in single_channel_readout=1 is 600kHz, so this will be 18750 Hz.
    #S.set_filter_alpha(band,1638) # DDS filter.  I forget what f3dB this was, but it's about right for decimation=5.

    #600 kHz
    S.set_decimation(band,0)
    # 32768 ends up negative when you read it back for some reason
    S.set_filter_alpha(band,32767) # DDS filter.  f3dB ~ 275kHz
    nsamp=2**25
    # 2 for 2.4Mhz, 1 for 600khz
    single_channel_readout=2
    
    # make sure we're on resonance.
    S.run_serial_gradient_descent(band)
    S.run_serial_eta_scan(band)

    ffs=np.linspace(0,ffphi0,nffsteps)
    for ffrb in ffs:
        S.set_fixed_flux_ramp_bias(ffrb)
        # want to stay on resonance.  flux ramp changed, so resonator frequency changed.  Re-center
        # tone on resonator's new position.
        S.run_serial_gradient_descent(band)
        S.run_serial_eta_scan(band)
    
        ctime1=int(S.get_timestamp())
        filename='%d.dat'%ctime1
        # take ~56 sec of data (18750 Hz)^-1 * (2^20) ~ 55.9sec.  Have to set kludge_sec=60.
        #input('turn on steps, then press return')
        f, df, sync = S.take_debug_data(band, channel=channel, IQstream=False, single_channel_readout=single_channel_readout, nsamp=nsamp, filename=str(ctime1)+'_b%d_ch%d_percentflux%d'%(band,chan,ffrb*100));
        #input('turn off steps, then press return')
        dfs.append(df)
        data=fmt.format([str(ctime1),'%0.6f'%(ffrb),'%0.6f'%(S.channel_to_freq(band,channel)),filename])
        of.write(data)
        of.flush()

    S.unset_fixed_flux_ramp_bias(ffrb)

    of.close()
    S.turn_off_fixed_tones(band)

S.set_feedback_enable(band,1)
S.set_lms_enable1(band,True)
S.set_lms_enable2(band,True)
S.set_lms_enable3(band,True)

S.find_freq(band,drive_power=12,make_plot=True,show_plot=False,save_plot=True)
S.setup_notches(band,drive=12,new_master_assignment=False)
S.run_serial_gradient_descent(band)
S.run_serial_eta_scan(band)
Bad_Track_Chans = [15, 23, 28, 29, 59, 63, 87, 91, 127, 151, 183, 187, 255,267, 287, 295, 303, 315, 319, 335, 423]
for chans in Bad_Track_Chans:
    S.channel_off(band,chans)

S.toggle_feedback(band)
S.tracking_setup(band,reset_rate_khz=4,lms_freq_hz=20000,fraction_full_scale=0.14139048298824738,make_plot=True,show_plot=False,channel=S.which_on(band),nsamp=2**18,feedback_start_frac=0.02,feedback_end_frac=0.94)
S.take_noise_psd(band=band,meas_time=60,low_freq=[1],high_freq=[5],nperseg=2**16,save_data=True,show_plot=False,make_channel_plot=False)
