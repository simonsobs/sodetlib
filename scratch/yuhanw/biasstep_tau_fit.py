def fit_exponential(data,sample_rate):
    function = lambda t,p0,p1,p2: p0 + p1*(np.exp(-t/p2))
    p0 = [data[-1],data[0]-data[-1],0.02]
    time = np.arange(0,len(data)).astype(float)/sample_rate
    success = False
    try:
        popt,pcov = optimize.curve_fit(function,time,data,p0=p0)
        success = True
    except (ValueError, RuntimeError):
        popt = [0,0   ,0]
        pcov = [[0,0,0],[0,0,0],[0,0,0]]      
    try:
        float(pcov)
        success = False
        popt = [0,0,0,0]
        pcov = [[0,0,0],[0,0,0],[0,0,0]]
    except TypeError:
        ''

    for k in range(len(pcov)):
        if pcov[k][k] < 0:
            success = False
            pcov = [[0,0,0],[0,0,0],[0,0,0]]
                         
    if success:
        r_2 = r2(function,time,data,popt)
    else:
        r_2 = 0      
    return popt,pcov,r_2,success

def r2(fit_func,time,data,p1):

    data_av = sum(data)/len(data)

    sserr = 0
    sstot = 0
    
    for i in range(len(data)):
        sserr = sserr + (data[i] - fit_func(time[i],*p1))**2
                         
        sstot = sstot + (data[i] - data_av)**2

    r_2 = 1 - sserr/sstot
                         
    return r_2

def f_exp(t,p0,p1,p2):
                         
    return p0 + p1*(np.exp(-t/p2))


def fit_step_exponential(data):
    function = lambda t,p0,p1,p2,p3,p4: p0 + p1*(np.exp(-p2*(t-p3)))*np.heaviside(t-p3,0) + p4*np.heaviside(t-p3,0)
    p0 = [0,-2.5,11,0.12,4]
    time = np.arange(0,len(data)).astype(float)/sample_rate
    success = False
    try:
        popt,pcov = optimize.curve_fit(function,time,data,p0=p0)
        success = True
    except (ValueError, RuntimeError):
        popt = [0,0,0,0,0]
        pcov = [[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]]      
    try:
        float(pcov)
        success = False
        popt = [0,0,0,0,0]
        pcov = [[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]]  
    except TypeError:
        ''

    for k in range(len(pcov)):
        if pcov[k][k] < 0:
            success = False
            pcov = [[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]]  
                         
    if success:
        r_2 = r2(function,time,data,popt)
    else:
        r_2 = 0
                         
    return popt,pcov,r_2,success

def f_step_exp(t,p0,p1,p2,p3,p4):
    
    return p0 + p1*(np.exp(-p2*(t-p3)))*np.heaviside(t-p3, 0) + p4*np.heaviside(t-p3, 0)

def power_density(data):
    sample_quantity = len(data)
    fourier_transform = np.fft.rfft(data)
    time = np.arange(0,len(data)).astype(float)/sample_rate
    samptime = np.median(np.diff(time))
    norm_fact = (1.0/samptime)*np.sum(np.abs(np.hanning(sample_quantity))**2)
    power_density = np.sqrt(np.abs( np.fft.rfft(data))**2/norm_fact)
    frequency = np.linspace(0, sample_rate, len(power_density))
    print(norm_fact)
    return frequency,power_density




def timeconstant(file,band_list,chan_list,bias_groups):
    related_phase = []

    timestamp, phase, mask, tes_bias = S.read_stream_data(file,return_tes_bias=True)
    bands, channels = np.where(mask!=-1)
    defined_step = 0.05
    S._bias_line_resistance = 15600
    ch_idx_self = 0 
    S.pA_per_phi0 = 9e6
    phase *= S._pA_per_phi0/(2.*np.pi*1e6) #uA
    S.high_low_current_ratio = 6.08
    period = 2
    fs = 4000
    S._R_sh = 0.0004
    sampleNums = np.arange(len(phase[ch_idx_self]))
    t_array = sampleNums / fs
    v_bias = 2 * tes_bias[bias_groups] * S._rtm_slow_dac_bit_to_volt * S.high_low_current_ratio
    assigned_v_bias = 5
    i_bias = np.abs(1.0E6 * v_bias / S._bias_line_resistance) #uA
    sample_rate = fs
    
    
    phase = np.abs(phase)

    tau = []
    chan_taulist = []
    band_taulist = []
    f3db = []
    for i, (b, c) in enumerate(zip(bands, channels)):
        chantau = []
        chanf3db = []
        ##magic number here to match the dummy good det list
        for index_k in range(len(band_list)):
            if b == band_list[index_k]  and c == chan_list[index_k]:
        ## identifying possible steps
                target_phase_all = np.array(phase[i])  
                related_phase.append(target_phase_all-np.mean(target_phase_all))
                if(target_phase_all[0]<target_phase_all[-1]):
                    target_phase_all = -target_phase_all
                average_phase = np.ones(len(target_phase_all))*np.mean(target_phase_all)
                dary = np.array([*map(float, target_phase_all)])
                dary -= np.average(dary)
                step = np.hstack((np.ones(int(1*len(dary))), -1*np.ones(int(1*len(dary)))))
                neg_dary = np.max(dary)-dary
                dary_step = np.convolve(dary, step, mode='valid')
                step_indx = np.argmax(dary_step)
                dip_index,_ = find_peaks(-dary_step, height=0)
                peak_index,_ = find_peaks(dary_step, height=0)
    #             plt.figure()
    #             plt.plot(dary)
    #             plt.plot(dary_step/40000)
    #             print(dip_index)
    #             print(peak_index)
                mid_points = np.append(dip_index,peak_index)
        ##now looking at each chunk of data
                if len(mid_points) < 5:
                    for target_dip in mid_points:
                        target_phase = []
                        target_time = []
                        start_time = t_array[target_dip]
                        ## defining what data to look at
                        for j, t_array_select in enumerate(t_array):
                            if t_array_select < start_time + 0.25*period and t_array_select > start_time - 0.25*period:
                                target_time.append(t_array_select)
                                target_phase.append(phase[i][j])
                        target_phase = np.array(target_phase)        
                        if(target_phase[0]<target_phase[-1]):
                            target_phase = -target_phase
                        average_phase = np.ones(len(target_time))*np.mean(target_phase)
                        ## finding the step again
                        dary_perstep = np.array([*map(float, target_phase)])
                        dary_perstep -= np.average(dary_perstep)
                        step = np.hstack((np.ones(int(1*len(dary_perstep))), -1*np.ones(int(1*len(dary_perstep)))))
                        dary_step = np.convolve(dary_perstep, step, mode='valid')
                        step_indx = np.argmax(dary_step)
                        dip_index,_ = find_peaks(-dary_step, height=0)
        #                 plt.figure()
        #                 plt.plot(dary_perstep)
        #                 plt.plot(dary_step/40000)
                        target_dip = dip_index[0]
                        target_time_step= np.arange(0,len(target_phase)).astype(float)/sample_rate
                        sample_rate = fs
                        ## linear fitting the upper and lower part of steps
                        target_phase_step_up = target_phase[target_dip-int(0.15*period*sample_rate):target_dip-int(0.02*period*sample_rate)]
                        target_time_step_up = target_time_step[target_dip-int(0.15*period*sample_rate):target_dip-int(0.02*period*sample_rate)]
                        try:
                            z_up = np.polyfit(target_time_step_up, target_phase_step_up, 1)
                        except:
                            continue
        #                     print('error at the linear fit step:')
                        target_dip = dip_index[0]
                        sample_rate = fs
                        target_phase_step_down = target_phase[target_dip+int(0.02*period*sample_rate):target_dip+int(0.15*period*sample_rate)]
                        target_time_step_down = target_time_step[target_dip+int(0.02*period*sample_rate):target_dip+int(0.15*period*sample_rate)]
                        try:
                            z_down = np.polyfit(target_time_step_down, target_phase_step_down, 1)
                        except:
                            continue
        #                     print('error at the linear fit step:')
        #                 plt.figure(figsize=(11,11))
        #                 plt.plot(np.array(target_time_step),np.array(target_phase),'o')
        #                 plt.plot(np.array(target_time_step),np.array(target_time_step)*z_up[0]+z_up[1],'-')
        #                 plt.plot(np.array(target_time_step),np.array(target_time_step)*z_down[0]+z_down[1],'-')
        #                 plt.plot(np.array(target_time_step_up),np.array(target_phase_step_up),'o')
        #                 plt.plot(np.array(target_time_step_down),np.array(target_phase_step_down),'o')
                        ## some flags
                        if z_up[0] * z_down[0] >= 0 and z_up[1] - z_down[1] > 0: 
                            ##correcting the trend
                        ##not doing correction at the moment
                            correction_target_phase = target_phase #- np.array(target_time_step)*z_up[0]
                            step_size_meas = z_up[1] - z_down[1]
                            for p in np.arange(len(correction_target_phase)):
                                # defining magic start point cutting the first few data out to avoid electrical timeconstant effect
                                if  z_up[1] - correction_target_phase[p] > 0.15 * step_size_meas and z_up[1] - correction_target_phase[p-1]  < 0.15 * step_size_meas:
                                    start_index = p
                            auto_select_phase = correction_target_phase[start_index:start_index+int(0.1*period*sample_rate)]
                            auto_select_time  = target_time_step[start_index:int(start_index+0.1*period*sample_rate)]
                        #### fit
                            try:
                                popt,pcov,r_2,success = fit_exponential(auto_select_phase,sample_rate)
                            except:
                                continue
        #                         print('error at the fit:')
        #                         print(b,c)
                            fit_time = np.arange(0,len(auto_select_time)).astype(float)/sample_rate                    
        #                     plt.figure()
        #                     plt.plot(fit_time,np.array(auto_select_phase),'o',label='correction')
                            chan_tau = popt[2]
                            if chan_tau != '' and chan_tau > 0:
                                chan_f3db = 1/(2*3.14*float(chan_tau))
                                fit_time = np.arange(0,len(auto_select_time)).astype(float)/sample_rate
                                chantau.append(chan_tau)
                                chanf3db.append(chan_f3db)
        #                     try:
        #                         plt.figure()
        #                         plt.plot(fit_time,np.array(auto_select_phase),'o',label='correction')
        #                         plt.plot(fit_time,f_exp(auto_select_time-auto_select_time[0],*popt),'x',label='fit')
        #                     except:
        #                         print('plot error')
        #                     try:
        #                         plt.figure()
        #                         plt.plot(np.array(target_time_step),np.array(correction_target_phase),'o')                    
        #                         plt.plot(fit_time+target_time_step[start_index],np.array(auto_select_phase),'o',label='correction')
        #                         plt.plot(fit_time+target_time_step[start_index],f_exp(auto_select_time-auto_select_time[0],*popt),'x',label='fit')
        #                     except:
        #                         print('plot error')
        if len(chantau) != 0:
            np.array(tau.append(np.median(chantau)))           
            chan_taulist.append(c)
            band_taulist.append(b)

    common_mode = np.median(related_phase, axis = 0)
    try:
        plt.figure(figsize=(11,6))
        plt.plot(t_array,common_mode,color='C0')
    except:
        print('something seems wrong')
    return tau, chan_taulist, band_taulist