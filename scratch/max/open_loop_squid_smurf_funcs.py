def serial_corr(wave, lag=1):
    n = len(wave)
    y1 = wave[lag:]
    y2 = wave[:n-lag]
    corr = np.corrcoef(y1, y2)[0, 1]
    return corr

def autocorr(wave):
    lags = range(len(wave)//2)
    corrs = np.array([serial_corr(wave, lag) for lag in lags])
    return lags, corrs

def take_squid_open_loop(S,cfg,bands,wait_time,Npts,NPhi0s,Nsteps,relock,frac_pp,
                        lms_freq,reset_rate_khz,lms_gain):
    """
    Takes data in open loop (no tracking the flux ramp) and steps through flux
    values to trace out a SQUID curve. This can be compared against the tracked
    SQUID curve which might not perfectly replicate this is these curves are
    poorly approximated by a sine wave (or ~3 harmonics of a fourier expansion).

    Parameters
    ----------
    bands: (int list)
        list of bands to take dc SQUID curves on
    wait_time: (float)
        how long you wait between flux step point in seconds
    Npts: (int)
        number of points you take at each flux bias step to average
    Nphi0s: (int)
        number of phi0's or periods of your squid curve you want to take at
        least 3 is recommended and more than 5 just takes longer without much
        benefit.
    Nsteps: (int)
        Number of flux points you will take total.
    relock: (bool)
        Whether or not to relock before starting flux stepping

    Returns
    -------
    raw_data : (dict)
        This contains the flux bias array, channel array, and frequency
        shift at each bias value for each channel in each band.
    """
    ctime = S.get_timestamp()
    fn_raw_data = f'{S.output_dir}/{ctime}_fr_sweep_data.npy'
    band_cfg = cfg.dev.bands[band]
    if frac_pp is None:
        frac_pp = band_cfg['frac_pp']
    if lms_freq is None:
        lms_freq = band_cfg['lms_freq_hz']
    if reset_rate_khz is None:
        reset_rate_khz = band_cfg['flux_ramp_rate_khz']
    if lms_gain is None:
        lms_gain = band_cfg['lms_gain']

    #This calculates the amount of flux ramp amplitude you need for 1 phi0
    #and then sets the range of flux bias to be enough to achieve the Nphi0s
    #specified in the fucnction call.
    frac_pp_per_phi0 = frac_pp/(lms_freq/reset_rate_khz)
    bias_low=-frac_pp_per_phi0*Nphi0s
    bias_high=frac_pp_per_phi0*Nphi0s

    #This is the step size calculated from range and number of steps
    bias_step=np.abs(bias_high-bias_low)/float(Nsteps)

    channels = {}

    bias = np.arange(bias_low, bias_high, bias_step)

    # final output data dictionary
    raw_data = {}
    raw_data['bias'] = bias
    bands_with_channels_on=[]
    for band in bands:
        channels[band] = S.which_on(band)
        if len(channels[band])>0:
            S.log(f'{len(channels[band])} channels on in band {band}, configuring band for simple, integral tracking')
            S.log(f'-> Setting lmsEnable[1-3] and lmsGain to 0 for band {band}.', S.LOG_USER)
            prev_lms_enable1 = S.get_lms_enable1(band)
            prev_lms_enable2 = S.get_lms_enable2(band)
            prev_lms_enable3 = S.get_lms_enable3(band)
            prev_lms_gain = S.get_lms_gain(band)
            S.set_lms_enable1(band, 0)
            S.set_lms_enable2(band, 0)
            S.set_lms_enable3(band, 0)
            S.set_lms_gain(band, lmsGain)

            raw_data[band]={}

            bands_with_channels_on.append(band)

    bands=bands_with_channels_on
    fs = {}

    sys.stdout.write('\rSetting flux ramp bias to 0 V\033[K before tune'.format(bias_low))
    S.set_fixed_flux_ramp_bias(0.)

    ### begin retune on all bands with tones
    for band in bands:
        fs[band] = []
        S.log('Retuning at tone amplitude {} and UC attenuator {}'.format(amplitude,uc_att))
        if relock:
            S.relock(band)
        for i in range(2):
            S.run_serial_gradient_descent(band)
            S.run_serial_eta_scan(band)

        # toggle feedback if functionality exists in this version of pysmurf
        time.sleep(5)
        S.toggle_feedback(band)

        ### end retune

    small_steps_to_starting_bias=None
    if bias_low<0:
        small_steps_to_starting_bias=np.arange(bias_low,0,bias_step)[::-1]
    else:
        small_steps_to_starting_bias=np.arange(0,bias_low,bias_step)

    # step from zero (where we tuned) down to starting bias
    S.log('Slowly shift flux ramp voltage to place where we begin.', S.LOG_USER)
    for b in small_steps_to_starting_bias:
        sys.stdout.write('\rFlux ramp bias at {:4.3f} V\033[K'.format(b))
        sys.stdout.flush()
        S.set_fixed_flux_ramp_bias(b,do_config=False)
        time.sleep(wait_time)

    ## make sure we start at bias_low
    sys.stdout.write('\rSetting flux ramp bias low at {:4.3f} V\033[K'.format(bias_low))
    S.set_fixed_flux_ramp_bias(bias_low,do_config=False)
    time.sleep(wait_time)

    S.log('Starting to take flux ramp.', S.LOG_USER)

    for b in bias:
        sys.stdout.write('\rFlux ramp bias at {:4.3f} V\033[K'.format(b))
        sys.stdout.flush()
        S.set_fixed_flux_ramp_bias(b,do_config=False)
        time.sleep(wait_time)
        for band in bands:
            fsamp=np.zeros(shape=(Npts,len(channels[band])))
            for i in range(Npts):
                fsamp[i,:]=S.get_loop_filter_output_array(band)[channels[band]]
            fsampmean=np.mean(fsamp,axis=0)
            fs[band].append(fsampmean)

    sys.stdout.write('\n')

    S.log('Done taking flux ramp data.', S.LOG_USER)

    for band in bands:
        fres = []
        for ch in channels[band]:
            fres.append(S.channel_to_freq(band,ch))
        fres=[S.channel_to_freq(band, ch) for ch in channels[band]]
        raw_data[band]['fres']=np.array(fres)
        raw_data[band]['channels']=channels[band]

        #stack
        lfovsfr=np.dstack(fs[band])[0]
        raw_data[band]['lfovsfr']=lfovsfr
        raw_data[band]['fvsfr']=np.array([arr/4.+fres for (arr,fres) in zip(lfovsfr,fres)])

    # save dataset for each iteration, just to make sure it gets
    # written to disk
    np.save(fn_raw_data, raw_data)
    S.pub.register_file(fn_raw_data,'dc_squid_curve',format='npy')

    # done - zero and unset
    S.set_fixed_flux_ramp_bias(0,do_config=False)
    S.unset_fixed_flux_ramp_bias()
    S.set_lms_enable1(band, prev_lms_enable1)
    S.set_lms_enable2(band, prev_lms_enable2)
    S.set_lms_enable3(band, prev_lms_enable3)
    S.set_lms_gain(band, lmsGain)
    return raw_data
