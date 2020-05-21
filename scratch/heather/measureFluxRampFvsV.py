# Runlike this exec(open("measureFluxRampFvsV.py").read())
# that way it will use your pysmurf S object.

import pysmurf
import numpy as np
import time
import sys

## instead of takedebugdata try relaunch PyRogue, then loopFilterOutputArray, which is 
## the integral tracking term with lmsEnable[1..3]=0

#S = pysmurf.SmurfControl(make_logfile=False,setup=False,epics_root='smurf_server_s2',cfg_file='/data/pysmurf_cfg/experiment_fp29_smurfsrv03_noExtRef_lbOnlyBay0.cfg')

import os
fn_raw_data = os.path.join('./', '%s_fr_sweep_data.npy'%(S.get_timestamp()))

#######
# [(None,None)] means don't change the amplitude or uc_att, but still retunes
#amplitudes_and_uc_atts=[(8,30),(9,30),(10,30),(11,30),(12,30),(12,24),(12,18),(12,12),(12,6),(12,0)]
amplitudes_and_uc_atts=[(None,None)]

# needs to be nonzero, or won't track flux ramp well,
# particularly when stepping flux ramp by large amounts
lmsGain=6
hbInBay0=False
relock=False 
bands=[3]
bias=None
# no longer averaging as much or waiting as long between points in newer fw which has df filter
wait_time=0.125
Npts=3
#bias_low=-0.432
#bias_high=0.432
bias_low=-0.60
bias_high=0.60
Nsteps=500
#Nsteps=25
bias_step=np.abs(bias_high-bias_low)/float(Nsteps)
channels=None
#much slower than using loopFilterOutputArray,
#and creates a bunch of files
use_take_debug_data=False

# Look for good channels
if channels is None:
    channels = {}
    for band in bands:
        channels[band] = S.which_on(band)
print(channels[band])

if bias is None:
    bias = np.arange(bias_low, bias_high, bias_step)
    
# final output data dictionary
raw_data = {}
raw_data['bias'] = bias
print(channels[band])
bands_with_channels_on=[]
for band in bands:
    print(band,channels[band])
    if len(channels[band])>0:
        S.log('{} channels on in band {}, configuring band for simple, integral tracking'.format(len(channels[band]),band))
        S.log('-> Setting lmsEnable[1-3] and lmsGain to 0 for band {}.'.format(band), S.LOG_USER)
        S.set_lms_enable1(band, 0)
        S.set_lms_enable2(band, 0)
        S.set_lms_enable3(band, 0)
        S.set_lms_gain(band, lmsGain)

        raw_data[band]={}

        bands_with_channels_on.append(band)

bands=bands_with_channels_on

for (amplitude,uc_att) in amplitudes_and_uc_atts:

    sys.stdout.write('\rSetting flux ramp bias to 0 V\033[K before tune'.format(bias_low))
    S.set_fixed_flux_ramp_bias(0.)

    # make sure we tune at bias low
    #sys.stdout.write('\rSetting flux ramp bias low at {:4.3f} V\033[K'.format(bias_low))
    #S.set_fixed_flux_ramp_bias(bias_low,do_config=True)
    #time.sleep(wait_time)
    
    ### begin retune on all bands with tones
    for band in bands:
        S.log('Retuning at tone amplitude {} and UC attenuator {}'.format(amplitude,uc_att))
        if relock:
            S.relock(band)
        if amplitude is not None:
            S.set_amplitude_scale_array(band,np.array(S.get_amplitude_scale_array(band)*amplitude/np.max(S.get_amplitude_scale_array(band)),dtype=int))
        if uc_att is not None:
            S.set_att_uc(band,uc_att)
        S.run_serial_gradient_descent(band)
        S.run_serial_eta_scan(band)

        # toggle feedback if functionality exists in this version of pysmurf
        time.sleep(5)
        if 'toggle_feedback' in dir(S):
            S.toggle_feedback(band)
        
        raw_data[band][(amplitude,uc_att)]={}
        
    ### end retune

    ##
    ## THIS DOESN'T WORK AND I DON'T UNDERSTAND WHY NOT
    small_steps_to_starting_bias=None
    if bias_low<0:
        small_steps_to_starting_bias=np.arange(bias_low,0,bias_step)[::-1]
    else:
        small_steps_to_starting_bias=np.arange(0,bias_low,bias_step)
    
    # step from zero (where we tuned) down to starting bias
    S.log('Slowly shift flux ramp voltage to place where we begin.'.format(amplitude,uc_att), S.LOG_USER)    
    for b in small_steps_to_starting_bias:
        sys.stdout.write('\rFlux ramp bias at {:4.3f} V\033[K'.format(b))
        sys.stdout.flush()
        S.set_fixed_flux_ramp_bias(b,do_config=False)
        time.sleep(wait_time)        

    ## make sure we start at bias_low
    sys.stdout.write('\rSetting flux ramp bias low at {:4.3f} V\033[K'.format(bias_low))
    S.set_fixed_flux_ramp_bias(bias_low,do_config=False)
    time.sleep(wait_time)
    
    S.log('Starting to take flux ramp with amplitude={} and uc_att={}.'.format(amplitude,uc_att), S.LOG_USER)    

    fs={}
    for band in bands:
        fs[band]=[]

    for b in bias:
        sys.stdout.write('\rFlux ramp bias at {:4.3f} V\033[K'.format(b))
        sys.stdout.flush()
        S.set_fixed_flux_ramp_bias(b,do_config=False)
        time.sleep(wait_time)

        for band in bands:
            if use_take_debug_data:
                f,df,sync=S.take_debug_data(band,IQstream=False,single_channel_readout=0)
                fsampmean=np.mean(f,axis=0)
                fs[band].append(fsampmean)
            else:
                fsamp=np.zeros(shape=(Npts,len(channels[band])))
                for i in range(Npts):
                    fsamp[i,:]=S.get_loop_filter_output_array(band)[channels[band]]
                fsampmean=np.mean(fsamp,axis=0)
                fs[band].append(fsampmean)

    sys.stdout.write('\n')

    S.log('Done taking flux ramp with amplitude={} and uc_att={}.'.format(amplitude,uc_att), S.LOG_USER)

    for band in bands:

        fres=[S.channel_to_freq(band, ch) for ch in channels[band]]
        raw_data[band][(amplitude,uc_att)]['fres']=np.array(fres) + (2e3 if hbInBay0 else 0)
        raw_data[band][(amplitude,uc_att)]['channels']=channels[band]

        if use_take_debug_data:
            #stack
            fovsfr=np.dstack(fs[band])[0]
            [sbs,sbc]=S.get_subband_centers(band)
            fvsfr=fovsfr[channels[band]]+[sbc[np.where(np.array(sbs)==S.get_subband_from_channel(band,ch))[0]]+S.get_band_center_mhz(band) for ch in channels[band]]
            raw_data[band][(amplitude,uc_att)]['fvsfr']=fvsfr + (2e3 if hbInBay0 else 0)
        else:
            #stack
            lfovsfr=np.dstack(fs[band])[0]
            raw_data[band][(amplitude,uc_att)]['lfovsfr']=lfovsfr
            raw_data[band][(amplitude,uc_att)]['fvsfr']=np.array([arr/4.+fres for (arr,fres) in zip(lfovsfr,fres)]) + (2e3 if hbInBay0 else 0)

    # save dataset for each iteration, just to make sure it gets
    # written to disk
    np.save(fn_raw_data, raw_data)
            
# done - zero and unset
S.set_fixed_flux_ramp_bias(0,do_config=False)
S.unset_fixed_flux_ramp_bias()
