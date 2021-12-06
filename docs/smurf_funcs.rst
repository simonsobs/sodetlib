Smurf Functions
================

All instruction here assume you are connected to the SMuRF Crate through a
pysmurf instance created as described in :ref:`LoadConfigs`.

Biasing Amplifiers
-------------------

Check and change Gate voltage settings::
    
    S.get_amplifier_biases()
    ## Example Numbers
    S.set_50k_amp_gate_voltage(-0.66)
    S.set_hemt_gate_voltage(-1.10)


Turning on and off LNAs::
    
    ## Turn on 
    S.C.write_ps_en(11)

    ## Turn off 
    S.C.write_ps_en(0)

Save to device config file::

    biases = S.get_amplifier_biases()
    cfg.dev.update_experiment({'amp_50k_Vg': biases['50K_Vg']})
    cfg.dev.update_experiment({'amp_hemt_Vg': biases['hemt_Vg']})
    cfg.dev.dump(cfg.dev_file, clobber=True)

.. todo:: Add information on using optimization functions for amplifier biases.

Tuning On Resonators
---------------------
There are several different ways to tune on the resonators and the choice of how
to run the tuning depends on where you are in a cooldown and how carefully you
are managing the channel mapping. We need to make new channel assignments every
cooldown, because the resonators shift a bit each time we get down to 100 mK.
Alternatively, every time we make a new channel assignment we cannot guarantee 
that the mapping between channel and TES remains the same. This means that if we
want to maintain the same detector mapping throughout a cooldown we need to tune
on the same set of channel assignments. The channel to detector mapping  will become 
very important when running in Chile or while doing optical tests where detector 
mapping information if acquired spread out in time. 

Eventually the code snippets below will need to become defined scripts or tasks,
but for now we will have them here.

Tuning from Sratch
...................

Assuming `bands` is a list of SMuRF Bands available on the device that you have
plugged in. The first time you're tuning on resonators for a cooldown you should
use::

    for band in bands:
        band_cfg = cfg.dev.bands[band]
        S.find_freq(band, tone_power=band_cfg['drive'], make_plot=True,
                    save_plot=True, amp_cut=0.1)
        S.setup_notches(band, tone_power=band_cfg['drive'],
                        new_master_assignment=True)

    for band in bands:
        ## no relock here because setup_notches calls relock at the end
        for _ in range(3):
            S.run_serial_gradient_descent(band)
            S.run_serial_eta_scan(band)
    
    ## WARNING FILE CHANGING!!!!
    cfg.dev.update_experiment({'tunefile': S.tune_file})
    cfg.dev.dump(cfg.dev_file, clobber=True)
    
    ## Defaul Setup Tracking Call, used after all versions of tuning
    for band in bands:
        band_cfg = cfg.dev.bands[band]
        S.tracking_setup(band, reset_rate_khz=cfg.dev.bands[band]['flux_ramp_rate_khz'],
                 fraction_full_scale=band_cfg['frac_pp'] ,
                 make_plot=True, save_plot=True, show_plot=False,
                 nsamp=2**18, lms_freq_hz=band_cfg['lms_freq_hz'] ,
                 meas_lms_freq=False, channel = S.which_on(band)[::20],
                 feedback_start_frac=cfg.dev.bands[band]['feedback_start_frac'],
                 feedback_end_frac=cfg.dev.bands[band]['feedback_end_frac'],
                 lms_gain=cfg.dev.bands[band]['lms_gain'])

The critical parts here is that we run `S.find_freq` and `S.setup_notches(...
new_master_assignment=True)`. These two piece combined make a new channel
assignment file for that band.

**Adjustments the first time you use a device**

These are setup pieces that you usually have to do once per device + readout chain
but afterwards seem to be constant across cooldowns.

1. **Setting the phase delay for the system.** This delay value is saved in the
`pysmurf` config file and is generally constant across cooldowns. But the first
time you plug in a device to a specific readout chain you should set this delay
number. (We often check it each cooldown but if it's changed that's usually
reflecting a firmware update error and not an actual change.) Unlike the device
config file, the `pymurf` config file must be updated by hand.::
     
    for band in bands:
        band_cfg = cfg.dev.bands[band]
        S.set_att_uc( band, band_cfg['uc_att'] )
        S.set_att_dc( band, band_cfg['dc_att'] )
        delay, res = S.estimate_phase_delay(band)
        S.set_band_delay_us(band, delay)

2. **Setting the attenuation levels.** This is adjusting the `atten_uc` and
`atten_dc` values to minimize the readout noise for the band. These values are
constant across cooldowns.

.. todo:: Add info on how to use optimization functions for attenuation

3. **Setting the Tracking parameters.** We want to adjust some of the tracking
parameters. Chicago/LATRt has found this to be very close to constant across
cooldowns. To do this for each band::
    
    S.tracking_setup(..., meas_lms_freq=True, ...)
    band_cfg = cfg.dev.bands[band]
    frac_pp = band_cfg['frac_pp'] * 20000 / S.get_lms_freq_hz(band)
    cfg.dev.update_band( band, {'frac_pp':frac_pp, 'lms_freq_hz':20000})
    cfg.dev.dump(cfg.dev_file, clobber=True)
    S.tracking_setup(..., meas_lms_freq=False, ...)

.. todo:: Add info on how to use optimization functions for tracking parameters


Re-Tuning on an existing tune file
...................................

This set of calls retunes onto an existing set of channel assignments and is the
most aggressive set of tuning commands you can use while maintaining the same
channel assignments::
    
    S.load_tune(cfg.dev.exp['tunefile'])
    for band in bands:
        S.load_master_assignment( band, S.freq_resp[band]['channel_assignment'])
    
    for band in bands:
        S.setup_notches(band, tone_power=band_cfg['drive'],
                        new_master_assignment=False)
    for _ in range(3):
        S.run_serial_gradient_descent(band)
        S.run_serial_eta_scan(band)

After this you run the same `tracking_setup` call as the tuning from scrach
section. The most relevant parts here is that you do not call `find_freq` and
use `S.setup_notches(..., new_master_assignment=False)` The
`load_master_assignment` section has been found to matter when switching between
devices / slots such that the channel assignments you want to tune on are not the
most recent ones for the particular band in the `tune` folder. This set of
commands is what Chicago has been using to switch between two different devices
read out by the same slot while maintaining the same channel assignment set.

Restarting on the same Tuning
..............................

If you have restarted your system or just want to try a more mild and faster
reset on the resonators using (ex. if your readout noise is somewhat elevated):: 

    S.load_tune(cfg.dev.exp['tunefile'])
    for band in bands:
        S.relock(band)
        for _ in range(3):
            S.run_serial_gradient_descent(band)
            S.run_serial_eta_scan(band)

After this you run the same `tracking_setup` call as the tuning from scrach
section. 

Optimization Functions API
--------------------------
.. automodule:: sodetlib.smurf_funcs.optimize_params
   :members: optimize_tracking, optimize_bias,
             optimize_attens
   :noindex:

Smurf Operations API
--------------------
.. automodule:: sodetlib.smurf_funcs.smurf_ops
    :members: take_squid_open_loop, find_and_tune_freq, tracking_quality,
            get_session_files, load_session, take_g3_data, stream_g3_on,
            stream_g3_off, apply_dev_cfg, cryp_amp_check, get_wls_from_am,
            plot_band_noise, loopback_test, plot_loopback_results
    :noindex:

