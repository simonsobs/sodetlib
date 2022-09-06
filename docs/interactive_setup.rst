Interactive Setup
===================

This guide will walk you through the basic functions used to setup resonators
through pysmurf. For more information on the ``uxm_setup`` function that tries
to automate this, see the :ref:`Setup` operation page.

All instruction here assume you are connected to the SMuRF Crate through a
pysmurf instance created as described in :ref:`LoadConfigs`.

Biasing Amplifiers
-------------------

.. todo:: Add information on using health checks/optimization functions for 
    amplifier biases.

Just doing things by hand, you can check and change Gate voltage settings::
    
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


Tuning On Resonators
---------------------
There are several different ways to tune on the resonators and the choice of how
to run the tuning depends on where you are in a cooldown and how carefully you
are managing the channel mapping. We need to make new channel assignments every
cooldown, because the resonators shift a bit each time we get down to 100 mK.
This has to do with the Nb trannsition so this includes times you warm up to 4K
and go back down to 100mK.

Alternatively, every time we make a new channel assignment we cannot guarantee 
that the mapping between channel and TES remains the same. This means that if we
want to maintain the same detector mapping throughout a cooldown we need to tune
on the same set of channel assignments. The channel to detector mapping will become 
very important when running in Chile or while doing optical tests where detector 
mapping information is acquired spread out in time. 

Eventually the code snippets below will need to become defined scripts or tasks,
but for now we will have them here.


Tuning a Device for the First Time
..................................

These are setup pieces that you usually have to do once per device + readout
chain but afterwards seem to be constant across cooldowns. Note that these are
done in conjunction the call in :ref:`scratch_tuning`. That's because the first
time setup is the hardest.

1. **Setting the attenuation levels.** This is adjusting the `atten_uc` and
`atten_dc`, and the optimized values are constant across cooldowns. Initial
estimates are needed for these values before running any of the calls in
:ref:`scratch_tuning` because if they are way off they will affect the initial
setup of things like the phase delay measurements. A good starting point is
usually 15/15 but if you have no information it might be better to chat with
someone more experienced. 

After tracking has been completely set up there's another point where you want
to optimize these parameters to minimize the readout noise in the setup.

.. todo:: Add info on how to use optimization functions for attenuation

2. **Setting the phase delay for the system.** This delay value is saved in the
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

Note that `S.set_band_delay_us` overwrites `S.freq_resp` so you have to reload
channel assignments / tune files if you call this function.

3. **Setting the Tracking parameters.** This is technically after tuning, but
the very first time you run `S.tracking_setup` you need to measure and adjust
some of parameters used in this function. Chicago/LATRt has found this to be
very close to constant across cooldowns. To do this for each band::
    
    S.tracking_setup(..., meas_lms_freq=True, lms_freq_hz=None ...)
    band_cfg = cfg.dev.bands[band]
    frac_pp = band_cfg['frac_pp'] * 20000 / S.get_lms_freq_hz(band)
    cfg.dev.update_band( band, {'frac_pp':frac_pp, 'lms_freq_hz':20000})
    cfg.dev.dump(cfg.dev_file, clobber=True)
    S.tracking_setup(...,meas_lms_freq=False,lms_freq_hz=band_cfg['lms_freq_hz'] ...)

Even this explanation is a bit simplified from a complete optimization of the
tracking. If you are reading this and need more information / things aren't
working you should chat with someone with more experience.

.. todo:: Add info on how to use optimization functions for tracking parameters


.. _scratch_tuning: 

Tuning from Sratch for a Cooldown
.................................

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
    
    ## Default Setup Tracking Call, used after all versions of tuning
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



Re-Tuning on an existing tune file
...................................

This set of calls retunes onto an existing set of channel assignments and is the
most aggressive set of tuning commands you can use while maintaining the same
channel assignments::
    
    S.load_tune(cfg.dev.exp['tunefile'])
    for band in bands:
        S.load_master_assignment( band, S.freq_resp[band]['channel_assignment'])
    
    for band in bands:
        band_cfg = cfg.dev.bands[band]
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
