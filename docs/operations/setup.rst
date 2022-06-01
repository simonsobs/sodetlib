Setup
======

First Time Setup
-----------------

Before being able to properly take data on a new module, there are a number of
parameters which need to be properly configured, which can be done using the
``uxm_setup`` and ``uxm_relock`` modules in sodetlib.

For first time setup, many of these parameters must be computed, and a tunefile
must be made from scratch. The ``uxm_setup`` operation goes through the
procedures described below to figure out these initial values. Operating
parameters are all stored in the Device Config, so this really only needs to be
run once per cooldown / module. If this has already been done, you can use the
:ref:`relock` function to return to a functioning state based on params already
stored in the device cfg.

To run the setup procedure, you can simply run

.. code-block:: python

    from sodetlib.operations.uxm_setup import uxm_setup

    success, summary = uxm_setup(S, cfg, bands=np.arange(8))

The setup procedures take many input configuration parameters from the device cfg,
so to modify these you'll either want to edit the dev cfg file or modify the cfg
object before passing it to the setup function, for example

.. code-block:: python

    from sodetlib.operations.uxm_setup import uxm_setup

    # Increases the amp drain current tolerance to 0.5 mA
    cfg.dev.exp['amp_hemt_Id_tolerance'] = 0.5  
    success, summary = uxm_setup(S, cfg, bands=np.arange(8))

Amplifier Setup
````````````````

First we must bias the cold 50k and hemt amplifiers. Depending on the
cryocard revision, either one set (C02) or two sets (C04/C05) of amplifiers
can be biased via smurf. Additionally, the C04/C05 cryocards allow the drain
voltages to be specified, while for C02 cryocards these are fixed.

The ``setup_amps`` function will first determine which revision cryocard is
connected based on firmware version, and therefore which amplifiers are able
to be biased. To be biased, these amps must also be listed in the device cfg.
It then enables the amps / sets the drain voltages if necessary.
It then checks if the amps happen to already be
biased properly. If not, the function will sweep the gate-voltage until it
finds one that hits the target drain voltages specified in the device cfg. 

Attenuator Estimation
````````````````````````

In order for phase-delay-estimation and resonator tuning to work properly,
the power going into the smurf has to be balanced such that it doesn't
saturate the ADC's, but still is large enough for the peak-finding algorithms
to work. Two modifiable attenuators per-band can be used to determine the power
loss between the smurf-output and the cryostat-input (the UC attenuation) and 
the between the cryostat-output and smurf-input (the DC attenuation).

We can sweep over these parameters to determine what they should be to achieve
optimal noise, but for the time being we just want to estimate what the attens
should be in order to properly run estimate-phase-delay and the tuning
procedure.

For this step, we want to find the value ``att`` for each band such that when
we set both the UC and DC atten to ``att`` we get a reasonable amplitude
response. To do this we start at ``att=15``, and perform a binary search to
find a value of ``att`` that puts us in the correct power region.

Estimate Phase Delay
``````````````````````

Estimate Phase Delay is run for each band to measure the analog and digital
phase delay for each band. See the pysmurf docstrings for more detail.

Initial Tune
``````````````

To tune from scratch, we run the following pysmurf functions for each smurf-band:

1. First we run find_freq, which does a course sweep of the frequency range of the
   smurf-band, and finds approximate peak locations. The ``res_amp_cut`` and
   ``res_grad_cut`` options can be modified in the device cfg to change the
   threshold of which peaks are cut.
2. Then we run ``setup_notches``, which takes a fine scan across each resonance
   in series, obtaining a better estimate of the peak frequency. This takes a
   longer time, and will take roughly 45 min to run on all 8 bands.
3. Finally we run the ``run_serial_gradient_descent`` and ``serial_eta_scan``
   functions to further hone in on resonances, and to calculate the eta param
   for each channel.

After this procedure you end up with a populated tunefile containing the
resonance frequencies and eta parameters for each resonator. This is stored in
the device cfg and will be used whenever you have to relock detectors.

Tracking
``````````

Next we run the tracking setup procedures (See the :ref:`tracking` section).

Noise
``````

Finally we are able to take detector data! We take a 30 second timestream here
and calculate the white noise levels for all channels. 

The detectors at this point are superconducting, so the white noise is given by:

.. math::

    NEI = \sqrt{NEI_\mathrm{sh}^2 + NEI_\mathrm{readout}^2}

with

.. math::

    NEI_\mathrm{sh} = \sqrt{4 k_B T_b / R_{sh}} 
    \qquad \mathrm{and} \qquad
    NEI_\mathrm{readout} \approx 45 \;\mathrm{pA} / \sqrt{\mathrm{Hz}}

Here you want to plug in values for your bath temperature and shunt resistances
to see if the median white noise levels make sense.
For example, with :math:`T_b = 100 \; \mathrm{mK}` and
:math:`R_\mathrm{sh} = 0.4 \; \mathrm{m}\Omega`, you can expect a white noise level
of around  :math:`125 \; \mathrm{pA}/\sqrt{\mathrm{Hz}}`. If you're close to
this number, you're good to go and ready to continue with taking a biasgroup map!

If you're seeing white noise levels much higher than this, something is
probably wrong and you may need to go back and run the previous steps
individually to debug (see the API section for how to run each step
individually).

API
````
.. automodule:: sodetlib.operations.uxm_setup
    :members: 

.. _relock:

Relocking
----------

When resetting a system using an existing tune, we use the ``uxm_relock``
function, which uses existing device cfg parameters to get you into the state
where you're ready to take data. This is much faster because generally you
don't need to estimate attens, run setup notches, or figure out the tracking
parameters for each band.

Other than that, the relock procedure is similar to the setup procedure
described above:

  1. First amplifiers are setup using the same function
  2. Then the tunefile in the device cfg is loaded and we enable them with 
     ``relock`` or ``setup_notches``, and run the serial gradient descent and
     eta scan functions.
  3. Then we run ``tracking_setup`` based on parameters stored in the device
     cfg.
  4. Finally we take a noise timestream to verify that things are set up
     properly

To do this you can use the ``uxm_relock`` function

.. code-block:: python

    from sodetlib.operations.uxm_relock import uxm_relock

    success, summary = uxm_relock(S, cfg, bands=np.arange(8))

If band-medians give reasonable results (see the Noise section above)
you're good to go! 

Note that if ``setup_notches`` is run, a new tracking file will be created,
and if ``new_master_assignment`` is set to True, there is a small possibility
that the smurf channel assignments are not the same as the previous tune. 
If this is the case, you will probably want to retake a bgmap and an IV to get
data that corresponds to the current channel assignment.

API
````

.. automodule:: sodetlib.operations.uxm_relock
    :members: 

.. _tracking:

Tracking
----------

When we run ``setup_tracking`` we are configuring variables that SMuRF uses to
track resonators and extract the TES current. This is an essential part of the
setup procedure, and a common place for things to go wrong.

When setting up tracking, we are setting three main parameters:

 - The frequency of the flux-ramp wave (``reset_rate_khz``)
 - The amplitude of the flux-ramp wave (as a fraction of the max value output
   by the DAC: ``frac_pp``)
 - The squid modulation frequency being tracked by smurf for each band
   ``lms_freq``.

Setup Tracking Params
``````````````````````

The ``setup_tracking_params`` function will pull the desired flux-ramp freq
and :math:`N \Phi_0` (the number of squid-curve periods desired for each
channel) and calculate the optimal flux-ramp amplitude for each individual
band.

Tracking parameters can be configured through the device cfg object (see
``setup_tracking_params`` docstring for more details). For instance, if you
want to change the fraction full scale used to perform the initial lms_freq
estimation, you can run

.. code-block:: python

    from sodetlib.operations import tracking

    bands = np.arange(8)
    cfg.dev.exp['init_frac_pp'] = 0.45
    res = tracking.setup_tracking_params(S, cfg, bands)


Relock Tracking Setup
``````````````````````

Once optimal tracking params have been determined for each band individually,
these can be used to compute the tracking parameters that should be used if you
want to run with a combination of bands, or if you want to run at other
flux-ramp frequencies or :math:`N \Phi_0` values.

For instance, the following block will set up tracking using the optimal
parameters for band 0:

.. code-block:: python

    from sodetlib.operations import tracking

    bands = [0]
    res = tracking.relock_tracking_setup(S, cfg, bands)

And this will set up tracking for all bands running with a flux-ramp rate of 20
kHz and :math:`N \Phi_0 = 1`


.. code-block:: python

    from sodetlib.operations import tracking

    bands = np.arange(8)
    res = tracking.relock_tracking_setup(
        S, cfg, bands, reset_rate_khz=20, nphi0=1
    )

Tracking Results and Cutting Channels
``````````````````````````````````````

Both ``setup_tracking_params`` and ``relock_tracking_setup`` save and return a
``TrackingResults`` object, which contains the tracking data for all channels
on the specified bands. Along with the actual tracked-freq ``f`` and freq-error
``df`` for every channel, this will contain the peak-to-peak amplitude of ``f``
and ``df``, and the tracking-quality value ``r2``. The "tracking quality" is
the r-squared value between the measured freq response, and the average freq
response across all flux-ramp resets, giving a measure on how regular the
tracking response is over time. These values are used to determine which
channels are good and which should be cut.

To view the summary plot corresponding to a results object, you can run:

.. code-block:: python

    from sodetlib.operations import tracking

    bands = np.arange(8)
    res = tracking.relock_tracking_setup(
        S, cfg, bands, reset_rate_khz=20, nphi0=1
    )
    tracking.plot_tracking_summary(res)

which will produce a plot like this:

.. image:: ../../_static/images/tracking_summary.png
  :width: 700
  :alt: Alternative text

If most of the points in are blue and centered around the green region,
everything's probably working properly! The ``f_ptp_range``, 
``df_ptp_range``, and ``r2_min`` keys in the device cfg can be modified to
adjust what channels are accepted.

To investigate the behavior of an individual channel, you can run:


.. code-block:: python

    band, channel = 0, 10
    idx = np.where(
        (res.bands == band) & (res.channels == channel)
    )[0][0]
    tracking.plot_tracking_channel(res, idx)

which will produce the plot below.

.. image:: ../../_static/images/tracking_channel.png
  :width: 700
  :alt: Alternative text

Disabling Bad Channels
``````````````````````````

To turn off the "bad channels" determined by the tracking results object
you can use the ``disable_bad_chans`` function:

.. code-block:: python

    from sodetlib.operations import tracking

    bands = np.arange(8)
    res = tracking.relock_tracking_setup(
        S, cfg, bands, reset_rate_khz=20, nphi0=1
    )

    tracking.disable_bad_chans(S, res)


API
````
.. automodule:: sodetlib.operations.tracking
    :members: 

