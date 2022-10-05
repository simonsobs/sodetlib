Bias Steps and Bias Group Maps
================================

Bias steps are when we take small steps in the detector bias voltage and
measure the detector response. These are a fantastic tool for calibrating
detectors and can be used to estimate detector parameters such as resistance,
responsivity, effective time constant, and can also be used to quickly generate
a bias-group mapping.

Usage
-------

Generating a Bias Group Map
``````````````````````````````
To generate a bias-group map, use the function ``take_bgmap`` from the
bias_steps module. This will run the bias-steps function with parameters
optimal for generating a bias-group map, will create a bgmap file, and
save the filepath to the device cfg. This function returns a bias
step analysis object.

.. code-block:: python

    import sodetlib.operations as ops

    bsa = ops.take_bgmap(S, cfg)

To generate the bgmap, this function will set detectors to superconducting
and play a series of small bias-steps. Because the detectors are
superconducting, we expect a very clear bias-group correlation, and we expect
the estimated resistance to be very small.
By default, channels will be "unassigned" from the bgmap if the bg-correlation
factor is less than 0.9 or if the estimated resistance is larger than 10 mOhms
to avoid picking up crosstalk channels. 
These cuts can be modified by setting the corresponding keyword arguments manually
in the ``analysis_kwargs`` parameter, for example:

.. code-block:: python

    from sodetlib.operations import bias_steps

    analysis_kwargs = {
        'assignment_thresh': 0.5,
        'R0_thresh': 5e-3
    }
    bsa = bias_steps.take_bgmap(S, cfg, analysis_kwargs=analysis_kwargs)


The bgmap file stores key metadata along with information about which channels
were run to produce the map, the bias-group assignments of each channel,
and the polarity of each channel, or whether the squid response steps in the
same or opposite direction of the bias current. The bgmap file contains:

 - meta:
    Dictionary of metadata returned by the ``get_metadata`` funciton in
    ``sodetlib.util``
 - sid:
    Session-id of bgmap data
 - bands:
    Array containing the smurf-band number for each channel
 - channels: 
    Array containing the smurf-channel number for each channel
 - bgmap:
    Array containing the assigned bias group of each channel. This will be ``-1`` if unassigned.
 - polarity: 
    Array containing the polarity of the channel with respect to
    the bias line. This will be ``+1`` if the signal steps in the same direction as
    :math:`\Delta V` and ``-1`` if the signal steps in the opposite direction.
 

To load a bgmap for a set of channels, you can use the ``load_bgmap`` function.

.. code-block:: python

  import sodetlib as sdl

  sid = sdl.take_g3_data(S, 30)
  am = sdl.load_session(cfg.stream_id, sid)

  bgmap, polarity = sdl.load_bgmap(
      am.ch_info.band, am.ch_info.channel, cfg.dev.exp['bgmap_file']
  )


Running Bias Steps
````````````````````
The sodetlib function ``take_bias_steps`` can be used to take bias steps and
run the analysis to calculate the detector parameters mentioned above. The
default parameters *should* be good enough for the full analysis, for example, 
one can simply run

.. code-block:: python

    from sodetlib.operations.bias_steps import take_bias_steps

    bsa = take_bias_steps(S, cfg)

If all goes well, the returned object ``bsa`` is a fully analyzed instance of
the :ref:`BiasStepAnalysis` class. This means that all the detector parameter
fields (such as ``R0``, ``I0``, ``Pj``, ``tau_eff`` etc.) exist and can be
examined. If the analysis failed for whatever reason, the BiasStepAnalysis
class will still be returned, however most of the derived parameters will not
exist as attributes. However the axis manager will still be loaded (``bsa.am``)
so you can use that to investigate why the analysis may have failed.

You can re-run the analysis without retaking the data by running

.. code-block:: python

   bsa.run_analysis()

Sometimes playing with the analysis keywords (see the API for more details)
will give you better results.

Running in Low Current Mode
``````````````````````````````

It is possible to run this function in low-current-mode, which might be
desirable if switching to high-current-mode is causing excess heating for
whatever reason. Since the low-current-mode has a bias-line filter which
increases the signal decay time, in order to get an accurate reading of the TES
resistance / responsivity you need to increase the step_duration to at least a
couple of seconds. This will make the function take much longer (~2-3 minutes)
so this mode of operation may not be as useful. For example, this should work:

.. code-block:: python

    bsa = take_bias_steps(
        S, cfg, high_current_mode=False, step_duration=2,
        nsteps=5, analysis_kwargs={'step_window': 2}
    )

How it works
-------------
Data Taking
`````````````
With the default parameters, the ``take_bias_steps`` function will play
``nsteps`` on each of the specified bias groups one at a time.

Generating the Bias Group Map
``````````````````````````````

To generate the bias group assignment for a given channel, we measure the TES
change in phase :math:`\Delta \Phi` in response to the sweep bias steps. We add
:math:`\pm \Delta \Phi` for each of the "sweep steps" on a given bias group,
where we use + if it's a rising step and - for a falling step. If a channel has
a strong dependence on a particular bias group, the TES responses will co-add
and give you a large summed response, and if there is no dependence on a bias
group the rising and falling edges will cancel, even if there is a global trend
resulting in a small summed response. For a given channel, the summed responses
for each bias group are normalized such that they all add to 1, giving us a
"normalized correlation factor" (stored in the ``bg_corr`` array). 

When there is a very clear correlation, (like if the detector is
superconducting), there will be one bias group with a correlation value close
to 1 and the rest will be close to 0. If it is less clear, for instance if the
detectors are in transition and there is a non-linear trend in the detector
timestream (from heating for example), the maximum correlation factor will be
smaller (like 0.3-0.9) but the maximum will usually still give you the correct
bias group. If the maximum correlation factor is less than the
``assignment_thresh`` value, the channel is unassigned from all bias groups.

.. note::

   A channel that is perfectly uncorrelated to all bias groups will have a
   correlation factor of 1/NBiasGroups for each bias-group. Though this is not
   optimal for determining which bias groups are not connected to any
   bias-line, an additional cut is made after calculating the TES resistance,
   which is a better metric for determining which channels should be left
   unassigned. Any channel that has an estimated resistance larger than the
   ``R0_thresh`` parameter will be considered noise or crosstalk and unassigned
   from all bias-groups.


Estimating Detector Parameters
````````````````````````````````

First, the detector responses for each of the steps taken are averaged to give
us a "mean response" that has a large signal-to-noise ratio and can be used for
fitting and DC calculations.

There are two methods of calculating the DC detector parameters, one that
assumes the detector is well into transition and that the bias power is
constant over the step, and the other that assumes the detector is out
of the transition (either super-conducting or normal), and
:math:`R_\mathrm{TES}` is constant over the step.

The BiasStepAnalysis determines which method should be used based on the bias
voltage level. If the ``transition`` parameter is set to be a tuple (V0, V1),
then the analysis will use the transition method whenever V0 < Vbias < V1.
The parameter can also be set to true/false, which will force the analysis
to use the transition calculation.

In Transition
''''''''''''''

In the transition, it is assumed that the ratio of :math:`dI_\mathrm{rat} =
\frac{dI_\mathrm{TES}}{dI_\mathrm{bias}}` is negative due to the loop gain
being larger than 1. See Michael Niemack's thesis (Section 4.3.1) and Emily
Grace's thesis (Section 4.2.4) for the derivation. The bias power, TES current,
and TES resistance are then given by

.. math::

    P_J = \frac{I_\mathrm{bias}^2 R_\mathrm{sh} dI_\mathrm{rat}(dI_\mathrm{rat} - 1)}{(1 - 2 dI_\mathrm{rat})^2}

.. math::

   R_\mathrm{TES} = R_\mathrm{sh} \frac{
        I_\mathrm{bias} + \sqrt{I_\mathrm{bias}^2 - 4 P_J / R_\mathrm{sh}}
    }{
        I_\mathrm{bias} - \sqrt{I_\mathrm{bias}^2 - 4 P_J / R_\mathrm{sh}}
    }

.. math::

    I_\mathrm{TES} = \sqrt{P_J / R_\mathrm{TES}}
    = \frac{1}{2} \left(
    I_\mathrm{bias} - \sqrt{I_\mathrm{bias}^2 - 4 P_J / R_\mathrm{sh}}
    \right)


Out of transition
''''''''''''''''''''''
Outside of the transition, we assume that :math:`dI_\mathrm{rat} > 0`,
and using the assumption that R is constant over the step, we have:

.. math::

    R_\mathrm{TES} = R_\mathrm{sh}\left(
        \frac{dI_\mathrm{bias}}{dI_\mathrm{TES}} - 1
    \right)

.. math::
   I_\mathrm{TES} = \frac{I_\mathrm{bias} R_\mathrm{sh}}
   {R_\mathrm{TES} + R_\mathrm{sh}}

.. math::
   P_J  = I_\mathrm{TES}^2 R_\mathrm{TES}

API
-----

.. _take_bias_steps:

Take Bias Steps
`````````````````


.. automodule:: sodetlib.operations.bias_steps
    :members: take_bgmap, take_bias_steps


.. _BiasStepAnalysis:

BiasStepAnalysis
````````````````
The BiasStepAnalysis is the class containing all info pertaining to the bias
steps and analysis. It contains smurf parameters required to run the analysis
(such as ``S.high_low_current_ratio``, etc.), and analysis products.

Any analysis product that is per-detector will be stored in an array that is
``nchans`` long, where ``nchans`` is the number of channels being read out at
the time, and is indexed by the "readout channel number" (or index of the
channel in the axis-manager). The mapping from readout channel to absolute
smurf channel number (``band * 512 + channel``) can be found in the
``bsa.abs_chans`` attribute.

.. autoclass:: sodetlib.operations.bias_steps.BiasStepAnalysis
    :members: run_analysis

References
-----------
Below are some valuable references for information on bias steps and their
uses:

 - `Daniel Becker's thesis <https://ui.adsabs.harvard.edu/abs/2014PhDT........57B/abstract>`_ (Chapter 6)
 - `Michael Niemack's thesis <https://ui.adsabs.harvard.edu/abs/2008PhDT.........1N/abstract>`_ (Section 4.3.1)
 - `Emily Grace's thesis <https://www.proquest.com/docview/1766117824/abstract/29190104DAB64533PQ/1>`_ (Section 4.2.4)
