
IVs
===

IV curves are fundamental to TES calibration and characterization of detector
properties. An IV curve is taken by applying a strong bias to drive TES's into
their normal regime, and then by slowly lowering the bias voltage in steps
allowing the TES to pass through the normal, in-transition, and superconducting
states.

Through IVs we are able to obtain TES resistance, which is essential for
biasing detectors into their transition, along with the bias power, and an
estimate of the DC responsivity.

Though the current response we measure is relative, we can use the fact
that the TES current in both the normal and superconducting branches 
are linear with bias-current, with a y-intercept of 0 to back out the real
current at each point in the transition.


Taking IVs
-------------

To take an IV, use the ``take_iv`` function in the ``sodetlib/operations/iv`` module.

.. code-block:: python

   from sodetlib.operations import iv

   iva = iv.take_iv(S, cfg)

To take a good IV it important that:

 - The detectors start off properly overbiased. 

 - Each step allows detectors to settle for a good measurement. The default
   runs in high-curren-mode to avoid the bias-line filter, with wait-times of
   0.1 sec to achieve this.

 - Steps in bias voltage are small enough to avoid tracking phase-skips. Large
   jumps in squid current will mess up the tracking algorithm which may offset
   the response by a multiple of :math:`2 \pi`. This is difficult to correct
   for reliably in analysis so it is important for current steps to be small.

The default parameters of the take_iv function were chosen to satisfy these conditions,
however depending on your assembly you may need to tweek the parameters such as
``overbias_voltage``, ``overbias_wait``, ``bias_high``, ``bias_step``, and
``wait_time``. See the the :ref:`IV_API` for the take_iv function for more details.


The ``take_iv`` function returns an instance of the ``IVAnalysis`` class,
containing all the important information about the run-conditions of the IV,
and the analysis products if ``run_analysis`` is set to ``True``.
By default, this function will save the output filepath of the IVAnalysis object
to the device config under the key ``iv_file``.
This can easily be loaded again by running:

.. code-block:: python

   from sodetlib.operations import iv

   iva = iv.IVAnalysis.load(cfg.dev.exp['iv_file'])

Using the ``analyze_iv`` function, you can re-analyze the iv object, or pass
it to the ``plot_channel_iv`` function to plot the analysis products of a
given readout-channel.

.. note::

   Though the IV can be run in either high or low-current mode, the voltage
   biases specified in the ``take_iv`` keyword arguments, and the voltage
   bias values in the IVAnalysis object are **all** in low-current-mode
   units, so that biases can be easily compared and used without knowing
   which mode the IV was taken in.



.. _IV_API:

API
----

.. automodule:: sodetlib.operations.iv
    :members: take_iv

.. autoclass:: sodetlib.operations.iv.IVAnalysis
    :members:

Biasing Detectors into Transition
====================================

Once you have your bias-group mapping and your IV file, you are able to bias
detectors into their transition. There are two functions that you may use to do
this:

Biasing to Rfrac Range
----------------------
The function ``bias_to_rfrac_range`` will take in a range of Rfrac values,
and use the IV analysis objects to determine the bias voltages which maximize
the number of detectors which fall in that range. By default, this range is set
to (0.3, 0.6) based on the target operating resistances for SO MF detectors.

.. code-block:: python

   from sodetlib.operations import iv

   iv.bias_to_rfrac_range(S, cfg)

The overbiasing options ``overbias_voltage`` and ``overbias_wait`` can be
specified as keyword arguments.
The function will also base the determination only on channels that have
"reasonable" values for their normal resistance. The range of acceptable
Rn's can be set using the ``Rn_range`` parameter, and defaults to
``(5e-3, 12e-3)``.

Biasing to Rfrac
----------------------

The functino ``bias_to_rfrac`` is a lot like the function ``bias_to_rfrac_range``,
except that it takes a single Rfrac value (defaulting to 0.5), and determines
the bias-voltage by taking the median of what voltage is required for each 
channel to achieve that Rfrac. This is generally less useful than
``bias_to_rfrac_range``.

API
----
.. automodule:: sodetlib.operations.iv
    :noindex:
    :members: bias_to_rfrac_range, bias_to_rfrac







