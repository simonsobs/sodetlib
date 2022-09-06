Complex Impedance
===================

The complex impedance measurement is a simple measurement that gives us access
to many electrothermal properties of the TES.

Following the theory presented in :ref:`Irwin and Hilton<ci_ref>`,
the complex impedance is given by:

.. math::

  Z_\mathrm{tes}(\omega) = R(1 + \beta_I) + 
  \frac{R \mathscr{L}_I}{1 - \mathscr{L}_I}
  nf.
  \frac{2 + \beta_I}{1 + i \omega \tau_I}

where :math:`\beta_I = \left.\frac{d \log R}{d\log I}\right|_{T_\mathrm{bath}}`
is the current sensitivity at constant temp, :math:`\mathscr{L}_I` the loop-gain,
and :math:`\tau_I` the thermal time-constant at constant current.
We are able to fit these parameters to our complex impedance measurement,
and obtain derived parameters such as the effective thermal time-constant:

.. math::

  \tau_\mathrm{eff} = \tau_I (1 - \mathscr{L}_I)
  \left(1 + 
    \frac{(1 - R_\mathrm{sh}/R) \mathscr{L}_I}{1 + \beta_I + R_\mathrm{sh}/R}
  \right)^{-1}


Measurement
------------

The complex impedance can be measured via the complex transfer functions of the
TES as a function of frequency. To do this we play a sine-wave on the bias line
over a large frequency range, and measure the amplitude and phase (with respect
to the commanded bias) as a function of frequency.

In order to remove parasitic contamination from the bias circuitry, we use
measurements of the TES response while the detectors are in their
superconducting and normal states to obtain the Thevenin equivalent voltage and
impedance of the bias circuit (see :ref:`Lindeman et al.<References>`). The
equivalent voltage and impedance are given by:

.. math::

   V_\mathrm{th}(\omega) = \frac{R_\mathrm{N}}
   {I_\mathrm{ob}(\omega) - I_\mathrm{sc}(\omega)}
   \qquad
   Z_\mathrm{eq}(\omega) = \frac{V_\mathrm{th}(\omega)}{I_\mathrm{sc}(\omega)}

and the TES impedance given by

.. math::

   Z_\mathrm{TES} = V_\mathrm{th} / I - Z_\mathrm{eq}

Where :math:`V_\mathrm{th}`, :math:`I_\mathrm{ob}`, :math:`I_\mathrm{sc}`,
:math:`I` and :math:`Z_\mathrm{eq}` are all complex phasors that are functions
of frequency.

Operation
-----------
The complex impedance measurement consists of two parts. First, measuring
the superconducting and overbiased transfer functions. This only needs
to be done once per cooldown, and then the results can be used for any Ztes
measurement. Then we measure the transfer functions while the detectors are in
transition.

These measurements all use the same base function, ``take_complex_impedance``
which streams data for each bias-group while playing sine waves at different
frequencies.

The function ``take_complex_impedance_ob_sc`` function will take CI data in the
overbiased and superconducting states, and save their paths in the device
config for later use.

For example, the following code will take SC, OB, and in-transition datasets:

.. code-block:: python

  import sodetlib.operations.complex_impedance as ci
  from sodetlib.operations import bias_dets

  freqs = np.logspace(0, np.log10(2e3), 80)
  bgs = np.arange(12)
  ci.take_complex_impedance_ob_sc(S, cfg, bgs, freqs=freqs, run_analysis=True)

  bias_dets.bias_to_rfrac_range(S, cfg, (0.3, 0.6))
  time.sleep(60)
  ds = ci.take_complex_impedance(S, cfg, bgs, freqs=freqs, run_analysis=True)

The output `ds` (for dataset) is an AxisManager containing a bunch of fields
including the complex-impedance for each detector. See the docstring for the
analysis functions ``analyze_tods``, ``get_ztes``, ``fit_det_params`` to see
what fields it contains.

On creation, the ``ds`` AxisManager is automatically loaded with
superconducting and overbiased data from the files that are set in the device
cfg. This means that the saved hdf5 file contains `all` the info needed for
analysis, and there is no need to manually add sc and ob data on load. For
instance, to load and plot data from a file, one can simply run::

  from sotodlib.core import AxisManager
  import sodetlib.operations.complex_impedance as ci

  ds = AxisManager.load('/path/to/trans.h5')
  # Plot transfer functions for channel with index 0.
  ci.plot_transfers(ds, 0)

.. note::
  The complex impedance dataset can only measure frequencies up to half the
  sampling rate. It can be very beneficial to measure out to higher
  frequencies to better constrain detector parameters, so if you are measuring
  high frqeuencies it is recommended to set a flux-ramp rate of 10 kHz or more
  before taking data. This can be done easily using the
  ``relock_tracking_setup`` function.

  However, if you are running at high sample rates, analysis may take upwards
  of 20 min or so to run per dataset (for a ufm with 12 biasgroups). This time
  is dominated by the time it takes to load datafiles with high sample rates
  from G3.

.. _ci_api:

API
----

.. automodule:: sodetlib.operations.complex_impedance
   :members: analyze_seg, analyze_tods, get_ztes, fit_det_params,
             take_complex_impedance, take_complex_impedance_ob_sc


References
----------

.. _ci_ref:
Useful references:
 - `Irwin / Hilton chapter on Transition Edge Sensors <https://link.springer.com/chapter/10.1007/10933596_3>`_
 - `Lindeman on correcting for stray impedances <https://doi.org/10.1063/1.2723066>`_
 - `Nick Cothard's paper on CI vs bias steps for SO prototype dets <https://arxiv.org/abs/2012.08547>`_



