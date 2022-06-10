Squid Curves
============

The squid_curves module takes data in open loop (only slow integral tracking) 
and steps through flux values to trace out a SQUID curve. This can be compared 
against the tracked SQUID curve which might not perfectly replicate this if 
these curves are poorly approximated by a sine wave (or ~3 harmonics of a 
fourier expansion).

The user can then plot individual channels and the fit associated via the
plot_squid_fit function. The user may also plot summary statistics using 
the plot_squid_fit_summary function.

Data Taking and Plotting Example
`````````````````````````````````
This example steps through 50 bias voltages around 0.3 flux ramp peak-to-peak
value and takes one data point of SQUID data at each bias voltage. This is done
over all channels on band 1. Then it generates a SQUID fit plot for the
10th channel in the first band, and the last line creates summary plots for all 
channels in all bands. The output data product from take_squid_curve is written
to disk. 

Note, the user can instantiate two variables when calling take_squid_curve:
rawdat, the rawdata output from calling take_squid_curve, and fit_dict, 
a dictionary that contains all of the fitting information. Here, we load 
the SQUID curve data written to disk and plot fits using that. We're able to
load in the raw data because take_squid_curve writes the raw data to disk
(location stored in the raw data output).

Plots are displayed but not saved::

  from sodetlib.operations import squid_curves
  from sodetlib.det_config import DetConfig
  import numpy as np

  cfg  = DetConfig()
  cfg.load_config_files(slot=3)
  S = cfg.get_smurf_control(make_logfile=True)
  
  rawdat = squid_curves.take_squid_curve(
    S, cfg, Npts=1, Nsteps=50, bands=[1],
    run_analysis=True, show_pb=True)

  fname = rawdat[0]['filepath']
  data = np.load(fname, allow_pickle=True).item()
  
  squid_curves.plot_squid_fit(*data, 1, S.which_on(1)[10])
  squid_curves.plot_squid_fit_summary(data)


API
````

.. automodule:: sodetlib.operations.squid_curves
   :members: take_squid_curve, get_derived_params_and_text, dfduPhi0_to_dfdI,
            fit_squid_curves, plot_squid_fit, plot_squid_fit_summary,
            squid_curve_model, estimate_fit_parameters, autocorr
    :noindex:

