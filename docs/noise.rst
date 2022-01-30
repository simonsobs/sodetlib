Noise Model
------------
Noise data refers to processing of time ordered data to extract statistics
about the noise in the timestream. The core function is to take a power spectrum
using the ``scipy.welch`` function and then extract the white and low frequency
components of the resultant power spectrum. All outputs are returned in units
of pA/rtHz which is the sqrt of the psd returned by the ``welch`` algorithm and
sometimes the amplitude spectral density (ASD in pA/rtHz) is referred to as the
power spectrum colloquially (we tried to avoid this confusion in the docs but
be aware of this inprecise language). The white component is reported as the
amplitude of the flat section of the ASD and is either determined by fitting to
a model containing a white and low-f component or by taking the median of the
ASD between f_min and f_max which default to 10 and 30 Hz respectively. The
low-f component is characterized by the f_knee which is the point where the
low-f and white components in the power spectrum (here we really mean PSD) are
equal. We find this point by either fitting to a model that contains both
components or binning + averaging the ASD below the region used to
calculate the white component and finding where it rises to sqrt(2)*(white noise).
This doesn't fully constrain the low-f component because f_knee will scale
with the white noise level and the steepness (or spectral index i.e. 1/f^n) is
still unknown. To constrain this we additionally report the value of the ASD at
0.01 Hz relative to a baseline required ASD spectrum (215 pA/rtHz at 0.01 Hz
corresponding to a white noise = 65 pA/rtHz and the ASD spectral index of the
low-f component is 1/f^{1/2}) and in the case of using the fitting option we
also report the fitted spectral index n.

Data Taking Example Usage
``````````````````````````
This example takes 5 minutes of noise data (to get a good measure of 1/f you
really want a longer timestream like this but can process shorter data as well)
on all channels turned on attached to the smurf blade in slot 2. Then it also
generates band summary plots and a channel plot for readout channel 0. Plots are
displayed but not saved::

  from sodetlib import noise
  from sodetlib.det_config import DetConfig

  cfg  = DetConfig()
  cfg.load_config_files(slot=2)
  S = cfg.get_smurf_control()

  noise.take_noise(S, cfg, acq_time=300, plot_band_summary=True,
                   plot_channel_noise=True, show_plot=True, save_plot=False,
                   rchans=[0])


Analyzing Existing Data
````````````````````````
This example assumes that you have your data in an AxisManager loaded using
`sodetlib.load_session()` or `sotodlib.io.load_smurf.G3tSmurf`::

  from sodetlib import noise

  #am is the AxisManager with your noise data.
  #Can set fit=True but will take longer
  outdict = noise.get_noise_params(am,fit=False)
  #Make band summary plots
  _ = noise.plot_band_noise(am,noisedict=outdict)
  #Display but don't save a channel plot for readout channel 0.
  rc = 0
  fig,ax = noise.plot_channel_noise(am, rc, noisedict=outdict, show_plot=True,
                                    save_plot=False)

API
````

.. automodule:: sodetlib.noise
   :members:
