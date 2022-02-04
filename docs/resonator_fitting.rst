Resonator Fitting
-----------------
This package implements fitting the complex transmission of the resonators to a
functional form to extract physical parameters which can feed back to resonator
fab and be a strong indicator of how well performing one set of readout is
compared to another. The primary package takes as input a tunefile which is a
dictionary that contains complex transmission for all resonators found during
tuning (``find_freq`` and ``setup_notches``). It then fits the complex transmission
of each resonator to a model from equation 11 in `Kahlil et al 2011`_

.. _Kahlil et al 2011: https://arxiv.org/abs/1108.3117

.. math::

   S_{21} = (1 + \hat{\epsilon})\left(1 - \frac{Q\hat{Q_e}^{-1}}{1+2iQ\frac{\omega-\omega_0}{\omega_0}}\right)


Additionally a cable loss and delay is multiplied by this function which is
required to fit to since we have significant loss and delay from the rf cabling
in our data. This ends up being a 9 parameter fit.

Fitting Tunefile Example
````````````````````````
To get the a dictionary with all of the parametric fit results assuming you
already have generated a tunefile that is at filepath ``tune_fpath``::

  import sodetlib.resonator_fitting as resfit

  fit_dict = resfit.fit_tune(tune_fpath)


Plotting Fit Results Example
````````````````````````````
There are some functions to plot the resultant fit parameters either of a single
channel S21 data vs model or summary histograms.

Channel Plotting
''''''''''''''''
To make a plot showing data a fitted result for a resonator in smurf band ``band``
and channel ``channel`` assuming your tunefile is at the path ``tune_fpath``, and you
already have a fit results dictionary called ``fit_dict`` you can run::

  import sodetlib.resonator_fitting as resfit

  resfit.plot_channel_fit(tune_fpath, fit_dict, band, channel)

Summary Plotting
''''''''''''''''
To make some histograms showing some key fit parameters namely internal quality
factor (``Q_i``), channel bandwidth (``BW``), dip depths (``depth``), and
channel-to-channel frequency separation you can run::

  import sodetlib.resonator_fitting as resfit

  resfit.plot_fit_summary(fit_dict)


API
````

.. automodule:: sodetlib.resonator_fitting
   :members:
