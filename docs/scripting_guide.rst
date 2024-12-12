Scripting Guide
=================

In this section we'll break down the basic steps needed to write your own 
sodetlib functions and scripts.

An sodetlib function will almost always have the following structure:

1. Pull configuration parameters from the ``cfg`` object and the function
   arguments
2. Run a set of pysmurf functions.
3. Update the cfg object with new or optimized information.

To walk us through the process, I'll go over the ``find_subbands`` function
and describe each section of the program.

Writing an sodetlib function
------------------------------

Lets start by writing the function signature and docstring so its clear what 
we want the function to do.
Basically every function will require the SmurfControl and DetConfig objects
to be passed in. ::


    from pysmurf.client.util.pub import set_action

    @set_action()
    def find_subbands(S, cfg):
        """
        Do a noise sweep to find the coarse position of resonators.
        Return a list of active bands and a dictionary of the active subbands
        in each band.

        Parameters
        ----------
        S : pysmurf.client.SmurfControl
            Smurf control object
        cfg : DetConfig
            sodetlib config object

        Returns
        -------
        bands : int array
            Active bands
        subband_dict : dict
            A dictionary containing the list of subbands in each band.
        """
        bands = []
        subband_dict = {}

        # Put Function logic here

        return bands, subband_dict

.. note::
    ``set_action`` is a decorator that sets the current "action" of the pysmurf
    publisher to be the name of the function.  In this case, all outputs and
    plots will be tagged with the ``action`` "find_subbands", and so they will
    be archived together in the directory ``<timestamp>_find_subbands``.

    In order for this to work, the ``SmurfControl`` object must be the first
    parameter of the function. This decorator should go on pretty much every
    sodetlib function.

Next we should figure out any configuration info that we may need. In this
case, we need to determine which bands are enabled. This can be calculated 
through the ``SmurfControl`` object by determining which bays are active as
follows::

    
    amc = S.which_bays()
    if 0 in amc:
        bands += [0, 1, 2, 3]
    if 1 in amc:
        bands += [4, 5, 6, 7]

Now we can use the SmurfFunctions ``full_band_resp``, ``find_peak``, and
``freq_to_subband`` to determine what subbands have resonances in them in each
band::

    for band in bands:
        freq, resp = S.full_band_resp(band)
        peaks = S.find_peak(freq, resp, make_plot=True, show_plot=False,
                            band=band)
        fs = np.array(peaks*1.0E-6) + S.get_band_center_mhz(band)

        subbands = sorted(list({
            S.freq_to_subband(band f)[0] for f in fs
        }))
        subband_dict[band] = subbands


Inside this loop, we'll want to update the device config with the new
information. We'll set the ``active_subbands`` field to be the list of 
subbands that was found. See :ref:`det_config` for more info::

    cfg.dev.update_band(band, {'active_subbands': subbands})

That's pretty much it! The full function now looks like::


   @set_action()
   def find_subbands(S, cfg):
       """
       Do a noise sweep to find the coarse position of resonators.
       Return a list of active bands and a dictionary of the active subbands
       in each band.
   
       Parameters
       ----------
       S : pysmurf.client.SmurfControl
           Smurf control object
       cfg : DetConfig
           sodetlib config object
   
       Returns
       -------
       bands : int array
           Active bands
       subband_dict : dict
           A dictionary containing the list of subbands in each band.
       """
       subband_dict = {}
       bands = []

       amc = S.which_bays()
       if 0 in amc:
           bands += [0, 1, 2, 3]
       if 1 in amc:
           bands += [4, 5, 6, 7]
       if not bands:
           print('No active AMC')
           return bands, subband_dict

       for band in bands:
           freq, resp = S.full_band_resp(band)
           peaks = S.find_peak(freq, resp, make_plot=True, show_plot=False,
                               band=band)
           fs = np.array(peaks*1.0E-6) + S.get_band_center_mhz(band)

           subbands = sorted(list({
               S.freq_to_subband(band f)[0] for f in fs
           }))
           cfg.dev.update_band(band, {'active_subbands': subbands})

       return bands, subband_dict

Of course when testing you may find unexpected behavior that you may have to 
deal with. For the ``find_subbands`` function we found that we were mistaking
spurs in the filter response as resonator peaks, and so we had to add some 
logic to discard frequencies if they are too close to these spurs. The full
code can be seen on the sodetlib github.

Writing an sodetlib script
----------------------------
Once the function is written, writing a script that calls the function is 
simple. The script should pretty much do the following:

1. Create the DetConfig object.
2. (optional) Add command line arguments through argparse
3. Parse args and get the SmurfControl Object
4. Call the function
5. Dump the updated to the device file.

For the ``find_band`` script, this can be done as follows::

   import argparse
   from sodetlib.det_config import DetConfig
   from sodetlib.smurf_funcs import find_subbands
   import os


   if __name__ == '__main__':
       cfg = DetConfig()
       parser = argparse.ArgumentParser()
       # This is where we would add additional command line arguments if needed
       # See other sodetlib scripts for examples on how to do this.

       args = cfg.parse_args(parser)
       S = cfg.get_smurf_control(dump_configs=True)

       bands, subband_dict = find_subbands(S, cfg)
       dev_file = os.path.abspath(os.path.expandvars(cfg.dev_file))
       cfg.dev.dump(dev_file, clobber=True)

.. note::
   I use the ``abspath`` functions and ``expandvars`` functions when getting
   the dev_file path because I like to point to the device config file relative
   to the ``$OCS_CONFIG_DIR`` environment variable, which will work both inside
   and outside of the docker environment.

