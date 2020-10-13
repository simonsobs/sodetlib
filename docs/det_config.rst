The DetConfig System
=====================

DetConfig is the sodetlib module that manages and groups all configuration info.

Loading a Configuration
````````````````````````
Below is an example for how to populate the ``DetConfig`` object for slot 2::

    from sodetlib.det_config import DetConfig

    cfg = DetConfig()
    cfg.load_config_files(slot=2)

When being run in a script, the DetConfig object will populate an Argparse
parser with its own command line arguments with the ``parse_args`` function.
This can be used as follows::

   from sodetlib.det_config import DetConfig

   cfg = DetConfig()
   cfg.parse_args()

This will pull the slot number and additional info from the command line. 
Details can be seen in the `DetConfig parser`_ and `API`_ sections.

Once the configuration has been loaded, the ``get_smurf_control`` function can
be used to create a pysmurf-instance based on the configuration info.  The
DetConfig object will automaticlly determine the epics root and set the
publisher ID based on the crate and slot number.  If ``dump_configs`` is set to
``True``, all configuration files will be copied to the pysmurf output
directory. For example::

   from sodetlib.det_config import DetConfig

   cfg = DetConfig()
   cfg.lod_config_files(slot=2)
   S = cfg.get_smurf_control(dump_configs=True)


.. _DetConfig parser:

DetConfig parser arguments
````````````````````````````
.. argparse::
   :module: sodetlib.det_config
   :func: make_parser
   :prog: python3 sodetlib_script.py

Using and Updating the Device Config
``````````````````````````````````````
The device configuration options are stored in the ``cfg.dev`` object,
split between the ``exp`` dictionary for "experiment" level configuraiton,
``bias_groups`` which is a list of 12 config dicts containing bias group options,
and ``bands`` which is a list of 8 dicts containing band configuration.

The methods ``update_experiment``, ``update_bias_group`` and ``update_band``
can be used to update these configuration objects with new or modified config
parmeters. Once updated, a new device config file can be written with the
``dump`` function. See the `API`_ section for function details.

Here is an example of a function that takes in a config object, modifies dome
device configuration options, and writes over the device file with the new
parameters::

   def optimize_drive_power(S, cfg, band):
      # Logic to find optimial drive power for current cfg params goes here:
      optimal_drive = get_optimal_drive()

      cfg.dev.update_band(band, {'drive': optimal_drive})
      cfg.dev.dump(cfg.dev_file, clobber=True)


.. _API:

API
````````

DetConfig
""""""""""""
.. autoclass:: sodetlib.det_config.DetConfig
    :members: parse_args, get_smurf_control, load_config_files

DeviceConfig
""""""""""""""
.. autoclass:: sodetlib.det_config.DeviceConfig

