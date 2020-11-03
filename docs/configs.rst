The Configuration Hierarchy
============================

SMuRF is a complex system with many configuration parameters which vary
depending on the cryostat, the experiment being conducted, and the network
configuration of the smurf-server and data-acquisition nodes.

These config parameters will be stored in various files in the
``OCS_CONFIG_DIR``, where institutions will maintain and version-control the
configuration of their systems. In the following sections we will go over what
is contained in each file.  The hierarchy of configuration files is as follows:

sys_config
    The `sys_config` file is top-level system configuration. It contains
    general system info such as the crate-id, active slots, and docker image
    tags for all the software being used. In addition, it points to which
    lower-level config files should be used for each slot. 

pysmurf config files
    The pysmurf config files set certain pysmurf settings and rogue registers.
    This file is loaded in the pysmurf ``setup`` function, which will usually
    be run once per reboot.

device config files
    The device config files contain many of the parameters that are passed to
    pysmurf function calls.  These values are optimized per system and
    experiment, so each system-slot-experiment combination will have its own
    file. These parameters are used and updated throughout testing in sodetlib
    functions and scripts.


Sys Config
-----------
The system configuration file is located in ``$OCS_CONFIG_DIR/sys_config.yml``
and contains general system configuration, certain slot-specific configuration,
and docker environment variables. See `here`_ for a template.

.. _here: 
   https://github.com/simonsobs/ocs-site-configs/blob/master/templates/smurf-srv/sys_config.yml

General System Configuration
``````````````````````````````
The following parameters can be set in the top-level of the yaml file

.. list-table:: General System Configuration
   :widths: 10 80

   * - ``crate_id``
     - ID of the smurf-crate, e.g. ``1``
   * - ``shelf_manager``
     - Shelf manager name
   * - ``max_fan_level``
     - Max Fan Level to be set on reboot
   * - ``comm_type``
     - Communication type to the crate -- either ``eth`` or ``pcie``
   * - ``meta_register_file``
     - Path to the meta registers file. This is where we set what metadata
       should be streamed to G3 files.
   * - ``slot_order``
     - A list containing all active slots in the order that they should be 
       started.

Slot Configuration
````````````````````
Slot configuration will grouped under the ``slots`` key, and each slot will 
have its own entry ``SLOT[<slot>]``.

.. list-table:: Slot Configuration
   :widths: 10 80

   * - ``stream_port``
     - Port to stream G3Frames. Currently using ``453<slot>``.
   * - ``pysmurf_config``
     - Path to the pysmurf_config file for this slot
   * - ``device_config``
     - Path to the device_config file for this slot


Docker Environment
````````````````````
Any keys in the ``docker_env`` section will be copied to ``$OCS_CONFIG_DIR/.env``
whenever ``jackhmmer hammer`` is run, which will treat them as environment variables
inside the docker-compose file. These are used to set the docker image tags which will be used.

Required variables are:

.. list-table:: Docker environment variables
   :widths: 10 80

   * - ``STREAMER_TAG``
     - Tag of the `smurf-streamer docker`_ image
   * - ``SOCS_TAG``
     - Tag of the `socs docker`_ image
   * - ``SODETLIB_TAG``
     - Tag of the `sodetlib docker`_ image
   * - ``CB_HOST``
     - Address of the crossbar host from the smurf-server

.. _`smurf-streamer docker`: 
   https://hub.docker.com/r/simonsobs/smurf-streamer/tags

.. _`socs docker`: 
   https://hub.docker.com/r/simonsobs/socs/tags

.. _`sodetlib docker`: 
   https://hub.docker.com/r/simonsobs/sodetlib/tags

Pysmurf Config
---------------
The pysmurf configurtion file is loaded through a pysmurf instance when 
`S.setup` is run. There will be one per slot per site, and this should not 
change once it has been configured. For more details, see `the pysmurf readthedocs`_

.. _`the pysmurf readthedocs`:
   https://pysmurf.readthedocs.io/en/main/user/configuration.html

Device Config
--------------
The device config file has three main sections. ``experiment``, ``bias_groups``
and ``bands``. See here for an `example device config file`_.  Specific
experiment, bias group, and band configuration variables are not set in stone,
but certain sodetlib scripts will sometimes require specific config variables
to exist.  Currently there are no restrictions on what config variables can be
added, and sodetlib script writers should feel free to add new ones if they
think it would be useful for their script.

.. _`example device config file`:
   https://github.com/simonsobs/ocs-site-configs/blob/master/templates/smurf-srv/device_configs/dev_cfg_s2.yaml

The ``experiment`` section contains experiment-level configurations that are
not specific to a band or bias groups, such as ``amp_50k_Id``, and
``tune_file``.

The ``bias_groups`` section contains configuration per-bias group.  Each key in
this section must be a list of 12 items, one value per bias group.  Examples
include ``bias_high``, ``bias_low``, ``bias_step``, ``enabled`` etc.

The ``bands`` section contains config info on each band. There are two
subsections, one for each AMC labeled ``AMC[0]`` and ``AMC[1]``.  Each AMC
subgroup contains band-specific configuration, where each key must be a list of
four values, one for each band in the AMC.  Examples include ``dc_att``,
``drive``, ``flux_ramp_rate_khz``, etc.

