The Configuration Hierarchy
============================

SMuRF is a complex system with many configuration parameters which vary
depending on the cryostat, the experiment being conducted, and the network
configuration of the smurf-server and data-acquisition nodes.

These config parameters will be stored in various files in the configuration
directory, where institutions will maintain and version-control the
configuration of their systems.

The configuration directory may or may not be the same as the
``OCS_CONFIG_DIR``, which contains the OCS site-config file. If it is different,
the correct configuration path can be set separately using the
``SMURF_CONFIG_DIR`` environment variable.

In the following sections we will go over what is contained in each file.  The
hierarchy of configuration files is as follows:

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
The system configuration file is located in the ``sys_config.yml`` file
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
   * - ``smurf_fans``
     - Dict used to set fan-policy and speed on hammer
   * - ``comm_type``
     - Communication type to the crate -- either ``eth`` or ``pcie``
   * - ``meta_register_file``
     - Path to the meta registers file. This is where we set what metadata
       should be streamed to G3 files.
   * - ``slot_order``
     - A list containing all active slots in the order that they should be 
       started.
  
Fan Configuration
```````````````````
The ``smurf_fans`` key is a dict which can be used to set the fan policy and
speed during hammer, or with the ``jackhammer setup-fans`` command.

Before configuring you need to determine the address of the fans on the smurf. 
To do this, ssh into the shelf-manager, and run the following:

.. code-block:: bash

  # clia fans
  > Pigeon Point Shelf Manager Command Line Interpreter
  > 20: FRU # 4
  > Current Level: 14
  > Minimum Speed Level: 1, Maximum Speed Level: 100 Dynamic minimum fan level: 11
  > 20: FRU # 3
  > Current Level: 14
  > Minimum Speed Level: 1, Maximum Speed Level: 100

This tells us that there are two configurable fans, one with address 20 and FRU-id 4, and the other with address 20 and FRU-id 3.
This also tells us that the maximum speed of the fan is 100.

Our sys-config entry under ``smurf_fans`` should look like

.. code-block:: yaml

  smurf_fans:
    addresses:
      - [<address-1>, <fru_id-1>]
      - [<address-2>, <fru_id-2>]
    speed: <speed>
    policy: <policy>

Here, ``speed`` is the speed it will set all fans to, ``policy`` can be
``enable`` or ``disable``, where ``enable`` will have the fan-controller
auto-adjust the speed based on the junction temps, and disable will keep the fan
speed fixed.

It can be dangerous to disable the fan-controller if you are not setting fans to
their max speed, as the controller will have no ability to increase the speed to
keep smurf cool if they need to.

However, keeping the fan speeds fixed at their max speed can be beneficial, as
fan-speed changes cause sudden changes FPGA temp and electronics temp, which make its way into our data.

For operating the SAT (and for most SO smurfs), the max-fan speed is 100, so our configuration looks like:

.. code-block:: yaml

  smurf_fans:
    addresses:
      - [20, 3]
      - [20, 4]
    speed: 100
    policy: disable



Slot Configuration
````````````````````
Slot configuration will grouped under the ``slots`` key, and each slot will 
have its own entry ``SLOT[<slot>]``.

.. warning::
    Are the stream ports used at all anymore?

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
Any keys in the ``docker_env`` section will be copied to the ``.env`` file in
the configuration directory.  Whenever ``jackhmmer hammer`` is run, which will
treat them as environment variables inside the docker-compose file. These are
used to set the docker image tags which will be used.

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

