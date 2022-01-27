Data Streaming
---------------

Each data stream on a site is uniquely specified by two ids.
The ``stream_id``, which is the id given to a particular smurf-slot. Often this
will be the the uxm assembly ID, but currently defaults to something like
``crate1slot2``.
The ``session_id`` is the id given to a particular session for a given slot. This
is determined by the smurf-streamer, and is simply the timestamp that the first
G3Frame was received.

Data Taking
````````````
Sodetlib provides two methods for taking data.
The ``take_g3_data`` function can be used to stream data for a specified length of time.
The ``stream_g3_on`` and ``stream_g3_off`` functions can be used to and stop a stream manually, which
can be used to take data for indeterminite amounts of time.

All three of these functions return the ``session_id`` of the stream they
corresond to.
The ``stream_id`` can be accessed through ``cfg.stream_id`` using the DetConfig
object which will always be defined for smurf operation.

.. note::

  The streaming methods defined in pysmurf will still generate G3 files,
  so any data streams produced by pysmurf functions or older sodetlib functions
  will be archived and loadable through the sodetlib load methods, though you
  may need to search out the corresponding stream-id.

  However, by default the sodetlib stream methods no longer generate .dat files
  (the native pysmurf format) to save space on smurf-servers. If you want
  to save .dat files as well as G3, you can run the sodetlib stream methods
  with ``make_datfile=True``.

Data Loading
``````````````

Sodetlib also provides functions for loading data streams into axis-managers
using the sotodlib load_smurf framework.
To load a particular stream, you can simply call ``load_session``
which takes the ``stream_id`` and ``session_id`` as input, and will
return an axis-manager for the corresponding stream session.
If running on a system that is not a smurf-server such as simons1,
the ``base_dir`` option will need to be passed with the base directory of
timestreams, since the default is only true for smurf-servers.

To get the ``SmurfStatus`` of a given session, you can use the
``load_session_status`` function which takes the ``stream_id``, ``session_id``,
and optionally the ``base_dir`` and will return the corresponding SmurfStatus
object which has relevent metadata info from the g3 stream.



Example
````````

Here's a simple example of taking and loading data using the sodetlib functions::

  import sodetlib as sdl
  from sodetlib.det_config import DetConfig

  cfg  = DetConfig()
  cfg.load_config_files(slot=2)
  S = cfg.get_smurf_control()

  sid = sdl.take_g3_data(S, 30)
  am = sdl.load_session(cfg.stream_id, sid)
  status = sdl.load_session_status(cfg.stream_id, sid)


API
````

.. automodule:: sodetlib.stream
   :members:

