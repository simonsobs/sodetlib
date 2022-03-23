from ._version import get_versions
__version__ = get_versions()['version']
del get_versions

import os
from functools import wraps
try:
    import epics
    from pysmurf.client.util.pub import set_action
except:
    # Just replace set_action with functool.wraps if we
    # can't import pysmurf
    set_action = lambda *args, **kwargs: wraps
    os.environ['NO_PYSMURF'] = 'true'


from sodetlib.util import (
    make_filename, pub_ocs_data, pub_ocs_log, load_bgmap, map_band_chans,
    get_metadata, get_tracking_kwargs, get_psd, get_asd, Registers,
    get_current_mode_array, set_current_mode, validate_and_save, set_session_data
)

from sodetlib.stream import (
    get_session_files, load_session_status, load_session,
    take_g3_data, stream_g3_on, stream_g3_off
)

from sodetlib import noise

