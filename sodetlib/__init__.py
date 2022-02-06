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


from sodetlib.util import *
from sodetlib.stream import *

from sodetlib import noise

