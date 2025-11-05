from ._version import get_versions
__version__ = get_versions()['version']
del get_versions

import os
from functools import wraps
try:
    from pysmurf.client.util.pub import set_action
except Exception as e:
    # Just return base function regularly if can't import set_action
    def set_action(*args, **kwargs):
        def dec(func):
            return func
        return dec
    os.environ['NO_PYSMURF'] = 'true'


from sodetlib.util import *
from sodetlib.stream import *
from sodetlib import noise

