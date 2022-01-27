from ._version import get_versions
__version__ = get_versions()['version']
del get_versions

from sodetlib.util import *
from sodetlib.stream import *
