import matplotlib
matplotlib.use('Agg')

import pysmurf.client
import argparse
import numpy as np
import os
import time
import glob

from sodetlib.det_config  import DetConfig
import numpy as np

from sodetlib.smurf_funcs import tracking_quality
import sodetlib.smurf_funcs.optimize_params as op
import sodetlib.util as su
from sodetlib.smurf_funcs.det_ops import take_iv

from scipy.interpolate import interp1d

out_fn = '/data/smurf_data/20210709/test_ready_2'
out = open(out_fn,'a')
print('hi')