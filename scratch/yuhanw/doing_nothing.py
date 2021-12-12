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


from scipy.interpolate import interp1d

cfg = DetConfig()
cfg.load_config_files(slot=5)
S = cfg.get_smurf_control()


print(S.high_low_current_ratio)
print(S.bias_line_resistance)