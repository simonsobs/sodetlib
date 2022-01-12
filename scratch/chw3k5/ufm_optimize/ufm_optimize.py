# selects a non-interactive plotting backend
import matplotlib
matplotlib.use('Agg')
# required imports
import os
import sys
# favorite packages.
import pysmurf.client
import argparse
import numpy as np
import time
import glob

import numpy as np
from scipy.interpolate import interp1d
import argparse
import time
import csv
import scipy.signal as signal
import matplotlib.pyplot as plt

# Here we append this python file's directory to the paths the python uses to look for imports.
# This is a temporary measure used as demonstration and testing tool.
basedir_this_file = os.path.basename(__file__)
sys.path.append(basedir_this_file)
from operators.controler import LoadS
from operators.smurf_band import SingleBand
from operators.time_stream import TimeStreamData
from operators.bias_group import GroupedBiases, AutoTune

# example parameters
band = 4
slot_num = 2
bias_group = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
stream_time = 20
nperseg = 2**12
verbose = True

# load a single S, or SMuRF controller instance for a given slot number
load_s = LoadS(slot_nums=[slot_num], verbose=verbose)
cfg = load_s.cfg_dict[slot_num]
S = load_s.S_dict[slot_num]

# configure a single band
single_band = SingleBand(S=S, cfg=cfg, band=band, auto_startup=True, verbose=verbose)
single_band.check_lock()

# configure a collection of bands as a single bias group.
grouped_biases = GroupedBiases(S=S, cfg=cfg, bias_group=bias_group, verbose=verbose)
grouped_biases.overbias_tes(sleep_time=120, tes_bias=1)

# acquire time stream data
auto_tune = AutoTune(S=S, cfg=cfg, nperseg=nperseg, bias_group=bias_group, verbose=verbose)
auto_tune.tune_selector_up_atten(uc_attens_centers_per_band=None, loop_count_max=5,
                                 stream_time=stream_time, do_plots=False, fmin=5, fmax=50)
# print the plotting directory
print(f"plotting directory is:\n{S.plot_dir}")
