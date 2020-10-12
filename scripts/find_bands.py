
import matplotlib
matplotlib.use('Agg')
import pysmurf.client
import argparse
import numpy as np
from sodetlib.det_config import DetConfig
from sodetlib.smurf_funcs import find_bands
import os



if __name__ == '__main__':
    cfg = DetConfig()
    parser = argparse.ArgumentParser()

    # Custom arguments for this script

    args = cfg.parse_args(parser)
    S = cfg.get_smurf_control(dump_configs=True)

    bands, subband_dict = find_bands(S, cfg)
    cfg.dev.dump(os.path.abspath(os.path.expandvars(cfg.dev_file)),
                 clobber=True)
