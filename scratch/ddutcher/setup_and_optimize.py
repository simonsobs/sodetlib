# setup_and_optimize.py

"""
Power amplifiers, optimize tone power and uc_att for each band,
setup with those optimized parameters, and take noise.
"""

import os, sys
import argparse
from sodetlib.det_config import DetConfig
import sodetlib.smurf_funcs.optimize_params as op
import logging

sys.path.append('/sodetlib/scratch/ddutcher')
from noise_stack_by_band import noise_stack_by_band
from uxm_setup import uxm_setup
from uxm_optimize_quick import uxm_optimize

logger = logging.getLogger()

cfg = DetConfig()

parser = argparse.ArgumentParser(
    description="Parser for setup_and_optimize.py script."
)

parser.add_argument(
    'assem_type',
    type=str,
    choices=['ufm','umm'],
    default='ufm',
    help='Assembly type, ufm or umm. Determines the relevant noise  thresholds.',
    )

parser.add_argument(
    "--bands",
    type=int,
    default=None,
    nargs="+",
    help="The SMuRF bands to target. Will default to the bands "
    + "listed in the pysmurf configuration file."
)

# optional arguments
parser.add_argument(
    "--stream-time",
    type=float,
    default=20.0,
    help="float, optional, default is 20.0. The amount of time to sleep in seconds while "
    + "streaming SMuRF data for analysis.",
)
parser.add_argument(
    "--fmin",
    type=float,
    default=5.0,
    help="float, optional, default is 5.0. The lower frequency (Hz) bound used "
    + "when creating a mask of white noise levels Suggested value of 5.0",
)
parser.add_argument(
    "--fmax",
    type=float,
    default=50.0,
    help="float, optional, default is 50.0. The upper frequency (Hz) bound used "
    + "when creating a mask of white noise levels Suggested value of 50.0",
)
parser.add_argument(
    "--detrend",
    default="constant",
    help="str, optional, default is 'constant'. Passed to scipy.signal.welch.",
)
parser.add_argument(
    "--loglevel",
    type=str.upper,
    default=None,
    choices=['DEBUG','INFO','WARNING','ERROR','CRITICAL'],
    help="Set the log level for printed messages. The default is pulled from "
    +"$LOGLEVEL, defaulting to INFO if not set.",
)

# parse the args for this script
args = cfg.parse_args(parser)
if args.loglevel is None:
    args.loglevel = os.environ.get("LOGLEVEL", "INFO")
numeric_level = getattr(logging, args.loglevel)
logging.basicConfig(
    format="%(levelname)s: %(funcName)s: %(message)s", level=numeric_level
)

S = cfg.get_smurf_control(dump_configs=True, make_logfile=(numeric_level != 10))

if args.assem_type == 'ufm':
    high_noise_thresh = 250
    med_noise_thresh = 150
    low_noise_thresh = 120
elif args.assem_type == 'umm':
    high_noise_thresh = 250
    med_noise_thresh = 65
    low_noise_thresh = 45
else:
    raise ValueError("Assembly must be either 'ufm' or 'umm'.")

# power amplifiers
op.cryo_amp_check(S, cfg)

# run the defs in this file
uxm_optimize(
    S=S,
    cfg=cfg,
    bands=args.bands,
    stream_time=args.stream_time,
    fmin=args.fmin,
    fmax=args.fmax,
    low_noise_thresh=low_noise_thresh,
    med_noise_thresh=med_noise_thresh,
    high_noise_thresh=high_noise_thresh,
    detrend=args.detrend,
)

uxm_setup(S=S, cfg=cfg, bands=args.bands)

# plot noise histograms
noise_stack_by_band(
    S,
    stream_time=args.stream_time,
    fmin=args.fmin,
    fmax=args.fmax,
    detrend=args.detrend,
)
