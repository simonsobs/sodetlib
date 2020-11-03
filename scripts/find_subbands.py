import argparse
from sodetlib.det_config import DetConfig
from sodetlib.smurf_funcs import find_subbands
import os


if __name__ == '__main__':
    cfg = DetConfig()
    parser = argparse.ArgumentParser()

    args = cfg.parse_args(parser)
    S = cfg.get_smurf_control(dump_configs=True)

    bands, subband_dict = find_subbands(S, cfg)
    cfg.dev.dump(os.path.abspath(os.path.expandvars(cfg.dev_file)),
                 clobber=True)
