import matplotlib
import argparse
import os
matplotlib.use('Agg')

from sodetlib.det_config import DetConfig
from sodetlib.smurf_funcs import health_check

if __name__ == '__main__':
    cfg = DetConfig()
    parser = argparse.ArgumentParser()

    args = cfg.parse_args(parser)
    S = cfg.get_smurf_control(dump_configs=True)

    # Put your script calls here
    health_check(S, cfg)

    out_file = os.path.abspath(os.path.expandvars(cfg.dev_file))
    print(f"Writing new device config to {out_file}")
    cfg.dev.dump(out_file, clobber=True)
