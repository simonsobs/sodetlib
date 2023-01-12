from IPython import embed
import traitlets.config
import argparse
import os
import yaml
from sodetlib.det_config import DetConfig
import sodetlib as sdl
import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception:
    print("Trouble importing Matplotlib! Using agg backend")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

if __name__ == "__main__":
    cfg = DetConfig()

    parser = argparse.ArgumentParser()
    args = cfg.parse_args()
    S = cfg.get_smurf_control(dump_configs=True)
    reg = sdl.Registers(S)

    _ipython_config = traitlets.config.get_config()
    _ipython_config.InteractiveShellEmbed.colors = "Linux"
    embed(config=_ipython_config)


