import numpy as np
import time
import sodetlib.smurf_funcs.optimize_params as op
from sodetlib.det_config import DetConfig

cfg = DetConfig()
cfg.load_config_files(slot=2)

S = cfg.get_smurf_control(dump_configs=True, make_logfile=False)
S.load_tune()

for band in [0,1,2,3]:
    S.relock(band)

    for _ in range(3):
        S.run_serial_gradient_descent(band)
        S.run_serial_eta_scan(band)

start = time.time()
print(start)
summary, fname = op.optimize_attens(S, cfg, [0,1,2,3])
stop = time.time()
print(f"Total time: {stop - start}")

print(f"Data saved in {fname}")
