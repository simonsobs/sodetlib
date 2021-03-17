from sodetlib.det_config import DetConfig
from sodetlib.smurf_funcs.det_ops import take_tickle

cfg = DetConfig()
cfg.load_config_files(slot=3)
S = cfg.get_smurf_control()

outfile = take_tickle(S, cfg, True)
print(outfile)
