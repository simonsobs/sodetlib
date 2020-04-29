import matplotlib
matplotlib.use('Agg')
from sodetlib.det_config import DetConfig


if __name__ == "__main__":
    cfg = DetConfig()
    cfg.parse_args()
    S = cfg.get_smurf_control(setup=True, dump_configs=True)
    S.set_stream_enable(0)
