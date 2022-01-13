import matplotlib
matplotlib.use('agg')
import numpy as np

from sodetlib.det_config import DetConfig
import sodetlib.smurf_funcs.det_ops as do
import sodetlib.analysis.det_analysis as da

cfg = DetConfig()
cfg.load_config_files(slot=2)
S = cfg.get_smurf_control(dump_configs=True, apply_dev_configs=True)

bg = [0]


iv_info_fp = do.take_iv(S=S,cfg=cfg,bias_groups=bg,bias_high=4.0,bias_low=0.0,
                        bias_step=0.005,wait_time=0.2, cool_wait=30., high_current_mode=True,
                        do_analysis=True, make_summary_plots=True, 
                        make_channel_plots=False,
                        cool_voltage=5.0,overbias_voltage=10.0)    


