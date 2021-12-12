import numpy as np
import matplotlib
from pprint import pprint

matplotlib.use('agg')
from sodetlib.det_config import DetConfig
import time
import sodetlib.smurf_funcs.det_ops as do
import sodetlib.analysis.det_analysis as da

cfg = DetConfig()
cfg.load_config_files(slot=2)
S = cfg.get_smurf_control(dump_configs=True, make_logfile=True,apply_dev_configs=True, load_device_tune=True)

bands = [0,1,2,3]
bgs = [0,1,2,3,4,5]

iv_info_fp = do.take_iv(S=S,cfg=cfg,bias_groups=bgs,bias_high=19.9,bias_low=0.0,
                     bias_step=0.1,cool_wait=20., high_current_mode=False,
                     do_analysis=False,cool_voltage=10.0,overbias_voltage=19.9)

iv_info = np.load(iv_info_fp,allow_pickle=True).item()
timestamp,phase,mask,tes_biases = da.load_from_dat(S,iv_info['datafile'])
iv_analyze_fp = da.analyze_iv_and_save(S,cfg,iv_info_fp,phase,tes_biases,mask)
iv_analyze = np.load(iv_analyze_fp,allow_pickle=True).item()

da.iv_summary_plots(iv_info,iv_analyze,show_plot=True)
