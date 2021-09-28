import matplotlib
matplotlib.use('agg')
import numpy as np

from sodetlib.det_config import DetConfig
import sodetlib.smurf_funcs.det_ops as do
import sodetlib.analysis.det_analysis as da

cfg = DetConfig()
cfg.load_config_files(slot=2)
S = cfg.get_smurf_control(dump_configs=True, apply_dev_configs=True)

S.R_sh = 738e-6
bg = [0]

iv_info_fp = do.take_iv(S=S,cfg=cfg,bias_groups=bg,bias_high=5.0,bias_low=0.0,
                        bias_step=0.005,cool_wait=30., high_current_mode=True,
                        do_analysis=False,cool_voltage=5.0,overbias_voltage=10.0)

iv_info = np.load(iv_info_fp,allow_pickle=True).item()
timestamp,phase,mask,tes_biases = da.load_from_dat(S,iv_info['datafile'])
iv_analyze_fp = da.analyze_iv_and_save(S,cfg,iv_info_fp,phase,tes_biases,mask)
iv_analyze = np.load(iv_analyze_fp,allow_pickle=True).item()
