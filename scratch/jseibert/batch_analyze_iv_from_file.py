import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import pysmurf.client
import argparse

band = 2
fname = open('/data/sodetlib_data/20200623_IV_NIST_SPB_3_D_UHF_r3_105_to_200_mK_5mk_step.txt','r')
Lines = fname.readlines()

S = pysmurf.client.SmurfControl(epics_root = 'smurf_server_s2',cfg_file = '/data/pysmurf_cfg/experiment_ucsd_sat1_smurfsrv16_lbOnlyBay0.cfg',make_logfile=False)

for i,line in enumerate(Lines):
    fn = line.split('iv_file : ')[-1].split('\n')[0]
    S.analyze_slow_iv_from_file(fn_iv_raw_data = fn,R_sh = 750e-6,bias_line_resistance = 10076)    

