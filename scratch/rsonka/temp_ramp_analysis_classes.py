#!/usr/bin/env python3
'''
python 3

author: rsonka
desc: collection of classes and methods related to mux map loading,
bath ramps, and cold load ramp analysis
'''

# NOTE! remember you have to use importlib's reload() to get a new version of 
# these classes in JupyterLab/ipython.

import os, sys
from glob import glob
import numpy as np
import scipy.optimize
from scipy.optimize import curve_fit
import scipy.signal
import scipy.interpolate
import matplotlib as mpl
import matplotlib.pyplot as plt
import re
import SPB_opt.SPB_opt_power as spb_opt # Also, think about name. 
from datetime import datetime
import copy
import warnings

mpl.use('nbAgg') # can remove this line when Daniel updates the system sodetlib install so that import optimize_params no longer switches the plotting backend.
default_figsize =  (2.5,2.5) # firefox (2.5,2.5) # chrome (5,5)


# import os, sys
# from glob import glob
# import numpy as np
# import scipy.optimize
# from scipy.optimize import curve_fit
# import scipy.signal
# import scipy.interpolate
# import matplotlib as mpl
# import matplotlib.pyplot as plt
# import argparse
# import re
# import pythonUtilities.numpyPlotting as myp # not really using..
# import SPB_opt.SPB_opt_power as spb_opt # Also, think about name. 
# from datetime import datetime
# import copy
# import inspect # mostly used for reminding myself of function signatures
# from importlib import reload # for use with the below
# import analysis_classes as ac
# import so3g
# from so3g import hk

# sys.path.append('/home/ddutcher/repos/pysmurf/python')

# from sodetlib.smurf_funcs import optimize_params as op
# mpl.use('nbAgg') # can remove this line when Daniel updates the system sodetlib install so that import optimize_params no longer switches the plotting backend.
# default_figsize =  (2.5,2.5) # firefox (2.5,2.5) # chrome (5,5)




# NOTE! remember you have to use importlib's reload() to get a new version of 
# these classes in JupyterLab/ipython.


#2345678901234567890123456789012345678901234567890123456789012345678901234567890
#========80 characters =========================================================

def is_float(num_string):
    try:
        float(num_string)
        return True
    except: # ValueError
        return False
    
# -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
# ========================= Test_Device Class ==================================
# -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~

#### Should be able to write to/load from a config file of file names
#### Saving plots also a good idea. 

# This has to be global for the function that updates mux_map
map_atts = """smurf_band, smurf_chan, smurf_freq, vna_freq, design_freq, index, 
         mux_band, pad, mux_posn, biasline, pol, TES_freq, det_row, det_col, 
         rhomb, opt, det_x, det_y""".replace("\n","").replace(" ","").split(",")

class Test_Device:
    '''A class for representing a UMM, UFM, SPB, mux box, 512 box, or other such
    device. Really anything we're testing through SMuRF. 
    It stores the datafiles associated with that device thus far and the data 
    structures calculated from them for analysis purposes. 
    Eventually it should be able to read/write (at least) the datafiles to 
    human-readable config file. 
    
    # =============================== VARIABLES ================================
    # --------- REQUIRED Vars: ---------
    dName    # Used in plots and filenames.
    outPath  # to the directory that contains the files it writes out. 
    # Config file = outPath/dName_config.(?) <-TODO
    # Config file should note the path to its full saved data if such exists. 
    # Possibly add "number of mux" expected?
    
    # --------- Mapping ---------
    masked_biaslines # array. Used to update mux maps of Kaiwen's format to have 
                     # opt account for masking. Class var so can be referenced 
                     # on detectors  that weren't mapped, had bl found later.
    mux_map_file     # FILEPATH In Kaiwen's format
    base_mux_map     # the loaded array
    map_atts         # Listing Kaiwen's headers (set in above global var):
        # smurf_band, smurf_chan, smurf_freq, vna_freq, design_freq, index, 
        # mux_band, pad, mux_posn, biasline, pol, TES_freq, det_row, det_col,
        # rhomb, opt, det_x, det_y
    opt_dict         # mux_map opt numbers->human readable,& lists linestyles
    mux_map          # CRITICAL: Used for lots of things. format:
        # {smurf_band : {channel: 
        #  {map_atts[i] : mux_map[sb&channel][i] for i in range(2,len(map_atts))}
        # }}
    tes_dict         # tes-exclusive (hopefully). {sb:ch:{<tes data>}} 
                     # contains data calculated for tes's and from mux_map. 
                     # guaranteed: {sb:{ch:{'TES_freq','pol','opt','opt_name','linestyle'}}}
                     # other data may join it from tests. ex:
        # If you do normal correction, Temp_Ramp adds:
           # R_n and R_n_NBL (Normal Branch Length, v_bias [V]).
        # Bath_Ramp adds the following keys:
           # k, k_err, tc, tc_err, n, n_err, cov_G_tc_n (default) OR cov_K_tc_n, 
           # G, G_err, p_sat100mK, p_sat100mK_err
    
    # --------- Other OPTIONAL Vars: ---------
    # Added by anything that gets channel data with associated biasline data:
    bls_seen         # {sb:{ch:[list of biaslines of associated channel datas]}}   
                     # ^ Exists because Kaiwen's mux_map doesn't tell me some bls
                     # and because you can seem to get data when you aren't actually being powered
    # ---- Fine_VNA object
    fine_VNA         # = Fine_VNA(vna_directory)
    # ---- Bath temperature ramp
    bath_ramp        # Bath_Ramp object
    
    # ======================= METHODS (Helpers indented) =======================
    __init__(self, dName, outPath, mux_map_file, opt_dict=None, masked_biaslines=[])
    check_mux_map_for_channel_and_init(mux_map, sb, ch)
    check_tes_dict_for_channel_and_init(tes_dict, sb, ch)
    '''
    
    # Obviously not complete. 
    def __init__(self, dName, outPath, mux_map_file, opt_dict=None, masked_biaslines=[]):
        # mux_map_file must be a csv in Kaiwen's standard format. I made it in excel for SPB-B14; 
        # frankly, I think that's the right way to go for SPB, where P_sat knowledge
        # is important and you often have to swap things around/have crazier detectors.
        s=self
        self.dName   = dName
        self.outPath = outPath
        s.masked_biaslines=masked_biaslines
        s.bls_seen={}
        s.base_mux_map_file = mux_map_file
        s.base_mux_map = np.genfromtxt(mux_map_file,delimiter=",",skip_header=1, dtype=None)
        s.map_atts = map_atts # has to be global for the function that updates mux_map        
        # This was a poor decision to put it all in opt. 
        # Should have separate masked/unmasked and dark-fab/opt-fab values.
        # unfortunately changing at this point would require a ton of work. 
        if not opt_dict:
            # Defining the opt map here!
            # I think Kaiwen just uses -1, 0 and 1. Does not account for masks here.
            # opt_dict = {opt_num: (name,linestyle)}
            s.opt_dict = {-1.0: ('no TES', 'None'),
                          -0.75: ('Fabrication error TES (effectively disconnected)', 'None'),
                          0.0: ('dark masked', 'dotted'),
                          0.25: ('opt masked', 'dashdot'),
                          0.75: ('dark horn-pixel', 'dashed'),
                          1.0: ('opt horn', 'solid')} 
        else:
            s.opt_dict = opt_dict
        
        # At least in Kaiwen's Mv6, opt=0.0 if dark fab OR opt fab masked. opt=1.0 if opt and unmasked.
        # 
        if masked_biaslines:
            for i in range(len(s.base_mux_map)):  
                if int(s.base_mux_map[i][9]) in masked_biaslines: # det[9] = the bias line. Is masked.
                    #print("recognized a masked bias line")
                    if float(s.base_mux_map[i][15]) == 1.0:        # det[15] = opt. masked optical
                        s.base_mux_map[i][15] = 0.25               # masked dark (0.0) stays the same. 
                    #print(s.mux_map[i][10])
                    if (str(s.base_mux_map[i][10])[2:-1] == 'A' or str(s.base_mux_map[i][10])[2:-1] == 'B') and float(s.base_mux_map[i][15])==0.0: # pol=D=Kaiwen's dark fab note 
                        s.base_mux_map[i][15] = 0.25  # That's actually an optical masked detector. 
                else:                                         # unmasked
                    if float(s.base_mux_map[i][15]) == 0.0 and str(s.base_mux_map[i][10])[2:-1] == 'D':
                        s.base_mux_map[i][15] = 0.75               # dark fab on area getting light.                      

        mux_map = {}
        tes_dict = {}
        
        for chan in s.base_mux_map:
            sb, ch = chan[0], chan[1]
            if not sb in mux_map.keys():
                mux_map[sb] = {}
            mux_map[sb][ch] = {}
            for i in range(2,len(map_atts)):
                mux_map[sb][ch][map_atts[i]] = chan[i]
            mux_map[sb][ch]['TES_freq'] = str(mux_map[sb][ch]['TES_freq'])[2:-1]
            if mux_map[sb][ch]['opt'] >= 0: # TES expected
                if not sb in tes_dict.keys():
                    tes_dict[sb] = {}
                tes_dict[sb][ch] = {}
                tes_dict[sb][ch]['TES_freq'] = mux_map[sb][ch]['TES_freq']
                for tes_key in ['pol','opt']:
                    tes_dict[sb][ch][tes_key] = mux_map[sb][ch][tes_key]
                if mux_map[sb][ch]['opt'] in s.opt_dict.keys():
                    tes_dict[sb][ch]['opt_name'] = s.opt_dict[mux_map[sb][ch]['opt']][0]
                    tes_dict[sb][ch]['linestyle'] = s.opt_dict[mux_map[sb][ch]['opt']][1]
                else:
                    #print(f"{ch} unknown opt: {mux_map[sb][ch]['opt']}")
                    tes_dict[sb][ch]['opt_name'] = f"? opt: {mux_map[sb][ch]['opt']}"
                    tes_dict[sb][ch]['linestyle'] = (0, (3, 5, 1, 5, 1, 5)) # dashdotdotted      
        
        s.mux_map = mux_map
        s.tes_dict = tes_dict

        
# possibly move to a "generic function" cell?        
# Two floating functions for if mux_map and tes_dict are missing real data, later:
def check_mux_map_for_channel_and_init(mux_map, sb, ch):
    if  sb not in mux_map.keys():
        mux_map[sb] = {}
    if ch not in mux_map[sb].keys():
        mux_map[sb][ch] = {}
        for i in range(2,len(map_atts)):
            mux_map[sb][ch][map_atts[i]] = -42 # Unknown

def check_tes_dict_for_channel_and_init(tes_dict, sb, ch):
    if sb not in tes_dict.keys():
        tes_dict[sb] = {}
    if ch not in tes_dict[sb].keys():
        tes_dict[sb][ch] = {}
        tes_dict[sb][ch]['TES_freq'] = str(-42) # Unknown
        tes_dict[sb][ch]['linestyle'] = (0, (3, 5, 1, 5, 1, 5)) # dashdotdotted      
        for tes_key in ['pol','opt','opt_name']:
            tes_dict[sb][ch][tes_key] = "?"
            
            
            
# -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
# ========================== Temp_Ramp Class ===================================
# -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~

class Temp_Ramp:
    '''
    This class serves as a data container for EITHER a bath temperature sweep
    OR a cold load temperature ramp. Those are child classes of this one.
    
    # We do add channels to mux_map if they're missing there; not to tes_dict.
    -- adds 'R_n' to tes_dict if not already there (updates if gets better one),
    # using the R_n reported in IV curves that weren't removed by cuts that 
    # has the longest normal branch. (because while pysmurf has less normal 
    # region to work with, it (usually---can go other way, we account for this) 
    # overestimates R_n (1/slope of fit orange dashed line) due to including 
    # bits of the IV curve). Our non-raw p_b90's are also p_b90's of THAT R_n. 
    
    # =============================== VARIABLES ================================
    r_sh                 # 0.0004 # Pysmurf's r_sh, in ohms.
    # ---- Init argument vals
    test_device          # Test_Device object of this array/SPB
    ramp_type = 'bath' or 'coldload'
    temp_unit = 'mK' for bath or 'K' for coldoad
    dName                # device name. TODO: add to Temp_Ramp plots.
    therm_cal            # [a, b] of ax + b thermometer calibration
    metadata_fp_arr      # [FILEPATH] <-array in case of multiple bath ramps. 
    p_sat_cut            # what p_sat= unbelievably high. 15 for MF, 30 for UHF?
    use_p_sat_SM         # Whether to consider pysmurf's p_sat in cuts.
    input_file_type      # defaults "pysmurf", set "original_sodetlib" for the 
                         # original take_iv.
    
    # ---- reference/analysis from loaded data
    key_info             # {key in (other) dictionary in this class : {
                         #    'name','units','lim'}} # 'lim' MAY NOT EXIST.
    idx_fix_fails        # idx_fix debugging dict {sb:{ch:{temp:{
                         #     sc_idx_old,nb_idx_old,nb_idx_new}}}
    temp_list_raw        # List of all temps loaded in, in increasing order
    temp_list            # Increasing List of all temps with >= 1 uncut data point 
    det_R_n              # {sb:{ch:R_n}} made if you do normal correction.
    det_R_n_NBL          # {sb:{ch:<normal branch length in v_bias [V] 
                         #          of IV curve det_R_n taken from>}}
    iv_cat               # categorizor dict of lists of iv names = (sb,ch,temp,bl):
                         # {'is_iv':[],'not_iv':[],
                         #  'wrong_bl':[],'bad_channel':[],'bad_temp':[]}
    
    # ---- loaded ramp vals.
    # NOTE: "p_satSM" = pysmurf p_sat = median p_tes in [sc_idx:nb_idx]
    # Below arrs are [for each metadata, <stuff in comments>]
    iv_analyzed_info_arr # [for each temp: {'temp', 'bl', 'iv_analyzed_fp', 
                         #                  'iv_analyzed','human_time'}]
                         # 'iv_analyzed':{sb:{ch:{'trans idxs', 'v_bias', 
                         #     'i_bias', 'i_tes', 'c_norm', 'R', 'R_n', 
                         #     'v_bias_target', 'v_tes', 'v_tes_target', 
                         #     'p_tes', 'p_trans', 'p_b90', 'is_iv', 'is_iv_info'}}
    ramp_raw_arr         # {sb:{ch:{'temp_raw', 'p_satSM', 'R_nSM_raw', 'temp90R_n', 'p_b90R_n'}}}
                         # p_satSM = pysmurf 'p_sat'  | p_b90R_n = SO p_sat
    ramp_arr             # {sb:{ch:{'temp', 'p_b90','R_nSM'}}} 
                         # ^ ramp_arr has bad "IVs" cut out.
    # Only one array, has all from each metadata combined:                     
    ramp                 # the combination of the ramp_arr dictionaries into one.
    
    # ======================= METHODS (Helpers indented) =======================
    # ---------- Initialization and loading methods
    __init__(s, test_device, ramp_type, therm_cal, metadata_fp_arr, 
             norm_correct=True,p_sat_cut=15,use_p_satSM=True,
             fix_pysmurf_idx=False,input_file_type="pysmurf")
        # __init__ structure: 
        # 1. basic init
        # 2. appends s.load_iv_analyzed(metadata)'s to iv_analyzed_info_arr
        # 3. s.load_temp_sweep_IVs()
        # 4. if norm_correct, make non-corrected data structures and s.do_normal_correction
        # 5. merge ramps for s.ramp. Categorize ivs.
        setup_key_info(s)
        load_iv_analyzed(s,metadata) # loads file data & makes IV cuts. 
            original_sodetlib_to_pysmurf(s, sod_iv_analyzed)
            remove_sc_offset(s,iva)
            fix_pysmurf_trans_idx(s, py_ch_iv_analyzed,sb,ch,temp)
        standalone_IV_cut(s,d) # called by load_iv_analyzed for bls, and below for cuts:
        load_temp_sweep_IVs(s) # decides bls, performs cuts, makes ramp_raw_arr and ramp_arr
            cut_iv_analyzed(s,bathsweep_raw,bathsweep,now_bad_IVs,iai)
            calc_P_b90(s,r_n,r_tes,p_tes)
            add_ramp_raw_iv(s, bathsweep_raw, temp, sb, ch, d)
            add_ramp_iv(s,bathsweep, temp, sb, ch, d, p_b90=-42)
        do_normal_correction(s)
            calculate_det_R_ns(s)
            pysmurf_iv_data_correction(s, py_ch_iv_analyzed,\
                                 nb_start_idx=0,nb_end_idx=-1,force_R_n=-42.0,
                                 force_c=-42.0)
        merge_ramps(s,bs1,bs2)
            dUnique(s,d1,d2)
        categorize_ivs(s)
            categorize_ivs_helper(s,iv_line)
    redo_cuts(s) # maybe run this when norm_correct makes BIG difference
    
    # ---------- Plotting Methods
    find_temp_ch_iv_analyzed(s,sb,ch,temp,bl=-42,run_indexes=[])
    # --- individual IV curve plotters ---
    plot_IV(s,sb,ch,temp,x_lim=[],y_lim=[], tes_dict=True, bl=-42,run_indexes=[], 
                                  own_fig=True,linewidth=1)
    plot_iv_analyzed_keys(s,x_key,y_key, sb, ch, temp, 
                                  x_lim=[],y_lim=[], tes_dict=True, bl=-42,
                                  run_indexes=[], own_fig=True,linewidth=1)
        internal_plot_iv_analyzed_keys(s,d,sb,ch,temp, x_key,y_key,
                                  x_lim=[],y_lim=[], tes_dict=True, bl=-42,
                                  run_indexes=[], own_fig=True,linewidth=1)
    plot_RP_easy(s,sb, ch, temp, tes_dict=True, bl=-42, run_indexes=[])
    plot_RP(s,analyzed_iv_info, temp, bl, sb, ch)
    # --- by sb plotters ---
    plot_ramp_raw(s,ramp_raw,tes_dict=True)
    plot_ramp_raw90R_n(s,ramp_raw,tes_dict=True)
    plot_ramp(s,ramp,tes_dict=True,zero_starts=False)
    plot_ramp_keys_by_sb(s,bathsweep,x_key,y_key,x_lim=[],y_lim=[], 
                                 prefix='',tes_dict=True,zero_starts=False)
    plot_ramp_keys_by_sb_2legend(s,bathsweep,x_key,y_key,x_lim=[],y_lim=[], 
                                 prefix='',tes_dict=True,zero_starts=False)
    # --- by BL plotters ---
    plot_ramp_by_BL(s,bathsweep,tes_dict=True,y_lim=[0,8])
    plot_ramp_keys_by_BL(s,bathsweep, x_key,y_key,
                                 x_lim=[],y_lim=[], prefix='',tes_dict=True)
    '''
    
    def __init__(s, test_device, ramp_type, therm_cal,
                 metadata_fp_arr, norm_correct=True,p_sat_cut=15,use_p_satSM=True,
                 fix_pysmurf_idx=False,input_file_type="pysmurf"):
        """Set input_file_type = "original_sodetlib" for the original take_iv."""
        s.use_p_satSM=use_p_satSM
        s.p_sat_cut=p_sat_cut
        s.test_device = test_device 
        # these three come up so often, they get their own variables. 
        s.dName = test_device.dName
        s.mux_map = test_device.mux_map
        s.tes_dict = test_device.tes_dict
        s.r_sh = 0.0004 # Pysmurf's r_sh, in ohms. 
        s.fix_pysmurf_idx=fix_pysmurf_idx
        s.idx_fix_fails={}
        s.input_file_type = input_file_type
        
        assert ramp_type == 'bath' or ramp_type == 'coldload', \
           "ramp_type must be 'bath' or 'coldload'"
        s.ramp_type = ramp_type # used in key info
        s.therm_cal  = therm_cal
        s.metadata_fp_arr = metadata_fp_arr
        # initialize data structures
        s.temp_list = []
        s.temp_list_raw = []
        s.iv_analyzed_info_arr = []
        s.ramp_raw_arr = []
        s.ramp_arr = []
        # load data
        for metadata_fp in s.metadata_fp_arr: 
            metadata = np.genfromtxt(metadata_fp,delimiter=',', dtype='str') #skip_header=0
            s.iv_analyzed_info_arr.append(s.load_iv_analyzed(metadata)) 
        s.load_temp_sweep_IVs() # checks bls, sets up ramp_raw_arr and ramp_arr
        if norm_correct:
            s.iv_analyzed_info_arr_nnc = copy.deepcopy(s.iv_analyzed_info_arr)
            s.ramp_raw_arr_nnc = copy.deepcopy(s.ramp_raw_arr)
            s.ramp_arr_nnc = copy.deepcopy(s.ramp_arr)
            s.det_R_n = {}
            s.det_R_n_NBL = {}
            s.do_normal_correction() # updates the non-nnc arrays
            # did this part below just to have an easy plot of differences 
            s.ramp_nnc = s.ramp_arr_nnc[0]
            for i in range(1,len(s.ramp_arr_nnc)):
                s.ramp_nnc = s.merge_ramps(s.ramp_nnc, s.ramp_arr_nnc[i])
        s.ramp = s.ramp_arr[0] # merging
        for i in range(1,len(s.ramp_arr)):
            s.ramp = s.merge_ramps(s.ramp, s.ramp_arr[i])
        s.setup_key_info() # Has to come last b/c looks at what's loaded. 
        s.categorize_ivs() # can remove if I ever don't need anymore
    
    
    
    # ============== non-data init functions =================
    def setup_key_info(s):
        # name and units 
        s.key_info = { \
            'p_satSM'  : {'name' : 'Pysmurf P_bias in transition', 
                          'units' : 'pW', 'lim': [0,12.0/15.0*s.p_sat_cut]},
            'p_b90R_n' : {'name' : 'P_bias at R=90% R_n', 
                          'units' : 'pW', 'lim': [0,12.0/15*s.p_sat_cut]},
            'p_b90'    : {'name' : 'P_bias at R=90% R_n', 
                          'units' : 'pW', 'lim': [0,12.0/15*s.p_sat_cut]},
            'R_nSM_raw': {'name' : 'Pysmurf R_n fit', 
                          'units' : 'Ω', 'lim': [0.007,0.008]},
            'R_nSM'    : {'name' : 'Pysmurf R_n fit', 
                          'units' : 'Ω', 'lim': [0.007,0.008]}, # Now, iv_analyzed ones
            'v_bias'   : {'name' : 'Bias voltage through bias line full DR', 
                          'units' : 'V', 'lim': [0,20]},
            'v_tes'    : {'name' : 'TES voltage aka R_sh voltage', 
                          'units' : 'μV', 'lim': [0,0.5]},
            'R'        : {'name' : 'TES resistance', 
                          'units' : 'Ω', 'lim': [0,0.008]},
            'p_tes'    : {'name' : 'Electrical power on TES', 
                          'units' : 'pW', 'lim': [0,12]},          
            'i_bias'   : {'name' : 'Bias current on bias line', # only in corrected
                          'units' : 'μA', 'lim': [0,1200]},   
            'i_tes'    : {'name' : 'Current through TES', 
                          'units' : 'μA'}} # no limits, varies much with temp.   
        if s.ramp_type == 'bath':
            temp_dict = {'name': 'Bath temperature', 'units': 'mK'}     
        elif s.ramp_type == 'coldload':
            temp_dict = {'name': 'Coldload temperature', 'units': 'K'}
        s.key_info['temp_raw']  = temp_dict
        s.key_info['temp90R_n'] = temp_dict.copy()
        s.key_info['temp']      = temp_dict.copy()
        s.key_info['temp_raw']['lim'] = [0.95*s.temp_list_raw[0],s.temp_list_raw[-1]+0.05*s.temp_list_raw[0]]
        s.key_info['temp90R_n']['lim'] = [0.95*s.temp_list_raw[0],s.temp_list_raw[-1]+0.05*s.temp_list_raw[0]]
        s.key_info['temp']['lim'] = [0.95*s.temp_list[0],s.temp_list[-1]+0.05*s.temp_list[0]]
        
    # ============== Functions to load data, cut bad IVs, and merge data ==========    
    def load_iv_analyzed(s,metadata):
        # loading iv_analyzed data from file and track bias lines seen.
        mux_map = s.mux_map
        tes_dict = s.tes_dict
        analyzed_iv_info = []
        for line in metadata:
            temp, _, bl, sbs, fp, meas_type = line # bl = bias line, sbs= smurf bands (not subband), fp = file path
            if not meas_type.upper() == 'IV': # cut out the noise measurements.
                continue
            #iv_analyzed_fp = fp.replace("/data/smurf_data", "/data/legacy/smurfsrv")
            iv_analyzed_fp = fp.replace("/data/smurf_data", "/data2/smurf_data")
            iv_analyzed_fp = iv_analyzed_fp.replace("iv_raw_data", "iv")
            iv_analyzed_fp = iv_analyzed_fp.replace("iv_info", "iv_analyze") # orig. sodetlib
            iv_analyzed = np.load(iv_analyzed_fp, allow_pickle=True).item()
            # Now, if this isn't really a pysmurf dictionary, adapt it to that form
            if s.input_file_type == "original_sodetlib":
                iv_analyzed = s.original_sodetlib_to_pysmurf(iv_analyzed)
            # Let's grab times
            #ctime = int(re.search(r'.*/(\d+)_iv\.npy', iv_analyzed_fp).group(1)) # doesn't work on _iv_info.npy, which I assume comr from the original sodetlib take_iv
            ctime = int(re.search(r'.*/(\d+)_iv.*', iv_analyzed_fp).group(1))
            ctime -= 4*60*60 # I guess it's the wrong timezone?
            human_time = datetime.utcfromtimestamp(ctime).strftime('%Y-%m-%d %H:%M:%S')
            #print(human_time)
            # calibrate temperatures
            temp = float(temp)*s.therm_cal[0] + s.therm_cal[1]
            
            analyzed_iv_info.append({'temp':temp, 'bl':int(bl), \
                                     'iv_analyzed_fp':iv_analyzed_fp, \
                                     'iv_analyzed':iv_analyzed,'human_time':human_time})
            #print("bl:"+bl + " sb's:" + str(iv_analyzed.keys()))
            for sb in iv_analyzed:
                if sb == "high_current_mode":
                        continue
                for ch in iv_analyzed[sb]:
                    # REMOVE THIS IF the superconducting offset thing comes out!!!
                    #iv_analyzed[sb][ch] = s.remove_sc_offset(iv_analyzed[sb][ch])
                    if s.fix_pysmurf_idx:
                        iv_analyzed[sb][ch] = s.fix_pysmurf_trans_idx(iv_analyzed[sb][ch], sb,ch,temp)
                    if not s.standalone_IV_cut(iv_analyzed[sb][ch]):
                        continue
                    if sb not in s.test_device.bls_seen.keys():
                        s.test_device.bls_seen[sb] = {}
                    if ch not in s.test_device.bls_seen[sb].keys():
                        s.test_device.bls_seen[sb][ch] = []
                    s.test_device.bls_seen[sb][ch].append(int(bl))
        return  analyzed_iv_info
    
    # ---- three load_iv_analyzed(s,metadata) helper functions:
    def original_sodetlib_to_pysmurf(s, sod_iv_analyzed):
        given = sod_iv_analyzed
        iv_analyzed = {'high_current_mode': \
                       given['metadata']['iv_info']['high_current_mode']}
        for sb in given['data']:
            iv_analyzed[sb] = {}
            for ch, d in given['data'][sb].items():# i_tes not in pysmurf, keepin it 
                iv_analyzed[sb][ch] = {key:d[key] \
                                       for key in ["R","R_n","p_tes","i_tes",\
                                                   "v_bias","v_tes","si"]}
                sc_idx, nb_idx = int(d["idxs"][0]),int(d["idxs"][2])
                #v how I discovered that it will send you stuff that has sc_idx > nb_idx
                #print(f"sb{sb}ch{ch} p_tes len{len(d['p_tes'])} idxs{[sc_idx, nb_idx]}")
                if sc_idx > nb_idx:
#                     print(f"sb{sb}ch{ch} p_tes len{len(d['p_tes'])}" +\
#                           f" idxs{[sc_idx, nb_idx]}; swapping idxs")
                    temporary = sc_idx
                    sc_idx = nb_idx
                    nb_idx = sc_idx
                iv_analyzed[sb][ch]['trans idxs'] = [sc_idx, nb_idx]    
                                
                iv_analyzed[sb][ch]['p_trans'] = \
                    np.median(d['p_tes'][sc_idx:nb_idx])
                target_idxs = np.where(d["R"] < 0.007)[0]
                if len(target_idxs) == 0:
                    target_idx = 0 # something is horribly wrong though
                else:  
                    target_idx = target_idxs[-1]
                for targ in ["si","v_bias","v_tes"]:
                    # Si being derivative based, one shorter than R
                    if targ == "si" and target_idx == len(d['si']): 
                        iv_analyzed[sb][ch][targ+"_target"] = d[targ][target_idx-1]
                    else:
                        iv_analyzed[sb][ch][targ+"_target"] = d[targ][target_idx]
        return iv_analyzed
    
    def remove_sc_offset(s,iva):
        # note: important to only run this once because of the deletion!
        sc_idx, nb_idx = iva['trans idxs']
        if sc_idx == 0:
            return iva
        gap = iva['i_tes'][sc_idx] - iva['i_tes'][sc_idx-1]
#         pre = iva['i_tes'][0] 
        iva['i_tes'][:sc_idx] += gap
#         if iva['i_tes'][0] == pre:
#             print("nogap change")
#             print("hey" + pre)
        # take out the point behind sc_idx if there's more than one, so not two identical 
        # messing up the algorithm
        data_len = len(iva['i_tes'])
        rmv = min(sc_idx-1,1) # let's not pull the only sc branch point if there's only one.
        for key,val in iva.items():
            if type(val) == type(iva['i_tes']) and len(val) == data_len:
                iva[key] = np.concatenate((val[:sc_idx-rmv],val[sc_idx:]))
        if rmv ==1:
            iva['trans idxs'][0] -=1 # gotta reset sc_idx
        return iva
    
    def fix_pysmurf_trans_idx(s, py_ch_iv_analyzed,sb,ch,temp):
        iv = py_ch_iv_analyzed
        i_bias = iv['v_tes']*(1/s.r_sh + 1/iv['R'])
        resp_bin = i_bias/(iv['R']/s.r_sh + 1) # the current through TES
        sc_idx = iv['trans idxs'][0]
        # the +1 is necessary because the sc_idx could be on either side of the artificial jump
        #correct_normal_index = np.where(resp_bin == min(resp_bin[iv['trans idxs'][0]+1:]))[0][0]
        i_tes_diff_sign = np.sign(np.diff(resp_bin[sc_idx:]))
        # Note that if sc_idx is much smaller than it should be,  could screw this
        # but if I weight the downward slopes too much, crashes on the "IV"s that are noisy lines
        correct_normal_index = sc_idx+dict_balance_idx(i_tes_diff_sign,{-1:1},{1:1})
        # Debugging
        #sc_idx = iv['trans idxs'][0]
        nb_idx_old = iv['trans idxs'][1]
        nb_idx_new = correct_normal_index
        debug = False
        if debug and not nb_idx_old == nb_idx_new and (nb_idx_new<sc_idx or nb_idx_new>nb_idx_old):
            print(f"{temp:.0f} {sb} {ch} sc_idx_old{sc_idx}(i_b{i_bias[sc_idx]:.3},i_t{resp_bin[sc_idx]:.3},R_t{iv['R'][sc_idx]:.3})")
            print(f"{temp:.0f} {sb} {ch} nb_idx_old{nb_idx_old}(i_b{i_bias[nb_idx_old]:.3},i_t{resp_bin[nb_idx_old]:.3},R_t{iv['R'][nb_idx_old]:.3})")
            print(f"{temp:.0f} {sb} {ch} nb_idx_new{nb_idx_new}(i_b{i_bias[nb_idx_new]:.3},i_t{resp_bin[nb_idx_new]:.3},R_t{iv['R'][nb_idx_new]:.3})")
        # Uncomment the below once debugged
        #if correct_normal_index > sc_idx and correct_normal_index<=:
        # I'm cheating here.
        if correct_normal_index > sc_idx+2 and correct_normal_index <= nb_idx_old: #len(resp_bin): # CHEATER
            iv['trans idxs'] = [iv['trans idxs'][0], correct_normal_index]
        else:
            if not sb in s.idx_fix_fails.keys():
                s.idx_fix_fails[sb]={}
            if ch not in s.idx_fix_fails[sb].keys():
                s.idx_fix_fails[sb][ch]={}
            s.idx_fix_fails[sb][ch][temp] = \
                {'sc_idx_old':sc_idx,'nb_idx_old':nb_idx_old,'nb_idx_new':nb_idx_new}
            #print(f"cheated idx_fix: {temp:.0f} {sb} {ch} sc_idx_old{sc_idx} nb_idx_old{nb_idx_old} nb_idx_new{nb_idx_new}")
        return iv
    
    # called by load_iv_analyzed for bls, and load_temp_sweep_IVs's helper functions for cuts:
    def standalone_IV_cut(s,d):
        # TODO: GET THE FULL CONTEXTLESS IV EVALUATOR IN HERE!
        # Jack lashner said cutting on R_n was effective...and it does seem to be!
        # Well, for Mv13. Less so for Mv5. Not sure I should rely on this. 
        #if d['R_n'] <
        ind = np.where(d['p_tes']>s.p_sat_cut)[0] # for cutting ones when 
        if len(ind)==0: # there wasn't really bias line power, probably?
            return False
        # The below cuts plots that have wild upswings of power 
        # after the 'transition.' Originally 7e-3 but that hits some real ones.
        if np.min(d['R'][ind]) < 6e-3: # why is this here?
            return False
        if np.std(d['R'][-100:]) > 1e-4: # The file stores p_sat and R in 
            return False # reverse time-order to how it's taken; superconducting first.
        # So the above is checking that the normal resistance is steady. 
        # cuts on p_sat
        p_sat = s.calc_P_b90(d['R_n'],d['R'],d['p_tes'])
        if not p_sat:
            return False
        #p_sat = d['p_tes'][p_sat_idx]
        # Don't change the min p_sat from 0, you'll junk lots of good fits.  
        # TODO: some frequency dependence needs to be added to this upperbound. 
        if p_sat < 0 or p_sat > s.p_sat_cut: # sane P_sats. Daniel had P_sat > 10, I'll take up to 15 pW
            return False
        if s.use_p_satSM:
            if d['p_trans'] < 0 or d['p_trans'] > s.p_sat_cut: # Sane smurf/py P_sats. upped to 15 pW as well
                return False
            # If there's a BIG difference in py_p_sat and 90% R_n P_sat, something is probably wrong. 
            #if s.key_info['temp'][1] = 
            if max(p_sat,d['p_trans'])/min(p_sat,d['p_trans']) > 2 and \
               max(p_sat,d['p_trans'])-min(p_sat,d['p_trans']) > 2:
                return False
        # It's crazy that this actually hits things that passed the previous.
        # but rarely, it does (ex 5,50 of mv5_CL10_bath_ramp):
        lp = len(d['p_tes'])
        sc_idx, nb_idx = d['trans idxs']
        if lp-nb_idx <=3 or nb_idx -sc_idx <=3:
            return False
        return p_sat
    
    # -------- The cut controller and ramp_raw_arr and ramp_arr loader
    def load_temp_sweep_IVs(s): 
        # decides what bl each channel is on if that not given by map, performs cuts,
        # and sets up ramp_raw_arr and ramp_arr
        for sb in s.test_device.bls_seen.keys():
            for ch,bl_list in s.test_device.bls_seen[sb].items():
                bl_counts = np.unique(np.array(bl_list),return_counts=True)
                mode_idx = np.where(bl_counts[1]==max(bl_counts[1]))[0]
                if len(mode_idx)==1: # one clear mode
                    # Let's update the mux_map.
                    # b/c Kaiwen's map didn't contain all of these.  
                    check_mux_map_for_channel_and_init(s.mux_map, sb, ch)
                    s.test_device.mux_map[sb][ch]['biasline']= bl_counts[0][mode_idx[0]]
                # Is it possible to get equal of a wrong and a right?
        for ramp_run in s.iv_analyzed_info_arr:
            bathsweep_raw = {} # originally just made this for bathsweep. 
            bathsweep = {}
            now_bad_IVs = {} # sb: bad channel list. Cuts remaining data from channel if it throws something bad.
            for iai in ramp_run:
                s.cut_iv_analyzed(bathsweep_raw,bathsweep,now_bad_IVs,iai)
            s.ramp_raw_arr.append(bathsweep_raw)
            s.ramp_arr.append(bathsweep)
    
    
    def cut_iv_analyzed(s,bathsweep_raw,bathsweep,now_bad_IVs,iai):
        # names because I originally made it just for bathsweep.
        # iai an iv_analyzed_info 
        mux_map = s.mux_map
        tes_dict = s.tes_dict
        temp, bl, iv_analyzed = iai['temp'],iai['bl'],iai['iv_analyzed']
        for sb in iv_analyzed.keys():
            if sb=="high_current_mode":
                continue
            # prep now_bad_IVs:
            if sb not in now_bad_IVs.keys():
                now_bad_IVs[sb] = []
            for ch, d in iv_analyzed[sb].items(): # d for dictionary.
                # Let's update the mux_map.
                # b/c Kaiwen's map didn't contain all of these.  
                check_mux_map_for_channel_and_init(mux_map, sb, ch)

                # don't do anything when there definitely wasn't actually power in the bias line.
                # actually...just get rid of these if we knew their bias line. 
                # This does potentially keep bad ones from Kaiwen's omissions. 
                # hopefully that won't be a problem. 
                # actually...Kaiwen's bl assignment could be wrong if she typod. 
                # it would be better to do this 
                if not int(mux_map[sb][ch]['biasline']) == int(bl):
                    continue # yes, it does sometimes think there's an IV curve when no power was being run.
                
                # Otherwise, Load data into bathsweep_raw and temp_list_raw no matter what.  
                s.update_temp_list(temp, 'temp_list_raw')
                s.add_ramp_raw_iv(bathsweep_raw, temp, sb, ch, d) 

                # Now exit if the channel's been killed
                if ch in now_bad_IVs[sb]:
                    continue

                # Now, check for a valid IV curve. 
                p_sat = s.standalone_IV_cut(d)
                if not p_sat:
                    continue

                if sb in bathsweep.keys() and ch in bathsweep[sb].keys() and len(bathsweep[sb][ch]['p_b90'])>0:
                    # cut on increasing 90% R_n p_b and kills further data reading from the channel if that occurs
                    # Pysmurf p_sat is a more variable than 90% R_n p_sat when things are working, so 
                    # it doesn't kill on this stuff.
                    prev_p_sat = bathsweep[sb][ch]['p_b90'][-1] 
                    if s.ramp_type == "bath" and p_sat > prev_p_sat: 
                        now_bad_IVs[sb].append(ch)
                        continue
                    elif s.ramp_type =="coldload" and p_sat > 1.1*prev_p_sat:
                        now_bad_IVs[sb].append(ch)
                        continue
                    if s.use_p_satSM:
                        # cut, but not kill, on SIGNIFICANTLY increasing pysmurf p_sat since last good.
                        # prev_py_p_sat set below when a point is accepted. 
                        prev_py_p_sat_ind = np.argwhere(bathsweep_raw[sb][ch]['p_b90R_n']==prev_p_sat)[0][0]
                        prev_py_p_sat = bathsweep_raw[sb][ch]['p_satSM'][prev_py_p_sat_ind] 
                        if d['p_trans'] > 1.5*prev_py_p_sat:
                            continue 
                
                # I think this might be too aggressive with the cuts.
                # more sophisticated version of above using slopes:
                # But the slopes are too gentle in coldload ramps. 
                if s.ramp_type == 'bath' and sb in bathsweep.keys() and ch in bathsweep[sb].keys() and len(bathsweep[sb][ch]['p_b90'])>1:
                    # cut on notably slope-increasing 90% p_sat and DON'T kill further data reading from the channel if that occurs
                    # Pysmurf p_sat is a more variable than 90% R_n p_sat when things are working, so 
                    # it doesn't kill on this stuff.
                    prev_p_sat_slope = (bathsweep[sb][ch]['p_b90'][-1]-bathsweep[sb][ch]['p_b90'][-2])/ \
                                        (bathsweep[sb][ch]['temp'][-1]-bathsweep[sb][ch]['temp'][-2])
                    this_p_sat_slope = (p_sat-bathsweep[sb][ch]['p_b90'][-1])/ \
                                        (temp-bathsweep[sb][ch]['temp'][-1])
                    if this_p_sat_slope > prev_p_sat_slope*0.7: # slopes are negative; keep a little fudge factor
                        #now_bad_IVs[sb].append(ch)
                        # It will likely kill itself with this if necessary, b/c denom just getting migger.
                        continue
                    # doing this with pysmurf is annoying, start with just 90R_n
                    # cut, but not kill, on SIGNIFICANTLY increasing pysmurf p_sat slope since last good.
#                         prev_py_p_sat_ind = np.argwhere(bathsweep_raw[sb][ch]['p_b90R_n']==prev_p_sat)[0][0]
#                         #print(prev_py_p_sat_ind)
#                         prev_py_p_sat = bathsweep_raw[sb][ch]['p_satSM'][prev_py_p_sat_ind] 
#                         if d['p_trans'] > 1.5*prev_py_p_sat:
#                             continue 
#                         prev_py_p_sat = d['p_trans']

                # Passed cuts -- include it!       
                s.add_ramp_iv(bathsweep, temp, sb, ch, d, p_b90=p_sat) 
                s.update_temp_list(temp, 'temp_list') # Also, update temperature list 
                # What about updating tes_dict? In theory mux_map should contain all tes
                # that get iv curves, but Kaiwen's map didn't. 
                # That said, it's possible (though unlikely-not that unlikely actually) for one crazy 'IV' to get through cuts on non-tes.
                # 'TES_freq','pol','opt','opt_name','linestyle' <-required.
                # check_tes_dict_for_channel_and_init(tes_dict, sb, ch)                     
    
    def calc_P_b90(s,r_n,r_tes,p_tes): 
        # last two arrays.
        p_sat_idx = np.where(r_tes < 0.9 * r_n)[0] # standard SO P_sat def.
        if len(p_sat_idx) == 0:
            return False
        return  p_tes[p_sat_idx][-1] # again, data is time-reversed
                    
    def add_ramp_raw_iv(s, bathsweep_raw, temp, sb, ch, d):
        if sb not in bathsweep_raw.keys():
               bathsweep_raw[sb] = {}
        if ch not in bathsweep_raw[sb].keys():
            bathsweep_raw[sb][ch] = {"temp_raw":[], "p_satSM":[],"temp90R_n":[],"p_b90R_n":[],"R_nSM_raw":[]}
        bathsweep_raw[sb][ch]['temp_raw'].append(temp)
        bathsweep_raw[sb][ch]['p_satSM'].append(d['p_trans']) #Use pysmurf P_sat, since guaranteed to exist.
        # Try to also add the 90% R_n p_sat
        p_b90 = s.calc_P_b90(d['R_n'],d['R'],d['p_tes'])
        if p_b90:
            bathsweep_raw[sb][ch]['temp90R_n'].append(temp)
            bathsweep_raw[sb][ch]['p_b90R_n'].append(p_b90)
        bathsweep_raw[sb][ch]['R_nSM_raw'].append(d['R_n'])
        
    def add_ramp_iv(s,bathsweep, temp, sb, ch, d, p_b90=-42):
        if sb not in bathsweep.keys():
            bathsweep[sb] = {}
        if ch not in bathsweep[sb].keys():
            bathsweep[sb][ch] = {"temp":[], "p_b90":[], 'R_nSM':[]}
        bathsweep[sb][ch]['temp'].append(temp)
        if p_b90 == -42:
            p_b90 = s.calc_P_b90(d['R_n'],d['R'],d['p_tes'])
        bathsweep[sb][ch]['p_b90'].append(p_b90)
        bathsweep[sb][ch]['R_nSM'].append(d['R_n'])
    
    def do_normal_correction(s): 
        # corrects for pysmurf's poor fitting of the normal branch
        mux_map = s.mux_map
        tes_dict = s.tes_dict
        s.calculate_det_R_ns() # figure out what each detector's R_n is. 
        # now we need to change: [for each metadata, <stuff in comments>]
        # iv_analyzed_info_arr: [for each temp: 'iv_analyzed_fp']
        # AND: we reconstruct s.ramp_raw_arr and s.ramp_arr using the IV adders, 
        # referencing s.ramp_raw_arr_nnc and s.ramp_arr_nnc for which
        # temperatures had valid IV curves. 
        s.ramp_raw_arr = []
        s.ramp_arr = []
        for run_idx in range(len(s.iv_analyzed_info_arr_nnc)):
            iv_analyzed_info = s.iv_analyzed_info_arr[run_idx]
            orig_ramp = s.ramp_arr_nnc[run_idx] # needed to check if this real IV
            s.ramp_raw_arr.append({})
            s.ramp_arr.append({})   
            bathsweep_raw = s.ramp_raw_arr[run_idx]
            bathsweep = s.ramp_arr[run_idx]
            #ramp_raw = s.ramp_raw_arr()
            # currently just forcing R_n, nothing fancy with c:
            for line in iv_analyzed_info:
                iv_analyzed = line['iv_analyzed']
                temp = line['temp']
                for sb in iv_analyzed:
                    if sb == 'high_current_mode':
                        continue
                    for ch, iv in iv_analyzed[sb].items():
                        # we can only do meaningful corrections
                        # for channels that have actual iv curves. 
#                         if sb in tes_dict.keys() and ch in tes_dict[sb].keys()\
#                         and 'R_n' in tes_dict[sb][ch].keys():
                        # ...Let's apply the correction to everything that can get it, actually.
                        if sb in s.det_R_n.keys() and ch in s.det_R_n[sb].keys():
                            sc_idx, nb_idx = iv['trans idxs'][0],iv['trans idxs'][1]
                            # v hot fix: **TODO** real fix!! #py_start_idx == 0: 
                            lp = len(iv['p_tes'])
                            if not(lp-nb_idx <=3 or nb_idx -sc_idx <=3):
                                py_start_idx = -int((lp-nb_idx)/2)
                                iv_analyzed[sb][ch] = \
                                    s.pysmurf_iv_data_correction(iv,
                                            nb_start_idx=py_start_idx,
                                            force_R_n=s.det_R_n[sb][ch])
                                    
# **TODO**: Import my formal slope_fix_correct_one_iv instead!!
#                                 iv_analyzed[sb][ch] = \
#                                     slope_fix_correct_one_iv(iv,r_sh=s.r_sh,
#                                             nb_start_idx=py_start_idx,
#                                             force_R_n=s.det_R_n[sb][ch])
#                             else: 
#                                 print(f"SKIPPING R_n-correct: temp{temp} sb{sb}ch{ch} p_tes len{lp} idxs{iv['trans idxs']}")
                        if sb in s.ramp_raw_arr_nnc[run_idx] \
                        and ch in s.ramp_raw_arr_nnc[run_idx][sb] \
                        and temp in s.ramp_raw_arr_nnc[run_idx][sb][ch]['temp_raw']: 
                            s.add_ramp_raw_iv(bathsweep_raw, temp, sb, ch, iv_analyzed[sb][ch])
                        if  line['bl'] == mux_map[sb][ch]['biasline'] \
                        and sb in s.ramp_arr_nnc[run_idx].keys() \
                        and ch in s.ramp_arr_nnc[run_idx][sb].keys() \
                        and temp in s.ramp_arr_nnc[run_idx][sb][ch]['temp']: 
                            s.add_ramp_iv(bathsweep, temp, sb, ch, iv_analyzed[sb][ch])
    
    def calculate_det_R_ns(s):
        # for now, just taking the lowest R_n pysmurf gets that is > 0.006
        # Including of anything that wasn't really IV curves (namely, the 
        # Tb>T_c ones.)
        # ^ Nope. 
        # However we only bother to correct anything that actually
        # had an IV curve. 
        # note we do ramp_arr here, not ramp_raw_arr. Have to have actual 
        # iv curves to do a meaningful correction.
        tes_dict = s.tes_dict
        for run_index in range(len(s.ramp_arr)):
            ramp = s.ramp_arr[run_index]
            for sb in ramp.keys():
                for ch,d in ramp[sb].items():
                    if sb not in s.det_R_n.keys():
                        s.det_R_n[sb] = {}
                        s.det_R_n_NBL[sb] = {}
                    # Trying to include the post-IV curves didn't work well. 
                    #my_R_ns = np.array(s.ramp_raw_arr[run_index][sb][ch]['R_nSM_raw'])
                    #believe_R_ns = my_R_ns[np.where(my_R_ns > 0.006)[0]]
                    #min_R_n = my_R_ns[-1]
                    #min_R_n = min(my_R_ns[np.where(my_R_ns > 0.006)[0]])
                    #min_R_n = min(d['R_nSM'])
                    
                    # It is possible (and only uncommon) to mess up the other way, 
                    # and get an R_n that is too small with a lower bath temp point.
                    # So, HAS to be the last one. (Not really "min_R_n" anymore.)
                    #min_R_n = min(d['R_nSM'])
                    best_R_n = d['R_nSM'][-1]
                    # should have passed standalone cuts and had indices fixed...
                    temp = d['temp'][-1]
                    iva = s.find_temp_ch_iv_analyzed(sb,ch,temp,run_indexes=[run_index])[0]
                    longest_NBL = iva['v_bias'][-1] - iva['v_bias'][iva['trans idxs'][1]]
                    if ch not in s.det_R_n[sb].keys():
                        s.det_R_n[sb][ch]     = best_R_n
                        s.det_R_n_NBL[sb][ch] = longest_NBL
                    # if this is a tes, update tes_dict; 
                    if sb in tes_dict.keys() and ch in tes_dict[sb].keys():
                        if 'R_n' not in tes_dict[sb][ch].keys():#  or min_R_n < tes_dict[sb][ch]['R_n']:
                            tes_dict[sb][ch]['R_n']     = best_R_n
                            tes_dict[sb][ch]['R_n_NBL'] = longest_NBL
                        else:
                            if longest_NBL < tes_dict[sb][ch]['R_n_NBL']: # pre-existing better one
                                s.det_R_n[sb][ch]     = tes_dict[sb][ch]['R_n']
                                s.det_R_n_NBL[sb][ch] = tes_dict[sb][ch]['R_n_NBL']
                            else: # pre-existing worse one
                                tes_dict[sb][ch]['R_n']     = best_R_n
                                tes_dict[sb][ch]['R_n_NBL'] = longest_NBL
                            
                    
#     def calc_i_bias(s,iva):
#         if 0 in iva['R']:
#             return np.ones(len(iva['R']))
#         return iva['v_tes']*(1/s.r_sh + 1/iva['R'])
    
    def pysmurf_iv_data_correction(s, py_ch_iv_analyzed,\
                                   nb_start_idx=0,nb_end_idx=-1,force_R_n=-42.0,
                                  force_c=-42.0):
        # **TODO**: Import my formal slope_fix_correct_one_iv instead!!
        # nb_end_idx exists because I often saw this little downturn in the 
        # iv curve at the very highest v_bias. I'm not sure why, but it's in mv6 too.
        # takes a pysmurf data dictionary for a given (sb,ch), 
        # backcalculates the rawer values, makes corrected & expanded version.
        # Except I am not yet dealing with the responsivity calculations. 
        # And, if r_n provided, forces the norm fit line's slope to be 1/(r_n/r_sh+1)
        # dict_keys(['R' [Ohms], 'R_n', 'trans idxs', 'p_tes' [pW], 'p_trans',\
        # 'v_bias_target', 'si', 'v_bias' [V], 'si_target', 'v_tes_target', 'v_tes' [uV]])
        r_sh = s.r_sh 
        iv = {} # what I will return. iv data
        iv_py = py_ch_iv_analyzed
        # first, things I don't need to change, really:
        iv['trans idxs'] = iv_py['trans idxs']
        iv['v_bias'] = iv_py['v_bias'] # commanded voltage over whole line and DR inputs
        # as of 11/10/2021, pysmurf's v_bias_target is bugged, 
        # hence not just copying iv's value for it.
        #  Now, get fundamentals: v_bias_bin, i_bias_bin [uA], and i_tes [uA] (with an offset) = resp_bin  
        # v_tes = i_bias_bin*1/(1/R_sh + 1/R), so i_bias_bin = v_tes*(1/R_sh + 1/R)
        iv['i_bias'] =  iv_py['v_tes']*(1/s.r_sh + 1/iv_py['R'])
        # R = r_sh * (i_bias_bin/resp_bin - 1), so 1/((R/r_sh+1)/i_bias_bin) = resp_bin
        resp_bin = iv['i_bias']/(iv_py['R']/r_sh + 1)
        # Now the moment we're all waiting for, fixing resp_bin's offset. 
        # A fit really isn't the best R_n estimate. 
        # The best R_n estimate is the last resistance in a proper R calculation.
        # But I can't get that without making some R_n estimate to get c. 
        #nb_fit_idx = int(s.r_n_start_percent*len(resp_bin))# Ours here. Not pysmurf's.
        #nb_fit_idx = s.r_n_start_idx # defaults to -10. 
        if (not force_c == -42) and (not force_R_n==-42):
            norm_fit = [force_R_n,force_c]
            r_n = force_R_n
        else:
            if nb_start_idx: # caller wants a specific nb_start_idx
                nb_fit_idx = nb_start_idx
            else:
                nb_idx = iv['trans idxs'][1] # point with minimum derivative btw sc_idx and end
                py_start_idx = -int((len(iv['i_bias'])-nb_idx)/2)
                nb_fit_idx = py_start_idx # TODO: do the derivative thing!!!
                # Katie does some smoothing.
                d_resp = np.diff(resp_bin)
                # looking for the place where second discrete derivative is flat 
                # and constant the same.
                # TODO!!!!!! this is still stolen from Katie mostly. Needs work!!
#                 try:
                new_index = nb_idx
                new_test = np.polyfit(np.arange(len(d_resp[new_index:-10])), \
                                      d_resp[new_index:-10], deg=1)
                while new_index <= -10:
                    old_test = new_test
                    old_index = new_index

                    new_index = int(0.75*new_index)

                    new_test = np.polyfit(np.arange(len(d_resp[new_index:-10])), \
                                      d_resp[new_index:-10], deg=1)
                    delta_slope = (new_test[1]-old_test[1])/old_test[1]
                    if np.abs(delta_slope)<=0.02: # 0.05:
                        break
                nb_fit_idx=old_index
#                 except:
#                     iv.flag[i]=True
#                     nb_idx = int(len(iv.i_bias[b])*0.2)
            i_bias_nb = iv['i_bias'][nb_fit_idx:nb_end_idx]
            #smooth_i_bias_nb = find_smooth_indices()
            # pysmurf fixed polarity for us, so don't need to worry about that. 
            if force_R_n == -42.0: # Caller hasn't given us an r_n to force. 
                norm_fit = np.polyfit(iv['i_bias'][nb_fit_idx:nb_end_idx],\
                                      resp_bin[nb_fit_idx:nb_end_idx],1)
                r_n = r_sh*(1/norm_fit[0]-1) # our fit r_n
            else: # caller gave us an r_n to force. 
                r_n = force_R_n
                norm_fit = []
                forced_slope = 1/(r_n/r_sh+1)
                i_tes_forced_r_n = lambda i_bias, c : forced_slope*i_bias + c            
                norm_fit.append(forced_slope) # forced slope. 
                # We don't want to make a fit because we know the lower points
                # will be more off. 
                # Instead, run norm_fit through a high-V_b value to calc c, 
                # forced line's y-intercept.
                # TODO: Need to do some averaging or something here!!!
                #norm_fit.append(resp_bin[nb_idx_end] - forced_slope*iv['i_bias'][-1])
                # However, last value always has that downturn.
                # So use a point a little bit away.

                # I think we have to make a fit, because running through just 1 point
                # is crazy risky, and I don't see an obvious way to smooth it effectively.
                # I guess literally just median smoothing could work...
                (popt, pcov) = curve_fit(i_tes_forced_r_n,
                                        iv['i_bias'][nb_fit_idx:nb_end_idx],
                                        resp_bin[nb_fit_idx:nb_end_idx])
                norm_fit.append(popt[0]) # c!
            
        # TODO: make residual plotter! 
        iv['i_tes'] = resp_bin - norm_fit[1] # there we go. 
        iv['c_norm'] = norm_fit[1]
        #  get pysmurf's target R_op_target:
        #py_target_idx = np.ravel(np.where(iv_py['v_tes']==iv_py['v_tes_target']))[0]
        # above doesn't work b/c sc offset removal can remove the point that it targeted, so:
        v_targ = iv_py['v_tes_target']
        py_target_idx = np.ravel(np.where( abs(iv_py['v_tes']-v_targ) == min(abs(iv_py['v_tes']-v_targ)) ))[0]
        R_op_target = iv_py['R'][py_target_idx]    
        # Now just refill the rest of the keys same way pysmurf does (except do correct v_bias_target):
        # dict_keys(['R' [Ohms], 'R_n', 'trans idxs', 'p_tes' [pW], 'p_trans',\
        # 'v_bias_target', 'si', 'v_bias' [V], 'si_target', 'v_tes_target', 'v_tes' [uV]]
        iv['R'] = r_sh * (iv['i_bias']/iv['i_tes'] - 1)
        iv['R_n'] = r_n 
        # the correct equivalent of pysmurf's i_R_op
        i_R_op = 0
        for i in range(len(iv['R'])-1,-1,-1):
            if iv['R'][i] < R_op_target:
                i_R_op = i
                break
        iv['v_bias_target'] = iv['v_bias'][i_R_op]
        iv['v_tes'] = iv['i_bias']/(1/r_sh + 1/iv['R']) # maybe use R_sh*(i_bias-i_tes)?
        iv['v_tes_target'] = iv['v_tes'][i_R_op]
        iv['p_tes'] = iv['v_tes']**2/iv['R']
        # pysmurf's p_trans: 
        iv['p_trans'] = np.median(iv['p_tes'][iv['trans idxs'][0]:iv['trans idxs'][1]])
        # SO p_trans (if possible):
        if len(np.ravel(np.where(iv['R']<0.9*iv['R_n']))) > 0:
            iv['p_b90'] = iv['p_tes'][np.ravel(np.where(iv['R']<0.9*iv['R_n']))[-1]]
        else:
            iv['p_b90'] = -42.0 # you never reach it. 
        # TODO: STILL NEED TO ADD si and si_target! 
        return iv   
    
    def update_temp_list(s,temp, temp_list_name):
        temp_list = getattr(s,temp_list_name)
        if temp not in temp_list:
            if len(temp_list) == 0:
                setattr(s,temp_list_name, [temp])
            else:
                i=0
                stop = False
                while i < len(temp_list) and not stop:
                    if temp < temp_list[i]:
                        setattr(s,temp_list_name,temp_list[:i] + [temp] + temp_list[i:])
                        stop = True
                    i+=1
                if not stop: # greatest in the list
                    setattr(s,temp_list_name,temp_list + [temp])
                    #temp_list.append(temp) 

    def merge_ramps(s,bs1,bs2):
        mbs = {} # merged bathsweep
        # The below will be useful if I ever add R_n to ramp...
        arr_keys = ['temp','p_b90','R_nSM'] # keys with arrays that go with temp 
                                    # and need to be merged
        sbs = s.dUnique(bs1,bs2) 
        for sb in sbs:
            mbs[sb]={}
            sb_chs = s.dUnique(bs1[sb],bs2[sb])
            for ch in sb_chs:
                mbs[sb][ch]={}
                for key in arr_keys:
                    mbs[sb][ch][key] = []
                if ch in bs1[sb].keys() and ch in bs2[sb].keys(): # Both have this channel. Interleave. 
                    i1, i2 = 0,0
                    temp1, temp2 = bs1[sb][ch]['temp'], bs2[sb][ch]['temp']
                    d1,d2 = bs1[sb][ch],bs2[sb][ch]
                    p_sat1, p_sat2 = bs1[sb][ch]['p_b90'], bs2[sb][ch]['p_b90']
                    while i1 < len(temp1) and i2 < len(temp2):
                        if temp1[i1] <= temp2[i2]:
                            for key in arr_keys:
                                mbs[sb][ch][key].append(d1[key][i1])
                            i1 += 1
                        else:
                            for key in arr_keys:
                                mbs[sb][ch][key].append(d2[key][i2])
                            i2 += 1
                    if i1 < len(temp1):
                        for key in arr_keys:
                            mbs[sb][ch][key] += d1[key][i1:]
                    if i2 < len(temp2):
                        for key in arr_keys:
                            mbs[sb][ch][key] += d2[key][i2:]

                elif ch in bs1[sb].keys(): # only bs1 has this channel
                    for key in arr_keys: # can't use d1/d2 here, they aren't defined!
                        mbs[sb][ch][key] = bs1[sb][ch][key]
                else:                      # only bs2 has this channel
                    for key in arr_keys:
                        mbs[sb][ch][key] = bs2[sb][ch][key]
        return mbs
    
    def dUnique(s,d1,d2):
        # merge two dictionary's key lists into numpy array.
        return np.unique([key for key in d1.keys()]+[key for key in d2.keys()])
    
    
    def categorize_ivs(s):
        # run if you want to do iv analysis.
        s.iv_cat = {'is_iv':[],'not_iv':[],
                    'wrong_bl':[],'bad_channel':[],'bad_temp':[]}
        for iv_analyzed_info in s.iv_analyzed_info_arr:
            for iv_line in iv_analyzed_info:
                s.categorize_ivs_helper(iv_line)
                
    def categorize_ivs_helper(s,iv_line):
        iva_dict = iv_line['iv_analyzed']
        for sb in iva_dict.keys():
            if sb == 'high_current_mode':
                continue
            for ch,d in iva_dict[sb].items():
                full_name = (sb,ch,iv_line['temp'],iv_line['bl'])
                if not s.test_device.mux_map[sb][ch]['biasline'] == iv_line['bl']:
                    s.iv_cat['wrong_bl'].append(full_name)
                    d['is_iv_info'] = "wrong_bl"
                elif ch not in s.ramp[sb].keys():
                    s.iv_cat['bad_channel'].append(full_name)
                    d['is_iv_info'] = "bad_channel"
                elif iv_line['temp'] not in s.ramp[sb][ch]['temp']:
                    s.iv_cat['bad_temp'].append(full_name)
                    d['is_iv_info'] = "bad_temp"    
                else:
                    s.iv_cat['is_iv'].append(full_name)
                    d['is_iv'] = True
                    d['is_iv_info'] = "is_iv"
                    continue
                d['is_iv'] = False
                s.iv_cat['not_iv'].append(full_name)
                
    def redo_cuts(s):
        # maybe run this when norm_correct makes BIG difference
        s.ramp_raw_arr = []
        s.ramp_arr = []
        s.ramp = []
        s.load_temp_sweep_IVs()
        s.ramp = s.ramp_arr[0] # merging
        for i in range(1,len(s.ramp_arr)):
            s.ramp = s.merge_ramps(s.ramp, s.ramp_arr[i])              
                
    # ==================== PLOTTING FUNCTIONS ===============
    def find_temp_ch_iv_analyzed(s,sb,ch,temp,bl=-42,run_indexes=[]):
        if not run_indexes:
            run_indexes = range(len(s.iv_analyzed_info_arr))
        temp_ch_iv_analyzed_arr = []
        for i in run_indexes:
            analyzed_iv_info = s.iv_analyzed_info_arr[i]
            for dicty in analyzed_iv_info:
                if abs(dicty['temp']-temp) < 0.5 \
                and sb in dicty['iv_analyzed'].keys() \
                and ch in dicty['iv_analyzed'][sb].keys():
                    # Make sure the real one, or chosen one, goes first
                    if (dicty['bl'] == s.test_device.mux_map[sb][ch]['biasline'] \
                    and len(temp_ch_iv_analyzed_arr)>0 and bl==-42)\
                    or dicty['bl']==bl:
                        temp_ch_iv_analyzed_arr = [dicty['iv_analyzed'][sb][ch]] \
                                                  + temp_ch_iv_analyzed_arr
                    elif bl==-42:
                        temp_ch_iv_analyzed_arr.append(dicty['iv_analyzed'][sb][ch])
                    dicty['iv_analyzed'][sb][ch]['bl'] = dicty['bl'] 
                    # just adding a key for plotRP easy.
        return temp_ch_iv_analyzed_arr
    
    def plot_IV(s,sb,ch,temp,x_lim=[],y_lim=[], tes_dict=True, 
                bl=-42, run_indexes=[], own_fig=True,linewidth=1):
        temp_ch_iv_analyzed_arr = s.find_temp_ch_iv_analyzed(sb,ch,temp,bl=bl,
                                                             run_indexes=run_indexes)
        # This is only going to do the first one if no run_index specified, so that 
        # I can do own_fig stuff. 
        # issue of multiple bl options. 
        if not temp_ch_iv_analyzed_arr:
            return []
        d = temp_ch_iv_analyzed_arr[0]
        if 'i_bias' in d.keys() and 'i_tes' in d.keys():
            i_bias, i_tes = d['i_bias'], d['i_tes']
        else:
            i_bias = d['v_tes']*(1/s.r_sh + 1/d['R'])
            d['i_bias'] = i_bias
            # R = r_sh * (i_bias_bin/resp_bin - 1), so 1/((R/r_sh+1)/i_bias_bin) = resp_bin
            i_tes = i_bias/(d['R']/s.r_sh + 1)
            d['i_tes'] = i_tes
        (plotted,d) = s.internal_plot_iv_analyzed_keys(d,sb,ch,temp,'i_bias','i_tes', 
                              x_lim=x_lim,y_lim=y_lim, tes_dict=tes_dict, 
                              bl=bl,run_indexes=run_indexes,own_fig=own_fig,
                             linewidth=linewidth)
        color = plotted[0].get_color()
        plt.vlines([i_bias[d['trans idxs'][0]]],
                min(i_tes),max(i_tes), colors=color, linestyles="dotted")
        plt.vlines([i_bias[d['trans idxs'][1]]],
                min(i_tes),max(i_tes), colors=color, linestyles="dashdot")
    
    
    
    def plot_iv_analyzed_keys(s,x_key,y_key, sb, ch, temp, 
                              x_lim=[],y_lim=[], tes_dict=True,bl=-42, run_indexes=[], 
                              own_fig=True,linewidth=1):
        # separate function so that i can check if i_bias and i_tes exist and ad them
        # if they don't. 
        temp_ch_iv_analyzed_arr = s.find_temp_ch_iv_analyzed(sb,ch,temp,bl=bl,
                                                             run_indexes=run_indexes)
        # This is only going to do the first one if no run_index specified, so that 
        # I can do own_fig stuff. 
        if not temp_ch_iv_analyzed_arr:
            return []
        d = temp_ch_iv_analyzed_arr[0]
        return s.internal_plot_iv_analyzed_keys(d, sb,ch,temp,x_key,y_key,
                              x_lim=x_lim,y_lim=y_lim, tes_dict=tes_dict, 
                              bl=bl, run_indexes=run_indexes,own_fig=own_fig,
                                linewidth=linewidth)
        
    
    def internal_plot_iv_analyzed_keys(s,d,sb,ch,temp, x_key,y_key,
                              x_lim=[],y_lim=[], tes_dict=True, bl=-42,run_indexes=[], 
                              own_fig=True,linewidth=1):    
        if tes_dict:
            tes_dict=s.tes_dict
        linestyle = 'solid'
        curve_name = f"sb{sb} ch{ch} {int(temp)}{s.key_info['temp']['units']}"
        if tes_dict and sb in tes_dict.keys() and ch in tes_dict[sb].keys():
            curve_name = curve_name + f" {tes_dict[sb][ch]['TES_freq']} {tes_dict[sb][ch]['opt_name']}"
            #linestyle = tes_dict[sb][ch]['linestyle']
        if not bl==-42:
            curve_name = f"bl {bl} "+ curve_name
        if own_fig:
            plt.figure(figsize=default_figsize)
            plotted = plt.plot(d[x_key],d[y_key], linestyle=linestyle,
                               linewidth=linewidth,label=curve_name)
            plt.title(curve_name)
        else:
            plotted =plt.plot(d[x_key],d[y_key], label=curve_name, 
                              linestyle=linestyle,linewidth=linewidth)
            plt.title(f"{s.dName} {s.key_info[y_key]['name']} vs. {s.key_info[x_key]['name']}")
        plt.ylabel(f"{s.key_info[y_key]['name']} [{s.key_info[y_key]['units']}]")
        plt.xlabel(f"{s.key_info[x_key]['name']} [{s.key_info[x_key]['units']}]")
        if x_lim:
            plt.xlim(x_lim[0],x_lim[1])
        else:
            if 'lim' in s.key_info[x_key].keys():
                plt.xlim(s.key_info[x_key]['lim'][0],s.key_info[x_key]['lim'][1])
        if y_lim:
            plt.ylim(y_lim[0],y_lim[1])
        else:
            if 'lim' in s.key_info[y_key].keys():
                plt.ylim(s.key_info[y_key]['lim'][0],s.key_info[y_key]['lim'][1])
        if not own_fig:
            plt.legend()#bbox_to_anchor=(1.01,1),loc="upper left"
        return (plotted,d)
    
    def plot_RP_easy(s,sb, ch, temp, tes_dict=True, bl=-42,run_indexes=[]):
        ivas = s.find_temp_ch_iv_analyzed(sb,ch,temp,bl=bl,run_indexes=run_indexes)
        plot_array = []
        if tes_dict:
            tes_dict=s.tes_dict
        for iva in ivas:
            #print('found it')
            d = iva # dicty['iv_analyzed'][sb][ch]
            plot_array.append(plt.figure(figsize=default_figsize))
            plt.plot(d['p_tes'],d['R'])
            plt.title(f"{s.dName} bl{d['bl']} sb{sb} ch{ch} temp{int(temp)} {s.key_info['temp']['units']}")
            plt.xlabel('P_tes [pW]')
            plt.ylabel('R_tes [Ohm]')
            plt.hlines(d['R_n'], 0, max(d['p_tes']), colors=['c'], \
                       linestyles='dashed', \
                       label=f"R_nSM this temp={d['R_n']:.4} {s.key_info['R_nSM']['units']}")
            if tes_dict and 'R_n' in tes_dict[sb][ch].keys():
                plt.hlines(tes_dict[sb][ch]['R_n'], 0, max(d['p_tes']), colors=['b'],\
                           linestyles='dashed',\
                           label=f"R_n TES best ={tes_dict[sb][ch]['R_n']:.4} {s.key_info['R_nSM']['units']}")
            #int(tes_dict[sb][ch]['R_n']*10000)/10000.0
            ymax = d['R_n']*1.1
            plt.vlines(d['p_trans'],0, ymax, colors=['r'],linestyles="dashed",\
                       label=f"Pysmurf (median) P_sat={int(d['p_trans']*10)/10.0} pW")
            p_sat = d['p_tes'][np.where(d['R'] < 0.9 * d['R_n'])[0][-1]]
            plt.vlines(p_sat,0, ymax, colors=['g'],linestyles="dashed", \
                       label=f"90% R_n P_b={int(p_sat*10)/10.0} pW")
            plt.xlim(0, max(d['p_tes']))
            plt.ylim(0, ymax)
            plt.legend()
        return plot_array
    
    def plot_RP(s,analyzed_iv_info, temp, bl, sb, ch):
        # was critical for deciding on the cutting program.
        # but legacy code at this point, likely removable. 
        for dicty in analyzed_iv_info:
            if dicty['temp']==temp and float(dicty['bl']) == str(bl):
                #print('found it')
                d = dicty['iv_analyzed'][sb][ch]
                plt.figure(figsize=default_figsize)
                plt.plot(d['p_tes'],d['R'])
                plt.title(f"{s.dName} bl{bl} sb{sb} ch{ch} temp{temp} {s.key_info['temp']['units']}")
                plt.xlabel('P_tes [pW]')
                plt.ylabel('R_tes [Ohm]')
                plt.hlines(d['R_n'], 0, max(d['p_tes']), colors=['c'], linestyles='dashed',label="R_n")
                ymax = d['R_n']*1.1
                plt.vlines(d['p_trans'],0, ymax, colors=['r'],linestyles="dashed", label=f"Pysmurf P_sat={int(d['p_trans']*10)/10.0}pW")
                p_sat = d['p_tes'][np.where(d['R'] < 0.9 * d['R_n'])[0][-1]]
                plt.vlines(p_sat,0, ymax, colors=['b'],linestyles="dashed", label=f"90% R_n P_b={int(p_sat*10)/10.0}pW")
                plt.xlim(0, max(d['p_tes']))
                plt.ylim(0, ymax)
                plt.legend()
                
    def plot_ramp_raw(s,ramp_raw,tes_dict=True):
        return s.plot_ramp_keys_by_sb_2legend(ramp_raw,'temp_raw','p_satSM', \
                    prefix='Raw ',tes_dict=tes_dict)

    def plot_ramp_raw90R_n(s,ramp_raw,tes_dict=True):
        return s.plot_ramp_keys_by_sb_2legend(ramp_raw,'temp90R_n','p_b90R_n', \
                    prefix='Raw ',tes_dict=tes_dict)

    def plot_ramp(s,ramp,tes_dict=True,zero_starts=False):
        if not zero_starts:
            y_lim = [0,15]
        else:
            y_lim = []
        return s.plot_ramp_keys_by_sb_2legend(ramp,'temp','p_b90',y_lim=y_lim, \
                    prefix='Cut ',tes_dict=tes_dict,zero_starts=zero_starts)

    def plot_ramp_keys_by_sb(s,bathsweep,x_key,y_key,x_lim=[],y_lim=[], 
                             prefix='',tes_dict=True,zero_starts=False):
        plot_array = []
        linestyle,label =  'solid', ''
        tab20 = plt.get_cmap('tab20')
        if tes_dict:
            tes_dict=s.tes_dict
        for sb in bathsweep.keys():
            #hsv = plt.get_cmap('hsv')
            colors = tab20(np.linspace(0, 1.0, len(bathsweep[sb].keys())+1))
            col_ind=0
            plot_array.append((plt.figure(figsize=default_figsize), sb)) #(default_figsize[0]*1.5,default_figsize[1])
            for ch, d in bathsweep[sb].items():
                if tes_dict and sb in tes_dict.keys() and ch in tes_dict[sb].keys():
                    linestyle = tes_dict[sb][ch]['linestyle']
                    label = f' {str(tes_dict[sb][ch]["TES_freq"])} {tes_dict[sb][ch]["opt_name"]}'
                ys = d[y_key]
                if zero_starts:
                    ys = d[y_key]-d[y_key][0]
                else:
                    ys = d[y_key]
                plt.plot(d[x_key], ys, label=f'ch{ch}' + label,
                         linestyle=linestyle, color=colors[col_ind])
                col_ind += 1
            if x_lim:
                plt.xlim(x_lim[0],x_lim[1])
            else:
                if 'lim' in s.key_info[x_key].keys():
                    plt.xlim(s.key_info[x_key]['lim'][0],s.key_info[x_key]['lim'][1])
            if y_lim:
                plt.ylim(y_lim[0],y_lim[1])
            elif not zero_starts:
                if 'lim' in s.key_info[y_key].keys():
                    plt.ylim(s.key_info[y_key]['lim'][0],s.key_info[y_key]['lim'][1])
            plt.legend() #bbox_to_anchor=(1.01,1),loc="upper left"
            plt.ylabel(f"{s.key_info[y_key]['name']} [{s.key_info[y_key]['units']}]")
            plt.xlabel(f"{s.key_info[x_key]['name']} [{s.key_info[x_key]['units']}]")
            plt.title(f"{prefix}{y_key} vs. {x_key} smurf band {sb} {s.dName}")
            #plt.tight_layout(rect=[0,0,0.75,1])
            #plt.subplots_adjust(right=0.75)
        return plot_array
    
    def plot_ramp_keys_by_sb_2legend(s,bathsweep,x_key,y_key,x_lim=[],y_lim=[], 
                             prefix='',tes_dict=True,zero_starts=False):
        plot_array = []
        linestyle,label =  'solid', ''
        tab20 = plt.get_cmap('tab20')
        opt_names = []
        linestyles = []
        if tes_dict:
            tes_dict=s.tes_dict
        for sb in bathsweep.keys():
            #hsv = plt.get_cmap('hsv')
            colors = tab20(np.linspace(0, 1.0, len(bathsweep[sb].keys())+1))
            col_ind=0
            plot_array.append((plt.figure(figsize=(default_figsize[0]*1.2,default_figsize[1])), sb))
            for ch, d in bathsweep[sb].items():
                if tes_dict and sb in tes_dict.keys() and ch in tes_dict[sb].keys():
                    linestyle = tes_dict[sb][ch]['linestyle']
                    if tes_dict[sb][ch]["opt_name"] not in opt_names:
                        opt_names.append(tes_dict[sb][ch]["opt_name"])
                        linestyles.append(linestyle)
                    #label = f' {str(tes_dict[sb][ch]["TES_freq"])} {tes_dict[sb][ch]["opt_name"]}'
                    label = f' {str(tes_dict[sb][ch]["TES_freq"])}'
                ys = d[y_key]
                if zero_starts:
                    ys = d[y_key]-d[y_key][0]
                else:
                    ys = d[y_key]
                plt.plot(d[x_key], ys, label=f'ch{ch}' + label,
                         linestyle=linestyle, color=colors[col_ind])
                col_ind += 1
            if x_lim:
                plt.xlim(x_lim[0],x_lim[1])
            else:
                if 'lim' in s.key_info[x_key].keys():
                    plt.xlim(s.key_info[x_key]['lim'][0],s.key_info[x_key]['lim'][1])
            if y_lim:
                plt.ylim(y_lim[0],y_lim[1])
            elif not zero_starts:
                if 'lim' in s.key_info[y_key].keys():
                    plt.ylim(s.key_info[y_key]['lim'][0],s.key_info[y_key]['lim'][1])
            ch_legend = plt.legend(bbox_to_anchor=(1.01,1),loc="upper left") # bbox_to_anchor=(1.2,0.1) the positioning doesn't work...
            # Now do the linestyle legend
            if len(linestyles) > 0:
                line_handles = []
                for i in range(len(opt_names)):
                    handle, = plt.plot([],[],linestyle=linestyles[i],color='0')
                    line_handles.append(handle)
                plt.legend(line_handles,opt_names,loc='best')
                plt.gca().add_artist(ch_legend) # put channel legend back in.
            plt.ylabel(f"{s.key_info[y_key]['name']} [{s.key_info[y_key]['units']}]")
            plt.xlabel(f"{s.key_info[x_key]['name']} [{s.key_info[x_key]['units']}]")
            plt.title(f"{prefix}{y_key} vs. {x_key} smurf band {sb} {s.dName}")
            #plt.tight_layout(rect=[0,0,0.75,1])
            plt.subplots_adjust(right=0.77)
        return plot_array

    # biasline
    def plot_ramp_by_BL(s,bathsweep,tes_dict=True,y_lim=[0,8]):
        return s.plot_ramp_keys_by_BL(bathsweep,'temp','p_b90',
                                     y_lim=y_lim,prefix="Cut ",tes_dict=tes_dict)
    
    def plot_ramp_keys_by_BL(s,bathsweep, x_key,y_key,
                             x_lim=[],y_lim=[], prefix='',tes_dict=True):
        mux_map = s.mux_map
        if tes_dict:
            tes_dict=s.tes_dict
        linestyle,label =  'solid', ''
        #fig, ax = plt.subplots(nrows=4, figsize=(9,9))
        fig, ax = plt.subplots(nrows=4, ncols=3, figsize=(9,9))
        for r in range(len(ax)):
            row = ax[r]
            for c in range(len(row)):
                p = row[c]
                for sb in bathsweep.keys():
                    for ch, d in bathsweep[sb].items():
                        #print(mux_map[sb][ch])
                        if mux_map[sb][ch]['biasline'] == 3*r + c:
                            if tes_dict and sb in tes_dict.keys() and ch in tes_dict[sb].keys():
                                linestyle = tes_dict[sb][ch]['linestyle']
                                label = f' {str(tes_dict[sb][ch]["TES_freq"])} {tes_dict[sb][ch]["opt_name"]}'
                            p.plot(d[x_key], d[y_key], 
                                   label=f'ch{ch}'+label,
                                   linestyle=linestyle)
                #p.set_xlabel('bath temp [mK]')
                #p.set_ylabel('90% R_n P_sat [pW]')
                p.set_title("Bias line " + str(3*r + c))
                if x_lim:
                    p.set_xlim(x_lim[0],x_lim[1])
                else:
                    if 'lim' in s.key_info[x_key].keys():
                        p.set_xlim(s.key_info[x_key]['lim'][0],s.key_info[x_key]['lim'][1])
                if y_lim:
                    p.set_ylim(y_lim[0],y_lim[1])
                else:
                    if 'lim' in s.key_info[y_key].keys():
                        p.set_ylim(s.key_info[y_key]['lim'][0],s.key_info[y_key]['lim'][1])
                #p.legend(fontsize='small') # These plots don't really have space for legend.
        plt.suptitle(f"{s.dName} {prefix}{s.key_info[y_key]['name']} [{s.key_info[y_key]['units']}] vs. {s.key_info[x_key]['name']} [{s.key_info[x_key]['units']}]")
        plt.tight_layout()
        return (fig, ax)
    

    
# -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
# =================== Bath_Ramp Class (Temp_Ramp child class) ==================
# -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~

class Bath_Ramp(Temp_Ramp):
    '''
    This child class deals with fitting and plotting results of 
    bath Temp_Ramps. 
    
    Adds the following to its passed tes_dict's channel dictionaries (
    even if the channel is excluded from results because of its opt_value):
    k, k_err, tc, tc_err, n, n_err, cov_G_tc_n (default) OR cov_k_tc_n, 
    G, G_err, p_sat100mK, p_sat100mK_err
    
    ONE BIG FLAW: no accounting for optical power on unmasked detectors TODO
    ^somewhat done with cold load ramp...
    
    # =========================== (Child) VARIABLES ============================
    opt_values_exclude# Which to exclude from 'results' (NOT all_results)
    min_b_temp_span   # minimum temperature range of bath that channel must have 
                      # points spanning in order to fit it as a real channel. 
    use_k             # Whether to fit to k (if not, to G). G IS DEFAULT. 
    g_guess_by_freq   # {TES_freq (int): guessed G for fits in pW/mK } 
    tc_guess          # in mK
    n_guess           # unitless
    
    # --- Constructed
    all_results       # k, G, tc, n, cov_G_tc_n (default) OR cov_k_tc_n
    
    # Organizational, for plotting, does not include excluded opt_values:
    results           # {sb:{ch:{k, G, tc, n, cov_G_tc_n OR cov_k_tc_n
    to_plot
    to_plot_TES_freqs 
    
    # ======================= METHODS (Helpers indented) =======================
    __init__(s, test_device, ramp_type, therm_cal, metadata_fp_arr, 
             norm_correct=True, p_sat_cut=15, use_p_satSM=True, 
             fix_pysmurf_idx=False,input_file_type="pysmurf",
             opt_values_exclude=[],min_b_temp_span=21,use_k=False,
             g_guess_by_freq={90: 0.090, 150: 0.165,225: 0.3,285:0.5},
             tc_guess=160,n_guess=3.5)
        do_fits(s)
            add_thermal_param_to_dict(s,dicty,use_k,k,g,tc,n,cov,p_sat100mK_err)
            p_satofT(s,Tb, k, Tc, n)
            p_satofTandG(s,Tb, G, Tc, n)
    # ---------- Plotting Methods
    make_summary_plots(s, nBins=60,split_freq=True, p_sat_Range=None,
                       t_c_Range=None,g_Range=None, n_Range=None)
    '''
    
    def __init__(s, test_device, ramp_type, therm_cal, metadata_fp_arr, 
                 norm_correct=True, p_sat_cut=15, use_p_satSM=True, 
                 fix_pysmurf_idx=False,input_file_type="pysmurf",
                 opt_values_exclude=[],min_b_temp_span=21,use_k=False,
                 g_guess_by_freq={90: 0.090, 150: 0.165,225: 0.3,285:0.5},
                 tc_guess=160,n_guess=3.5): 
        # SHOULD be opt_values_exclude, not bias lines
        super().__init__(test_device, ramp_type, therm_cal,metadata_fp_arr, 
                         norm_correct=norm_correct,p_sat_cut=p_sat_cut,
                        use_p_satSM=use_p_satSM,fix_pysmurf_idx=fix_pysmurf_idx,
                        input_file_type=input_file_type)
        # Now that the Temp_Ramp is set up...let's fit that data. 
                
        # Guesses for the fitting to start from. Important for yield. 
        s.g_guess_by_freq = g_guess_by_freq # int(freq):G in pW/mK
        s.tc_guess = tc_guess # mK
        s.n_guess = n_guess
        # passing do_fits more stuff
        s.opt_values_exclude = opt_values_exclude
        s.use_k = use_k
        s.min_b_temp_span = min_b_temp_span
        
        # This is a separate function so I can fiddle with things and re-run
        # As ex. I did with light leak experiment in SPB14, and in SPB3 when I 
        # didn't have time to fix the fitting function for the wacky procedure
        # from back when we didn't have real PID.
        s.do_fits()
        
        
    def do_fits(s):
        opt_values_exclude,use_k, g_guess_by_freq,tc_guess, n_guess = \
          (s.opt_values_exclude,s.use_k, s.g_guess_by_freq,s.tc_guess, s.n_guess)
        mux_map = s.mux_map
        tes_dict = s.tes_dict
        
        # tes_dict = {sb:{ch:{TES_atts}}}; TES_freq,pol,opt,opt_name,linestyle
        #       added by data analysis: R_n?; k,Tc,n,G,psat_100mK; opt_eff; 
        
        s.all_results = {}
        results = {}
        
        #print(['k','tc','n'])
        for sb in s.ramp.keys():
            for ch, d in s.ramp[sb].items():
                if len(d['p_b90']) < 4: # originally < 4
                    continue
                # Merging bathsweeps resulted in ones getting 4 points that have no business being fit
                # So, check for a minimum temperature range too:
                if max(d['temp'])-min(d['temp']) < s.min_b_temp_span: # 21; switched to 10 for SPB3 # in mK
                    continue
                if str(s.mux_map[sb][ch]['TES_freq']).isnumeric() and \
                   int(s.mux_map[sb][ch]['TES_freq']) in s.g_guess_by_freq.keys():
                    g_guess = s.g_guess_by_freq[int(s.mux_map[sb][ch]['TES_freq'])]
                else:
                    g_guess = 0.130
                k_guess = g_guess/(n_guess*tc_guess**(n_guess-1))
                try:
                    if use_k:
                        parameters = curve_fit(
                            s.p_satofT,
                            np.array(d['temp']),
                            np.asarray(d['p_b90']), # really p_bias @ p_tes=p_sat
                            p0 = [k_guess, tc_guess, n_guess])
                    else:
                        parameters = curve_fit(
                            s.p_satofTandG,
                            np.array(d['temp']),
                            np.asarray(d['p_b90']),  # really p_bias @ p_tes=p_sat
                            p0 = [g_guess, tc_guess, n_guess])
                except RuntimeError:
                    continue
                cov = parameters[1]
                # a fit exists!
                # NOW we're convinced this is real, update tes_dict if it doesn't have it:
                check_tes_dict_for_channel_and_init(tes_dict, sb, ch)
                if use_k:
                    k, tc, n = parameters[0]
                    k_err, tc_err, n_err = np.sqrt(np.diag(parameters[1]))
                else:
                    g, tc, n = parameters[0]
                    g_err, tc_err, n_err = np.sqrt(np.diag(parameters[1]))
                # The above is...not exactly enough, as covariances likely exist. 
                # Oh yeah. Everything but tc has larger covariances with another variable than
                # its own variance. tc has one (with n) that's close in magnitude (though negative). 
                # MUST NOT FORGET COVARIANCE IS LARGE
                if use_k:
                    g = n*k*(tc**(n-1))
                    # Jacobian of G = [[∂G/∂k, ∂G/∂tc, ∂G/∂n]] evaluated at k,tc,n=parameters[0]
                    jac_G = np.array([[n*tc**(n-1), 
                                       n*(n-1)*k*tc**(n-2), 
                                       k*tc**(n-1)+n*k*np.log(tc)*tc**(n-1)]])
                    g_err = np.sqrt(\
                        np.matmul(jac_G, np.matmul(parameters[1],
                                                   np.transpose(jac_G)))[0][0])
                else:
                    k= g / (n*tc**(n-1))
                    # Jacobian of k = [[∂k/∂G, ∂k/∂tc, ∂k/∂n]] evaluated at G,tc,n=parameters[0]
                    jac_k = np.array([[1/(n*tc**(n-1)),
                                       g*(1-n)/(n*tc**n),
                                       -g*(1/(n**2*tc**(n-1))+np.log(tc)/(n*tc**(n-1)))]])
                    k_err = np.sqrt(\
                        np.matmul(jac_k,np.matmul(parameters[1],
                                                  np.transpose(jac_k)))[0][0])
                # Jacobian of p_sat100mk = [[∂p_sat/∂k, ∂p_sat/∂tc, ∂p_sat/∂n]] evaluated at T_b=100mK and k,tc,n=parameters[0]
                tb = 100
                if use_k:
                    jac_p_sat = np.array([[tc**n-tb**n, 
                                           n*k*tc**(n-1),
                                           k*(np.log(tc)*tc**n-np.log(tb)*tb**n)]])
                    p_sat100mK_err = np.sqrt(\
                        np.matmul(jac_p_sat,np.matmul(parameters[1],
                                                      np.transpose(jac_p_sat)))[0][0])
                else:
                    jac_p_sat = np.array([[(tc-tb**n/tc**(n-1))/n, 
                                           (g/n)*(1-(1-n)*tb**n/tc**n),
                                           tc*g*(-1/n**2-(1/n)*np.log(tb/tc)*(tb/tc)**n+(1/n**2)*(tb/tc)**n)]])
                    p_sat100mK_err = np.sqrt(\
                        np.matmul(jac_p_sat,np.matmul(parameters[1],
                                                      np.transpose(jac_p_sat)))[0][0]) 
                
                # update TES dict.
                s.add_thermal_param_to_dict(tes_dict[sb][ch],use_k,k,g,tc,n,cov,p_sat100mK_err)
                # not sure why I made these separate, since cov already gets em
                tes_dict[sb][ch]['k_err'] = k_err
                tes_dict[sb][ch]['G_err'] = g_err
                tes_dict[sb][ch]['tc_err'] = tc_err
                tes_dict[sb][ch]['n_err'] = n_err
                                                   
                # update internal results
                if sb not in s.all_results.keys():
                    s.all_results[sb] = {}
                s.all_results[sb][ch] = {}
                s.add_thermal_param_to_dict(s.all_results[sb][ch],use_k,k,g,tc,n,cov,p_sat100mK_err)
                
                # Don't include this one in the analysis.
                if mux_map[sb][ch]['opt'] in opt_values_exclude:
                    continue

                # Should really change this to tes_dict, include everything (no cuts above)
                # and make to_plots here. 
                # Should I change to use same array org. as mux_map? makes two-tier fors...
                # but easier-to-remember access, no re's. Add 'name' if added. 
                #k, tc, n = parameters[0]
                if sb not in results.keys():
                    results[sb]={}
                results[sb][ch] = {}
                s.add_thermal_param_to_dict(results[sb][ch],use_k,k,g,tc,n,cov,p_sat100mK_err)
        s.results  = results  
        
        # CONTINUING ON TO PREP FOR PLOTTING
        to_plot = {'k':[],'G':[],'tc':[],'n':[],'p_sat100mK':[]}
        for sb in results.keys():
            for ch, d in results[sb].items():
                for param, val in d.items():
#                     if param not in to_plot.keys():
#                         to_plot[param] = []
                    if param in to_plot.keys():
                        to_plot[param].append(val)
        s.to_plot = to_plot
        # to_plot_TES_freqs = {TES_freq: {'k':[],'tc':[],'n':[],'G':[],'p_sat100mK':[]}}
        to_plot_TES_freqs = {}
        for sb in results.keys():
            for ch, d in results[sb].items():
#             match = re.search('sb(\d)+_ch(\d+)',sbch)
#             sb, ch = int(match.group(1)), int(match.group(2))
                if tes_dict[sb][ch]['TES_freq'] not in to_plot_TES_freqs.keys():
                    to_plot_TES_freqs[tes_dict[sb][ch]['TES_freq']] = {'k':[],'G':[],'tc':[],'n':[],'p_sat100mK':[]}
                for param, val in d.items():
                    if param in to_plot_TES_freqs[tes_dict[sb][ch]['TES_freq']].keys():
                        to_plot_TES_freqs[tes_dict[sb][ch]['TES_freq']][param].append(val)
        s.to_plot_TES_freqs = to_plot_TES_freqs
            
    def add_thermal_param_to_dict(s,dicty,use_k,k,g,tc,n,cov,p_sat100mK_err):
        dicty['k'] = k
        dicty['G'] = g
        dicty['tc'] = tc
        dicty['n'] = n
        if use_k:
            dicty['cov_k_tc_n'] = cov
        else:
            dicty['cov_G_tc_n'] = cov
        dicty['p_sat100mK'] = s.p_satofTandG(100,g, tc, n)
        dicty['p_sat100mK_err'] = p_sat100mK_err
        
    # =================== Fitting functions
    def p_satofT(s,Tb, k, Tc, n):
        return k * (Tc**n - Tb**n)

    def p_satofTandG(s,Tb, G, Tc, n):
        return (G/n) * (Tc - Tb**n/Tc**(n-1))
    
    # ==================== plotting functions
    def make_summary_plots(s, nBins=60,split_freq=True,
                           p_sat_Range=None,t_c_Range=None,g_Range=None, n_Range=None):
        to_plot = s.to_plot
        to_plot_TES_freqs = s.to_plot_TES_freqs
        if not split_freq:
            p_sats, t_cs, gs, ns = [to_plot['p_sat100mK']], [to_plot['tc']],[[1000*g for g in to_plot['G']]], [to_plot['n']]
            tes_freqs_labels = ["all freqs"]
        else: # to_plot_TES_freqs = {TES_freq: {'k':[],'tc':[],'n':[],'G':[],'p_sat100mK':[]}}
            tes_freqs_labels = [str(key) + " GHz" for key in to_plot_TES_freqs.keys()]
            p_sats = [to_plot_TES_freqs[key]['p_sat100mK'] for key in to_plot_TES_freqs.keys()]
            t_cs = [to_plot_TES_freqs[key]['tc'] for key in to_plot_TES_freqs.keys()]
            gs = [[1000*g for g in to_plot_TES_freqs[key]['G']] for key in to_plot_TES_freqs.keys()]
            ns = [to_plot_TES_freqs[key]['n'] for key in to_plot_TES_freqs.keys()]
        
        if not p_sat_Range:
            p_sat_Range = (min(to_plot['p_sat100mK']), max(to_plot['p_sat100mK']))  
        if not t_c_Range:
            t_c_Range = (min(to_plot['tc']), max(to_plot['tc'])) 
        if not g_Range:
            g_Range = (1000*min(to_plot['G']), 1000*max(to_plot['G'])) 
        if not n_Range:
            n_Range = (min(to_plot['n']), max(to_plot['n']))

        std_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

        fig, ax = plt.subplots(nrows=4, figsize=(9,9))
        
        '''alpha argument to .hist is transparency, which is unnecessary here because only plotting one dataset
        on that histogram. Potentially useful if I separate by frequency, however.'''

        #h = ax[0].hist(p_sats,  alpha=0.4, bins=nBins, label=tes_freqs_labels, rwidth=1.0) # range=(0,12)
        for i in range(len(tes_freqs_labels)):
            h = ax[0].hist(p_sats[i],  alpha=0.4, bins=nBins, range=p_sat_Range, label=tes_freqs_labels[i]) # range=(0,12)
            med = np.nanmedian(p_sats[i])
            ax[0].axvline(med, linestyle='--',color=std_colors[i],label='median={:.1f} pW'.format(med))
        ax[0].set_xlabel('p_sat at 100 mK [pW]')
        ax[0].set_ylabel('# of TESs')
        ax[0].set_title(' ')

        #h = ax[1].hist(t_cs,  alpha=0.4, bins=nBins, label=tes_freqs_labels ) #range=(100,210) #to_plot['tc'], range=(175,205), alpha=0.4, bins=30
        for i in range(len(tes_freqs_labels)):
            h = ax[1].hist(t_cs[i],  alpha=0.4, bins=nBins, range=t_c_Range, label=tes_freqs_labels[i] )
            med = np.nanmedian(t_cs[i])
            ax[1].axvline(med, linestyle='--',color=std_colors[i],label='median={:.0f} mK'.format(med))
        # med = np.nanmedian(to_plot['tc'])
        # ax[1].axvline(med, linestyle='--',label='median=%.3fmK'%med)
        ax[1].set_xlabel('Tc [mK]')
        ax[1].set_ylabel('# of TESs')
        ax[1].set_title(' ')

        #h = ax[2].hist(gs, alpha=0.4, bins=nBins, label=tes_freqs_labels) #to_plot['G'], alpha=0.4, range=(30,270), bins=30
        for i in range(len(tes_freqs_labels)):
            h = ax[2].hist(gs[i], alpha=0.4, bins=nBins, range=g_Range, label=tes_freqs_labels[i])
            med = np.nanmedian(gs[i])
            ax[2].axvline(med, linestyle='--',color=std_colors[i],label='median={:.0f} pW/K'.format(med))
        ax[2].set_xlabel('G [pW/K]')
        ax[2].set_ylabel('# of TESs')
        ax[2].set_title(' ')

        #h = ax[3].hist(ns,  alpha=0.4, bins=nBins, label=tes_freqs_labels) #range=(0,5)
        for i in range(len(tes_freqs_labels)):
            h = ax[3].hist(ns[i],  alpha=0.4, bins=nBins, range=n_Range, label=tes_freqs_labels[i])
            med = np.nanmedian(ns[i])
            ax[3].axvline(med, linestyle='--',color=std_colors[i],label='median={:.1f}'.format(med))
        ax[3].set_xlabel('n')
        ax[3].set_ylabel('# of TESs')
        ax[3].set_title(' ')

        for ind in [0,1,2,3]:
            ax[ind].legend(fontsize='small') #, loc=2

        totNumDets = sum([len(dataset) for dataset in t_cs])
        numDets = "(numFit = "+ str(totNumDets) + "; "+ ", ".join([str(len(t_cs[i])) + "x"+ tes_freqs_labels[i][:-4] for i in range(len(tes_freqs_labels))]) + ")"

        # Maybe add something to note if unmasked excluded?
        plt.suptitle(f"{s.dName} " + numDets, fontsize=16)
        plt.tight_layout()
        return ax




# -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
# =============== Coldload_Ramp Class (Temp_Ramp child class) ==================
# -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~

class Coldload_Ramp(Temp_Ramp):
    '''
    This child class deals with fitting and plotting results of 
    coldload Temp_Ramps. You should already have run a Bath_Ramp 
    on the tes_dict you pass to Coldload_Ramp.
    
    # ONLY WORKS FOR SPB THUS FAR. 
    
    Adds the following to its passed tes_dict's channel dictionaries:
    - TODO
    
    DEPRECATED, no longer does this: Adds the following to super().ramp, per channel: 
    'bath_temp': [] and 'bath_temp_err': []
    # ^ estimated. Not really using those now though.  
   
    =========================== (Child) VARIABLES ============================
    # ---- REQUIRED Child init var:
    bath_ramp   # a Bath_Ramp() that this Coldload_Ramp() references.
    # ---- Optional Child init vars (and defaults):
    do_opt      # = False
    device_z    # = -42   # (not provided)
    opt_power   # = {}    # loads if given, or constructs if not (& do_opt)
    br_cl_temp  # = -42   # The cold load temperature during the bath ramp. 
    spb_light_leak_sim=0, # adds horn's opt_power * this to masked opts
    # these below 3 are for me exploring what could cause T_b_est issue
    scale_G     # = 1.0   
    scale_n     # = 1.0
    add_tc      # = 0
    
    # --- Borrowed vars; gave this class direct references b/c used frequently
    tes_dict    # same as the test_device's.
    mux_map     # same as the test_device's. 
    
    # ---- Constructed Vars:
    opt_power   # {(det_x,det_y):{'load_obj':a spb_opt.SPBOptLd, <TES_freq (int)>:{
                #      'opt_power': [in pW, for each cl_temp], 
                #      'opt_fit': ([<slope>, <constant>],covariance matrix) 
                #      'opt_power_during_bath_ramp': pW}   }}
    # 'results' exists to not have to check the big ramp arrays for if ch was fit
    # and has a tes_dict entry. It also summarizes results.
    results     # {sb:{ch:{  (see below)
        # All: (Note:  p_b90 = p_bias at saturation (90% R_n))
            # 'cl_temp': [], 'p_b90_fit': ([<slope>, <constant>],covariance matrix),
        # If masked/dark: 
            # 't_b_est_thermal':{'t_bath':[],'t_bath_error':[]}
            # 'bath_temp_fit' = (popt, pcov) # linear fit
        # If opt unmasked: 
            # 'no_subtraction_opt_eff', and 
            # 'loc' = (det_x, det_y) # from mux_map b/c otherwise unwieldy.
            # 't_b_est':{'t_bath':[],'t_bath_error':[]} # used in together fit; 
                # }}
    slopes      # Slopes of fitlines of the tb_est_thermal for each masked/dark
    
    # ======================= METHODS (Helpers indented) =======================
    # ---------- Initialization and loading methods
    __init__(s, test_device, ramp_type, therm_cal, 
             metadata_fp_arr, bath_ramp, norm_correct=True, p_sat_cut=15, 
             use_p_satSM=True, fix_pysmurf_idx=False,input_file_type="pysmurf",
             cooldown_name='',
             do_opt=False, device_z=-42, opt_power={}, br_cl_temp=-42,
             spb_light_leak_sim=0, # adds horn's opt_power * this to masked opts
             scale_G=1.0, scale_n=1.0,add_tc=0)
        init_opt_power(s,device_z, cooldown_name, freq,loc)

    # ---------- Fitting Methods
    together_fit_opt_unmasked(s,tb_fit,sb,ch,make_plot=False, own_fig=True)
    # --- scipy.optimize.curvefit() fit functions (helpers of multiple funcs)
        tbath_of_dark_TES(s,P_b, g,T_c,n)
        tbath_line(s,tcl, slope, constant)
        fit_line(s, x, slope, constant)
        coldload_bathramp_opt_unmasked_p_b90_of_param(s,temp_arr, G, tc, n, 
                                                      opt_eff)
            point_coldload_bathramp_opt_unmasked_p_b90_of_param(s,temp, G, tc, n,
                                                                opt_eff)

    # ---------- Plotting Methods
    plot_est_T_bath(s)
    plot_est_T_bath_by_BL(s,tes_dict=True)
    plot_T_b_slopes_vs_param(s,param_name)
    '''
    
    def __init__(s, test_device, ramp_type, therm_cal, 
                 metadata_fp_arr, bath_ramp, norm_correct=True, p_sat_cut=15, 
                 use_p_satSM=True, fix_pysmurf_idx=False,input_file_type="pysmurf",
                 cooldown_name='',
                 do_opt=False, device_z=-42, opt_power={}, br_cl_temp=-42,
                 spb_light_leak_sim=0, # adds horn's opt_power * this to masked opts
                 scale_G=1.0, scale_n=1.0,add_tc=0): 
        # SHOULD be opt_values_exclude, not bias lines
        super().__init__(test_device, ramp_type, therm_cal, metadata_fp_arr,
                         norm_correct=norm_correct,p_sat_cut=p_sat_cut,
                        use_p_satSM=use_p_satSM,fix_pysmurf_idx=fix_pysmurf_idx,
                        input_file_type=input_file_type)
        # Now that the Temp_Ramp is set up...set up results, calculate universal stuff.
        tes_dict = s.tes_dict
        mux_map = s.mux_map
        s.bath_ramp = bath_ramp
        s.br_cl_temp = br_cl_temp 
        #s.spb_light_leak_sim = spb_light_leak_sim
        
        # ========= Setting up s.results: which dets can we work with
        s.results = {}  
        for sb in s.ramp.keys():
            for ch, d in s.ramp[sb].items(): # below: need optical classification to do anything here
                if sb not in tes_dict.keys() or ch not in tes_dict[sb].keys():
                    continue
                if not is_float(tes_dict[sb][ch]['opt']):
                    continue # again, need that optical classification to do anything here.
                if not len(d['temp']) >=4: # Gotta have data to do analysis.
                    continue
                if sb not in s.results.keys():
                    s.results[sb] = {}
                if ch not in s.results[sb].keys():
                    s.results[sb][ch] = {}
                s.results[sb][ch]['cl_temp'] = s.ramp[sb][ch]['temp']
    
        # ========= Calculating/importing optical load =================
        # ONLY WORKS FOR SPB CURRENTLY.
        # You have to follow the setup explained in SPB_opt/Get_optical_power.ipynb first
        # Calculating optical power takes a long time,
        # Many detectors are under the same horn center.
        # so we do it by horn center location and call that dictionary when we want it.
        # Also, we provide an option to import a previously-established opt_power
        if do_opt:
            if opt_power: # did they import one?
                s.opt_power = opt_power
                do_calc=False
            else:# gotta do it ourself.
                s.opt_power = {}
                do_calc = True
            for sb in s.results.keys():
                for ch, d in s.results[sb].items():
                    if tes_dict[sb][ch]['opt'] == 1 \
                      and is_float(tes_dict[sb][ch]['TES_freq'])  \
                      and len(s.ramp[sb][ch]['temp'])>2:
                        loc = (mux_map[sb][ch]['det_x'],
                               mux_map[sb][ch]['det_y'])
                        s.results[sb][ch]['loc'] = loc
                        freq = int(float(tes_dict[sb][ch]['TES_freq']))
                        if do_calc:
                            # Below had excessive line length, so made it a function
                            s.init_opt_power(device_z,
                                             cooldown_name, 
                                             freq,loc)
                        s.opt_power[loc][freq]['opt_power_during_bath_ramp'] = \
                            s.opt_power[loc]['load_obj'].get_power(T=s.br_cl_temp,freq=freq)
        
        
        # ========== Fit p_b90s as lines.  
        # ==== And Adjust opt masked p_b90s if spb_light_leak_sim > 0
        for sb in s.results.keys():
            for ch in s.results[sb].keys():
                if do_opt and spb_light_leak_sim > 0 \
                and tes_dict[sb][ch]['opt'] == 0.25: # optical masked
                    # only one horn, this is spb-only
                    for key in s.opt_power.keys():
                        loc=key
                    opt_powers_to_add = s.opt_power[loc][int(s.tes_dict[sb][ch]['TES_freq'])]['opt_power']
                    # this...actually WON'T  fail if more than one ramp_raw
                    for i in range(len(s.ramp[sb][ch]['temp'])):
                        j = s.temp_list_raw.index(s.ramp[sb][ch]['temp'][i])
                        s.ramp[sb][ch]['p_b90'][i] += spb_light_leak_sim*opt_powers_to_add[j] 
                    #print(s.ramp[sb][ch]['p_b90'])
                (popt, pcov) = curve_fit(s.fit_line, s.results[sb][ch]['cl_temp'], 
                                         s.ramp[sb][ch]['p_b90'])
                s.results[sb][ch]['p_b90_fit'] = (popt, pcov)
                # That isn't really a line! Look at mv6 to see clearly. 
            # should I do this:
            # 'opt', 'opt_name' # stolen from tes_dict because dang this uses them a lot. 
            
        
        # ========== Opt power w/out dark subtraction (if passed opt_power)
        # Assuming that T_b stays the same... 
        # η = -(d[P_bas_unmasked]/d[T_CL]) / (d[P_opt]/d[T_CL])
        if do_opt:
            for sb in s.results.keys():
                for ch, d in s.results[sb].items():
                    # Tes_dict can contain ones with only one point with Kaiwen's stuff..
                    if not 'loc' in d.keys(): # is optical unmasked, has sensible TES_freq, at least 3 points to fit
                        continue
                    my_slope = d['p_b90_fit'][0][0]
                    my_freq = int(float(tes_dict[sb][ch]['TES_freq']))
                    my_opt_slope = s.opt_power[d['loc']][my_freq]['opt_fit'][0][0]
                    s.results[sb][ch]['no_subtraction_opt_eff'] = -my_slope/my_opt_slope

        # optical power ignoring n differences
        # (should import from outside function?)   
        

        # ==================== T_b_est
        # let's check T_b, and look at the
        # slope of the resulting lines.
        s.slopes=[]
        br_r = s.bath_ramp.all_results
        for sb in s.ramp.keys():
            for ch, d in s.ramp[sb].items():
                if sb not in s.tes_dict.keys() or ch not in s.tes_dict[sb].keys():
                    continue # should I be iterating over tes_dict instead?
                if sb not in br_r.keys() or ch not in br_r[sb].keys():
                    continue
                # Tes_dict can contain ones with only one point with Kaiwen's stuff..
                if not s.tes_dict[sb][ch]['opt'] == 1.0 and len(d['temp']) >=4:   
                    # ^ not unmasked optical fab, at least 4 points
                    if sb not in s.results.keys():
                        s.results[sb] = {}
                    if ch not in s.results[sb].keys():
                        s.results[sb][ch] = {}
                    s.results[sb][ch]['cl_temp'] = s.ramp[sb][ch]['temp']
                    s.results[sb][ch]['t_b_est_thermal']={'t_bath':[],'t_bath_err':[]}
                    
                    g, tc, n, cov = (scale_G*br_r[sb][ch]['G'], add_tc+br_r[sb][ch]['tc'], \
                               scale_n*br_r[sb][ch]['n'], br_r[sb][ch]['cov_G_tc_n'])
                    #print(f'{ch},{g},{tc},{n},{cov}')
                    for i in range(len(d['temp'])):
                        pb = d['p_b90'][i]
                        bt = s.tbath_of_dark_TES(pb, g,tc,n)
                        s.results[sb][ch]['t_b_est_thermal']['t_bath'].append(bt)
                        # Now, let's get the error. 
                        # I checked my derivative with mathematica, so only errors should be typos...
                        # Double checked resultant numbers with mathematica too. 
                        bttn = bt**n 
                        jac_tb = np.array([[
                            (1/n)*(bttn)**(1/n-1) * (n*pb*tc**(n-1)/g**2),
                            (1/n)*(bttn)**(1/n-1) * (n*tc**(n-1)-n*(n-1)*pb*tc**(n-2)/g), 
                            bttn**(1/n)*(-np.log(tc**n-n*pb*tc**(n-1)/g)/n**2 + \
                                         (np.log(tc)*tc**n - (pb/g)*(tc**(n-1)+n*np.log(tc)*tc**(n-1))) \
                                         /(n*(tc**n-n*pb*tc**(n-1)/g)))]])
                        #print(jac_tb)
                        bt_err = \
                           np.sqrt(np.matmul(jac_tb,
                                             np.matmul(cov,
                                                       np.transpose(jac_tb)))[0][0])
                        s.results[sb][ch]['t_b_est_thermal']['t_bath_err'].append(bt_err)
                        #print([float(f'{val:.3f}') for val in [d['temp'][i],pb,bt,bt_err]])
                    
                    try:
                        (popt, pcov) = curve_fit(s.tbath_line, 
                                                 s.results[sb][ch]['cl_temp'], 
                                                 s.results[sb][ch]['t_b_est_thermal']['t_bath'],
                                                 sigma=s.results[sb][ch]['t_b_est_thermal']['t_bath_err'],
                                                 absolute_sigma=True)
                        s.results[sb][ch]['bath_temp_fit'] = (popt, pcov)
                        s.slopes.append(popt[0])
                    except:
                        print(f"{sb} {ch} t_b_est: {s.results[sb][ch]['t_b_est_thermal']['t_bath']} t_b_est_error:{s.results[sb][ch]['t_b_est_thermal']['t_bath_err']}")
            
                    #for_disp = [float(f'{param:.2f}') for param in popt]
                    #for_disp = [float(f'{temp:.1f}') for temp in s.ramp[sb][ch]['t_b_est_thermal']['t_bath']]
                    #print(f"{for_disp};{ch};{tes_dict[sb][ch]['opt_name']}")
        print(f"!opt-unmasked t_bath_est slopes: average{np.mean(s.slopes)},stdev{np.std(s.slopes)}")
        
        
    # ======== more init functions =========== 
    def init_opt_power(s,device_z, cooldown_name, freq,loc):
        if loc not in s.opt_power.keys():
            s.opt_power[loc] = {}
            beam_name = ''
            # I believe MF/UHF TES are never mixed on a single horn. 
            if 65<=freq and freq<=190: # expecting 90 or 150
                beam_name = 'MF_F'
            elif 190<=freq:
                print("update UHF freqs/beam names! Check center vs. name")
                beam_name = 'UHF_F' # a wild guess
            # this is the time consuming step. Hence doing it by loc.
            s.opt_power[loc]['load_obj'] = spb_opt.SPBOptLd(cooldown_name,
                                        loc[0],loc[1],device_z,
                                        beam_name=beam_name)
        if freq  not in s.opt_power[loc].keys():
            s.opt_power[loc][freq] = {'opt_power':[]}
            for temp in s.temp_list:
                # names of UHF freqs != beam center. May need to update this
                power = s.opt_power[loc]['load_obj'].get_power(T=temp,freq=freq)
                s.opt_power[loc][freq]['opt_power'].append(power)
            (popt, pcov) = curve_fit(s.fit_line, s.temp_list, 
                                     s.opt_power[loc][freq]['opt_power'])
            s.opt_power[loc][freq]['opt_fit'] = (popt,pcov)
                        
    #def bathramp_coldload_together_fits(s, tes_dict, ) 
    
    
    # ======== Fitting functions ======  
    def together_fit_opt_unmasked(s,tb_fit,sb,ch,make_plot=False, own_fig=True):
        # does not update tes_dict (or anything!) with the new values....
        # also need to have coldload power during bath_ramp
        tes_dict = s.tes_dict
        # set up the given t_b est line:
        ma, mb = tb_fit[0],tb_fit[1]
        s.results[sb][ch]['t_b_est']={'t_bath':[],'t_bath_err':[]}
        for temp in s.results[sb][ch]['cl_temp']:
            s.results[sb][ch]['t_b_est']['t_bath'].append(ma*temp +mb )
        # setup s.cur_det and run scipy.optimize.curvefit 
        # setup the function to be optimized
        bath_ramp = s.bath_ramp
        loc = s.results[sb][ch]['loc']
        s.cur_det = {}
        freq = int(tes_dict[sb][ch]['TES_freq'])
        s.cur_det['opt_power'] = s.opt_power[loc][freq]['opt_power']
        s.cur_det['cl_temp'] = s.results[sb][ch]['cl_temp']
        s.cur_det['t_b_cl_ramp'] = s.results[sb][ch]['t_b_est']['t_bath']
        # Should think about what should own this value. Note different for different locs.
        s.cur_det['cl_power_during_bath_ramp'] = s.opt_power[loc][freq]['opt_power_during_bath_ramp']
        # now, setup x and y data:
        x_data = s.results[sb][ch]['cl_temp'] + bath_ramp.ramp[sb][ch]['temp']
        y_data = s.ramp[sb][ch]['p_b90'] + bath_ramp.ramp[sb][ch]['p_b90']
        # now, parameter estimates:
        p0 = [bath_ramp.all_results[sb][ch]['G'],
              bath_ramp.all_results[sb][ch]['tc'],
              bath_ramp.all_results[sb][ch]['n'],
              0.5] # opt_eff
        # here we go:     
        (popt, pcov) = curve_fit(s.coldload_bathramp_opt_unmasked_p_b90_of_param,
                                              x_data,y_data, p0=p0)
        # this should be moved to plotting functions once this saves data. 
        if make_plot:
            if own_fig:
                plt.figure(figsize=default_figsize)
            g, tc, n, opt_eff = popt
            offscreen = s.bath_ramp.temp_list_raw[-1]*1.05 # let you see x-intercept
            together_fit_y = s.coldload_bathramp_opt_unmasked_p_b90_of_param(\
                                    x_data + [offscreen], g, tc, n, opt_eff)
            plt.plot(x_data+[offscreen], together_fit_y,label="together_fit")
            d = bath_ramp.results[sb][ch]
            g,tc,n,opt_eff = d['G'],d['tc'],d['n'],0
            bathramp_fit_y = s.coldload_bathramp_opt_unmasked_p_b90_of_param(\
                              bath_ramp.ramp[sb][ch]['temp']+[offscreen], g, tc, n, opt_eff)
            plt.plot(bath_ramp.ramp[sb][ch]['temp']+[offscreen],bathramp_fit_y, \
                     label="bathramp_fit", linewidth=1)
            plt.plot(x_data,y_data,marker=".",linestyle="None", label="measured p_b90")
            if own_fig:
                plt.title(f"sb{sb} ch{ch} together fit")
                plt.ylim(0,4.0/5*s.p_sat_cut)
                plt.legend()
        return (popt,pcov)
        
    # -------- for scipy.optimize.curvefit() ;
    def tbath_of_dark_TES(s,P_b, g,T_c,n):
        return (T_c**n-P_b*n*T_c**(n-1)/g)**(1/n)

    def tbath_line(s,tcl, slope, constant):
        return slope*tcl + constant
    
    def fit_line(s, x, slope, constant):
        return slope*x + constant
    
    # and here is where I discovered that curve_fit passes x_data as an array.
    def coldload_bathramp_opt_unmasked_p_b90_of_param(s,temp_arr, G, tc, n, opt_eff):
        # first, the function to be fit. 
        # We have to pass some det-specific arguments into this by saving 
        # them as class variables, so curve_fit works
        # these next 3 arrays size all = # of coldload ramp p_b90 points for that det
        # s.cur_det = {'opt_power':[],'cl_temp':[],'t_b_cl_ramp':[]
        #         'cl_power_during_bath_ramp'} 
        # just in case scipy later passes individual values:
        if is_float(temp_arr):
            val = s.point_coldload_bathramp_opt_unmasked_p_b90_of_param(temp_arr, G, tc, n, opt_eff)
            return val
        to_return = []
        for temp in temp_arr:
            val = s.point_coldload_bathramp_opt_unmasked_p_b90_of_param(temp, G, tc, n, opt_eff)
            to_return.append(val)
        return np.array(to_return) 
    
    # -------- fitting methods not directly for scipy.optimize.curvefit()
    def point_coldload_bathramp_opt_unmasked_p_b90_of_param(s,temp, G, tc, n, opt_eff):
        # first, the function to be fit. 
        # We have to pass some det-specific arguments into this by saving 
        # them as class variables, so curve_fit works
        # these next 3 arrays size all = # of coldload ramp p_b90 points for that det
        # s.cur_det = {'opt_power':[],'cl_temp':[],'t_b_cl_ramp':[]
        #         'cl_power_during_bath_ramp'}        
        if temp < 50: # because temp's really in K, the cold load temp
            point_ind = s.cur_det['cl_temp'].index(temp)
            t_b_point = s.cur_det['t_b_cl_ramp'][point_ind]
            p_sat = G/n*(tc-t_b_point**n/tc**(n-1))
            p_opt = s.cur_det['opt_power'][point_ind]
        else: # this temp is a bathramp point
            p_sat = G/n*(tc-temp**n/tc**(n-1))
            p_opt = s.cur_det['cl_power_during_bath_ramp']
        return p_sat - opt_eff*p_opt
    
    # ============ Plotting methods =========
    def plot_est_T_bath(s):
        # possibly could call the Parent class plot ramp by param thing?
        # oh...not quite, because errorbar. 
        tes_dict = s.tes_dict
        plot_array = []
        for sb in s.results.keys():
            plot_array.append((plt.figure(figsize=default_figsize), sb))
            for ch, d in s.results[sb].items():
#                 if 'bath_temp' not in s.ramp[sb][ch].keys():
#                     continue
                if not tes_dict[sb][ch]['opt'] == 1.0:
                    mylabel = f'ch{ch} {tes_dict[sb][ch]["TES_freq"]} {tes_dict[sb][ch]["opt_name"]}'
                    plt.errorbar(d['cl_temp'], d['t_b_est_thermal']['t_bath'], 
                                 yerr=d['t_b_est_thermal']['t_bath_err'],
                                 linestyle=tes_dict[sb][ch]["linestyle"],
                                 label=mylabel)

            #plt.ylim(0, 15)
            plt.ylabel("estimated bath temp [mK]")
            plt.xlabel("cold load temperature [K]")
            plt.title(f"estimated bath temp from bath_ramp and CLramp data, sb{sb}")
            plt.legend()
        return plot_array
    
    def plot_est_T_bath_by_BL(s,tes_dict=True):
        if tes_dict:
            tes_dict = s.tes_dict
        mux_map = s.mux_map
        linestyle,label =  'solid', ''
        #fig, ax = plt.subplots(nrows=4, figsize=(9,9))
        fig, ax = plt.subplots(nrows=4, ncols=3, figsize=(9,9))
        for r in range(len(ax)):
            row = ax[r]
            for c in range(len(row)):
                p = row[c]
                for sb in s.results.keys():
                    for ch, d in s.results[sb].items():
                        #print(mux_map[sb][ch])
                        if mux_map[sb][ch]['biasline'] == 3*r + c:
                            if tes_dict and sb in tes_dict.keys() and ch in tes_dict[sb].keys():
                                linestyle = tes_dict[sb][ch]['linestyle']
                                label = f' {str(tes_dict[sb][ch]["TES_freq"])} {tes_dict[sb][ch]["opt_name"]}'
                            if not tes_dict[sb][ch]['opt'] == 1.0:
                                p.errorbar(d['cl_temp'], d['t_b_est_thermal']['t_bath'], 
                                     yerr=d['t_b_est_thermal']['t_bath_err'],
                                     linestyle=linestyle,
                                     label=f'ch{ch}'+label)
                #p.set_xlabel('bath temp [mK]')
                #p.set_ylabel('90% R_n P_sat [pW]')
                p.set_title("Bias line " + str(3*r + c))
                #p.set_ylim([0, 8])
                p.set_xlim([0.95*s.temp_list[0], s.temp_list[-1] + (0.05*s.temp_list[0])])
                #p.legend(fontsize='small')
        plt.suptitle(f"{s.dName} estimated bath temp [mK] from bath_ramp and CLramp data vs. CL temp [K]")
        plt.tight_layout()
        return (fig, ax)
    
    def plot_T_b_slopes_vs_param(s,param_name):
        # Should really fix this to use test_device.opt_dict.
        tes_dict = s.tes_dict
        series = {'masked 90 opt': ([],[]),
                  'masked 150 opt': ([],[]),
                  'masked dark': ([],[]),
                  'unmasked dark': ([],[])} # param, slopes []
        for sb in s.results.keys():
            for ch, d in s.results[sb].items():
                if is_float(tes_dict[sb][ch]['opt']):
                    s_key = ''
                    if tes_dict[sb][ch]['opt'] == 0.75:
                        s_key = 'unmasked dark'
                    elif tes_dict[sb][ch]['opt'] == 0.25: 
                        s_key = f"masked {str(tes_dict[sb][ch]['TES_freq'])} opt"
                    elif tes_dict[sb][ch]['opt'] == 0: 
                        s_key = 'masked dark'
                    if s_key:
                        series[s_key][0].append(tes_dict[sb][ch][param_name])
                        series[s_key][1].append(s.results[sb][ch]['bath_temp_fit'][0][0])
        p = plt.figure(figsize=default_figsize)     
        for key in series:
            plt.plot(series[key][0], series[key][1], marker=".", linestyle="None", label=key)
        
        #plt.plot(unmasked_param, unmasked_slopes, marker=".", linestyle="None", label="dark fab TESs in unmasked area")
        plt.ylabel("slope of fit line of T_b estimate [mK/K]")
        plt.xlabel(f"{param_name}")
        plt.title(f"{s.dName} T_b_est fit line slopes vs. {param_name}")
        plt.legend()
        return p
   




