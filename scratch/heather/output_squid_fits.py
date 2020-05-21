# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import sys
import numpy as np
import matplotlib.pylab as plt
from datetime import datetime
import os as os
import lambdafit
import imp
import pandas as pd

#assumes that you have taken SQUID data using Shawn's code 

def fit_squid_data(self, datafile, band):
    #assumes one band and one power
    
    all_raw_data=np.load(datafile)
    all_raw_data=all_raw_data.item()
    
    
    power = list(all_raw_data[band].keys())[0]
    
    #assume there is only one power
    raw_data=all_raw_data[band][power]
    
    #a definition   
    def redchisq(ydata, ymod):
        chisq=np.sum((ydata-ymod)**2)  
        return chisq 
    
    #make your dataframe to store the fitted values
    d_squid = {'smurf_band':[],
        'smurf_chan':[],
        'chi2':[], 
        'I0':[],
        'm':[],
        'f2':[],
        'p2p':[], 
        'lamb':[], 
        'phi0_file':[]}
            
    d_squid = pd.DataFrame(d_squid)
    
    
    #now fit each curve
    bias=all_raw_data['bias']
    
    for chan in raw_data['channels']:
        chf=raw_data['fvsfr'][np.where(raw_data['channels']==chan)[0][0]]
        chan_params = lambdafit.lambdafit(bias, chf*1000)
        
        #get chi sq
        fit_chf = lambdafit.f0ofI(bias, chan_params[0], chan_params[1], chan_params[2], chan_params[3], chan_params[4])
        chisq = redchisq(chf*1000, fit_chf)
        
        #assign variable to make writing to df easier
        I0,m,f2,p2p,lamb = chan_params
        d_squid = d_squid.append({'smurf_band': band, 'smurf_chan':chan, 'chi2':chisq, 'I0':I0, 'm':m, 'f2':f2, 'p2p':p2p, 'lamb':lamb, 'phi0_file':datafile}, ignore_index=True)
    
    #and return the dataframe with all the fitted paramters
    return d_squid