import matplotlib
matplotlib.use('Agg')

import pysmurf.client
import argparse
import numpy as np
import time
import sys
import yaml

def health_check(S, config_file_name):
    """
    Performs a system health check. This includes checking/adjusting amplifier biases,
    checking timing, checking the jesd connection, and checking that noise can
    be seen through the system. 

    Inputs: A configuration file
    Output: An updated configuration file containing the original parameters
    along with health check parameters

    """
    #Load data from config yaml file
    with open(config_file_name, 'r') as stream:
        config = yaml.safe_load(stream) 
        
    bay = config["bay"]
    amp_hemt_Id = config["amp_hemt_Id"]
    amp_50K_Id = config["amp_50K_Id"]
    drive_power = config["drive_power"]
    up_attenuator = config["up_attenuator"]
    down_attenuator = config["down_attenuator"]

    
    #Check amplifier biases

    
    amp_biases = S.get_amplifier_biases()
    #Turns both amplifiers on if they aren't already
    config['hemt_biased'] = True
    config['50K_biased'] = True
    if np.abs(amp_biases['hemt_Id']) <0.2 or np.abs(amp_biases['50K_Id']) <0.2:
        S.C.write_ps_en(11)
        amp_biases = S.get_amplifier_biases()
        #Note whether they were turned on sucessfully
        if np.abs(amp_biases['hemt_Id']) <0.2:
            print("hemt amplifier could not be biased. Check for loose cable")
            config['hemt_biased'] = False          
        if np.abs(amp_biases['50K_Id']) <0.2:
            print("50K amplifier could not be biased. Check for loose cable")
            config['50K_biased'] = False           
    
    #If the hemt_Id isn't within 1mA of the desired value, adjust Vg
    iters = 0
    Vg_hemt = S.get_hemt_gate_voltage()
    Id_hemt_in_range = True
    while np.abs(amp_biases['hemt_Id']-amp_hemt_Id) > 0.5:
        delta_Id = amp_biases['hemt_Id']-amp_hemt_Id        
        #Vg step is set by the size of the offset
        if np.abs(delta_Id) > 1.5:
            Vg_step = 0.1
        else:
            Vg_step = 0.01
        
        #Increase Vg if Id is too high, and vice versa
        if amp_hemt_Id > amp_biases['hemt_Id']:
            S.set_hemt_gate_voltage(Vg_hemt + Vg_step)
            Vg_hemt = S.get_hemt_gate_voltage()
            
        else:    
            S.set_hemt_gate_voltage(Vg_hemt - Vg_step)
            Vg_hemt = S.get_hemt_gate_voltage()
        
        iters += 1
        print(f"Adjusting hemt_Id, iteration {iters}. Current hemt_Id ",
              f"= {S.get_hemt_drain_current()}, Vg_hemt={Vg_hemt}")
        if iters > 30:
            print("Could not get hemt_Id to acceptable range. Moving on")
            Id_hemt_in_range = False
            break
        
    config['Id_hemt_in_range'] = Id_hemt_in_range
    print("hemt_Id = {}".format(amp_biases["hemt_Id"]))

    #If the 50K_Id isn't within 1mA of the desired value, adjust Vg
    iters = 0
    hemt_50K = S.get_50K_gate_voltage()
    Id_50K_in_range = True
    while np.abs(amp_biases['50K_Id']-amp_50K_Id) > 0.5:
        delta_Id = amp_biases['50K_Id']-amp_50K_Id        
        #Vg step is set by the size of the offset
        if np.abs(delta_Id) > 1.5:
            Vg_step = 0.1
        else:
            Vg_step = 0.01
        
        #Increase Vg if Id is too high, and vice versa
        if amp_50K_Id > amp_biases['50K_Id']:
            S.set_50K_gate_voltage(Vg_50K + Vg_step)
            Vg_50K = S.get_50K_gate_voltage()
            
        else:    
            S.set_50K_gate_voltage(Vg_50K - Vg_step)
            Vg_50K = S.get_50K_gate_voltage()
        
        iters += 1
        print(f"Adjusting 50K_Id, iteration {iters}. Current 50K_Id ",
              f"= {S.get_50K_drain_current()}, Vg_50K={Vg_50K}")
        if iters > 30:
            print("Could not get 50K_Id to acceptable range. Moving on")
            Id_50K_in_range = False
            break
        
    config['Id_50K_in_range'] = Id_50K_in_range
    print("50K_Id = {}".format(amp_biases["50K_Id"]))


    #Check timing is active. Check with Shawn
    #Results in timing_active = True or False

    
    #Check JESD connection
    #Check what bay means with Shawn
    jesd_tx, jesd_rx = S.check_jesd(bay=2)
    config['jesd_tx'] = jesd_tx
    config['jesd_rx'] = jesd_rx
    if jesd_tx:
        print("jesd_tx connection working")
    else:
        print("ERROR: jesd_tx connection NOT working. Rest of script may not function")
    if jesd_rx:
        print("jesd_rx connection working")
    else:
        print("ERROR: jesd_rx connection NOT working. Rest of script may not function")
        

    # Full band response. This is a binary test to determine that things are plugged in.
    # Typical in-band noise values are around ~2-7, so here check that average value of     # noise through band 0 is above 1.

    
    freq, response = S.full_band_resp(band=0)
    #Get the response in-band
    resp_inband = []
    for f, r in zip(freq, response):
        if f > -2.5e8 and f < 2.5e8:
            resp_inband.append(r)
    #If the mean is > 1, say response received
    if np.mean(resp_inband) > 1:
        config['resp_check'] = True
        print("Full band response check passed").
    else:
        config['resp_check'] = False
        print("Full band response check failed - maybe something isn't plugged in?")

        
    #Check if ADC is clipping. Probably should be a different script, after
    # characterizing system to know what peak data amplitude to simulate
    #Should result in ADC_clipping = T/F
    #Iterate through lowest to highest band, stop when no clipping.
    #Find max value of output of S.read_adc_data(0), compare to pre-set threshold
    # Probably should have a 'good' 'warning', and 'failed' output

    
    #Output the updated config file    
    with open('health_check.yml', 'w') as outfile:
        yaml.dump(data, outfile)
    
    print("Updated config yaml file written. Health check finished")
    
    
