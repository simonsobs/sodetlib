import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pysmurf.client
import argparse
import numpy as np
import time
import sys
import yaml

def health_check(S, bay0, bay1, amp_hemt_Id, amp_50K_Id):
	"""
	Performs a system health check. This includes checking/adjusting amplifier biases,
	checking timing, checking the jesd connection, and checking that noise can
	be seen through the system. 

	Parameters
	----------
	bay0 : bool
		Whether or not bay 0 is active
	bay1 : bool
		Whether or not bay 1 is active
	amp_hemt_Id : float
		Target drain current for the hemt amplifier
	amp_50K_Id : float
		Target drain current for the 50K amplifier

	Returns
	-------
	biased_hemt : bool
		Whether or not the hemt amplifier was successfully biased
	biased_50K : bool
		Whether or not the 50K amplifier was successfully biased
	Id_hemt_in_range : bool
		Whether or not the hemt current is within 0.5mA  of the desired value without going over
	Id_50K_in_range : bool
		Whether the 50K current is within 0.5mA of the desired value without going over
	Vg_hemt : float
		Final hemt gate voltage
	Vg_50K : float
		Final 50K gate voltage
	jesd_tx0 : bool
		Whether or not the jesd_tx connection on bay 0 is working
	jesd_rx0 : bool
		Whether or not the jesd_rx connection on bay 0 is working
	jesd_tx1 : bool
		Whether or not the jesd_tx connection on bay 1 is working
	jesd_rx1 : bool
		Whether or not the jesd_rx connection on bay 1 is working
	resp_check : bool
		Whether or not a response was seen after sending noise through band 0

	"""
	
	#Check amplifier biases
	
	amp_biases = S.get_amplifier_biases()
	#Turns both amplifiers on if they aren't already
	if np.abs(amp_biases['hemt_Id']) <0.2 or np.abs(amp_biases['50K_Id']) <0.2:
		S.C.write_ps_en(11)
		amp_biases = S.get_amplifier_biases()        
	#Note whether they were turned on sucessfully
	if np.abs(amp_biases['hemt_Id']) <0.2:
		print("hemt amplifier could not be biased. Check for loose cable")
		biased_hemt = False
	else:
		print("hemt amplifier successfully biased.")
		biased_hemt = True         
	if np.abs(amp_biases['50K_Id']) <0.2:
		print("50K amplifier could not be biased. Check for loose cable")
		biased_50K = False
	else:
		print("50K amplifier successfully biased.")
		biased_50K = True     
	
	#Hemt_Id should be within 0.5mA of the desired value without going over
	#Until this is true, adjust Vg, unless it would go out of acceptable range
	iters = 0
	Vg_hemt = S.get_hemt_gate_voltage()
	amp_biases = S.get_amplifier_biases()
	while amp_biases['hemt_Id'] > amp_hemt_Id or amp_biases['hemt_Id'] < amp_hemt_Id - 0.5:
		delta_Id = amp_biases['hemt_Id']-amp_hemt_Id        
		#Vg step is set by the size of the offset
		Vg_min = -1.2
		Vg_max  = -0.6
		if np.abs(delta_Id) > 1.5:
			Vg_step = 0.1
		else:
			Vg_step = 0.01
		
		#Increase Vg if Id is too low
		if amp_biases['hemt_Id'] < amp_hemt_Id:
			if Vg_hemt + Vg_step > Vg_max:
				print(f"Adjustment would go beyond max Vg={Vg_max}. Unable to change"+
					  "Hemt_Id to desired value")
				break
			S.set_hemt_gate_voltage(Vg_hemt + Vg_step)
			Vg_hemt = S.get_hemt_gate_voltage()

		#Decrease Vg is Id is too high
		else:
			if Vg_hemt - Vg_step < Vg_min:
				print(f"Adjustment would go beyond min Vg={Vg_min}. Unable to change"+
					  "Hemt_Id to desired value")
				break
			S.set_hemt_gate_voltage(Vg_hemt - Vg_step)
			Vg_hemt = S.get_hemt_gate_voltage()
		
		iters += 1
		amp_biases = S.get_amplifier_biases()
		print(f"Adjusting hemt_Id, iteration {iters}. Current hemt_Id"+ 
			  "= {amp_biases['hemt_Id']}, Vg_hemt={Vg_hemt}")
		if iters > 30:
			print("Number of iterations too large: unable to change Hemt_Id to"+
				  "desired value") 
			break
		
	amp_biases = S.get_amplifier_biases()
	if amp_biases['hemt_Id'] <= amp_hemt_Id or amp_biases['hemt_Id'] >= amp_hemt_Id - 0.5:
		Id_hemt_in_range = True
	else:
		Id_hemt_in_range = False
	print(f"Final hemt current = {amp_biases['hemt_Id']}")
	print(f"Desired hemt current = {amp_hemt_Id}")
	print(f"hemt current within range of desired value: {Id_hemt_in_range}")
	print(f"Final hemt gate voltage is {Vg_hemt}")


	#50K_Id should be within 0.5mA of the desired value without going over
	#Until this is true, adjust Vg, as long as it doesn't go out of acceptable range
	iters = 0
	Vg_50K = S.get_50k_amp_gate_voltage()
	amp_biases = S.get_amplifier_biases()
	while np.abs(amp_biases['50K_Id'] - amp_50K_Id) > 0.5:
		delta_Id = amp_biases['50K_Id']-amp_50K_Id        
		#Vg step is set by the size of the offset
		Vg_min = -0.8
		Vg_max  = -0.3
		if np.abs(delta_Id) > 1.5:
			Vg_step = 0.1
		else:
			Vg_step = 0.01
		
		#Increase Vg if Id is too high, and vice versa
		if amp_50K_Id > amp_biases['50K_Id']:
			if Vg_50K + Vg_step > Vg_max:
				print(f"Adjustment would go beyond max Vg={Vg_max}. Unable to change"+
					  "50K_Id to desired value")
				break
			S.set_50k_amp_gate_voltage(Vg_50K + Vg_step)
			Vg_50K = S.get_50k_amp_gate_voltage()
			
		else:
			if Vg_50K - Vg_step < Vg_min:
				print(f"Adjustment would go beyond min Vg={Vg_min}. Unable to change"+
					  "50K_Id to desired value")
				break
			S.set_50k_amp_gate_voltage(Vg_50K - Vg_step)
			Vg_50K = S.get_50k_amp_gate_voltage()
		
		iters += 1
		amp_biases = S.get_amplifier_biases()
		print(f"Adjusting 50K_Id, iteration {iters}. Current 50K_Id "+
			  f"= {amp_biases['50K_Id']}, Vg_50K={Vg_50K}")
		if iters > 30:
			print("Number of iterations too large: unable to change 50K_Id to"+
				  "desired value") 
			break
		
	amp_biases = S.get_amplifier_biases()
	if amp_biases['50K_Id'] <= amp_50K_Id or amp_biases['50K_Id'] >= amp_50K_Id - 0.5:
		Id_50K_in_range = True
	else:
		Id_50K_in_range = False
	print(f"Final 50K current = {amp_biases['50K_Id']}")
	print(f"Desired 50K current = {amp_50K_Id}")
	print(f"50K current within range of desired value: {Id_50K_in_range}")
	print(f"Final 50K gate voltage is {Vg_50K}")

	
	#Check timing is active.
	#Waiting for smurf timing card to be defined
	#Ask if there is a way to add 122.8 MHz external clock check

	
	#Check JESD connection on bay 0 and bay 1
	#Return connections for both bays
	#Make distinction between config not having bay and jesd check failed
	if bay0:
		jesd_tx0, jesd_rx0 = S.check_jesd(0)
		if jesd_tx0:
			print(f"bay 0 jesd_tx connection working")
		else:
			print(f"bay 0 jesd_tx connection NOT working. Rest of script may not function")
		if jesd_rx0:
			print(f"bay 0 jesd_rx connection working")
		else:
			print(f"bay 0 jesd_rx connection NOT working. Rest of script may not function")
	if not bay0:
		jesd_tx0, jesd_rx0 = False, False
		print("Bay 0 not enabled. Skipping connection check")

	if bay1:
		jesd_tx1, jesd_rx1 = S.check_jesd(1)
		if jesd_tx1:
			print(f"bay 1 jesd_tx connection working")
		else:
			print(f"bay 1 jesd_tx connection NOT working. Rest of script may not function")
		if jesd_rx1:
			print(f"bay 1 jesd_rx connection working")
		else:
			print(f"bay 1 jesd_rx connection NOT working. Rest of script may not function")
	if not bay1:
		jesd_tx1, jesd_rx1 = False, False
		print("Bay 1 not enabled. Skipping connection check")
		

	# Full band response. This is a binary test to determine that things are plugged in.
	# Typical in-band noise values are around ~2-7, so here check that average value of
	# noise through band 0 is above 1.
	#Check limit makes sense when through system
	
	S.set_att_uc(0,30)
	freq, response = S.full_band_resp(band=0)
	#Get the response in-band
	resp_inband = []
	for f, r in zip(freq, np.abs(response)):
		if f > -2.5e8 and f < 2.5e8:
			resp_inband.append(r)
	#If the mean is > 1, say response received
	if np.mean(resp_inband) > 1:
		resp_check = True
		print("Full band response check passed")
	else:
		resp_check = False
		print("Full band response check failed - maybe something isn't plugged in?")

		
	#Check if ADC is clipping. Probably should be a different script, after
	# characterizing system to know what peak data amplitude to simulate
	#Should result in ADC_clipping = T/F
	#Iterate through lowest to highest band, stop when no clipping.
	#Find max value of output of S.read_adc_data(0), compare to pre-set threshold
	# Probably should have a 'good' 'warning', and 'failed' output
	#Above functions are 'startup_check", this is a seperate function

	print("Health check finished")
	return biased_hemt, biased_50K, Id_hemt_in_range, Id_50K_in_range, Vg_hemt, Vg_50K, jesd_tx0, jesd_rx0, jesd_tx1, jesd_rx1, resp_check


if __name__=='__main__':    
	parser = argparse.ArgumentParser()

	# Arguments that are needed to create Pysmurf object
	parser.add_argument('--setup', action='store_true')
	parser.add_argument('--config-file', required=True)
	parser.add_argument('--epics-root', default='smurf_server_s2')

	# Custom arguments for this script
	parser.add_argument('--bay0', type=bool, required=True, 
		help='Whether or not bay0 is active')

	parser.add_argument('--bay1', type=bool, required=True, 
		help='Whether or not bay1 is active')

	parser.add_argument('--amp-hemt-Id', type=float, required=True, 
		help='hemt amplifier target current')

	parser.add_argument('--amp-50K-Id', type=float, required=True, 
		help='50K amplifier target current')

	# Parse command line arguments
	args = parser.parse_args()

	S = pysmurf.client.SmurfControl(
			epics_root = args.epics_root,
			cfg_file = args.config_file,
			setup = args.setup, make_logfile=False,
	)
	
	#Put your script calls here
	health_check(S, bay0 = args.bay0, bay1 = args.bay1, amp_hemt_Id = args.amp_hemt_Id, amp_50K_Id = args.amp_50K_Id)
	
