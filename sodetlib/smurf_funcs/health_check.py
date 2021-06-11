"""
Module for smurf and ocs health_check operations.
"""
from sodetlib.util import TermColors, cprint
from sodetlib.smurf_funcs.optimize_params import optimize_bias
from pysmurf.client.util.pub import set_action
import numpy as np
import time


@set_action()
def health_check(S, cfg):
    """
    Performs a system health check. This includes checking/adjusting amplifier
    biases, checking timing, checking the jesd connection, and checking that
    noise can be seen through the system.

    Parameters
    ----------
    S : pysmurf.client.SmurfControl
        Smurf control object
    cfg : DetConfig
        sodetlib config object

    Returns
    -------
    success: bool
        Returns true if all of the following checks were successful:
            - hemt and 50K are able to be biased
            - Id is in range for hemt and 50K
            - jesd_tx and jesd_rx connections are working on specified bays
            - response check for band 0
    """
    amp_hemt_Id = cfg.dev.exp['amp_hemt_Id']
    amp_50K_Id = cfg.dev.exp['amp_50k_Id']

    bays = S.which_bays()
    bay0 = 0 in bays
    bay1 = 1 in bays

    # Turns on both amplifiers and checks biasing.

    cprint("Checking biases", TermColors.HEADER)
    S.C.write_ps_en(11)
    amp_biases = S.get_amplifier_biases()
    biased_hemt = np.abs(amp_biases['hemt_Id']) > 0.2
    biased_50K = np.abs(amp_biases['50K_Id']) > 0.2
    if not biased_hemt:
        cprint("hemt amplifier could not be biased. Check for loose cable",
               False)
    if not biased_50K:
        cprint("50K amplifier could not be biased. Check for loose cable",
               False)

    # Optimize bias voltages
    if biased_hemt and biased_50K:
        cprint("Scanning hemt bias voltage", TermColors.HEADER)
        Id_hemt_in_range = optimize_bias(S, amp_hemt_Id, -1.2, -0.6, 'hemt')
        cprint("Scanning 50K bias voltage", TermColors.HEADER)
        Id_50K_in_range = optimize_bias(S, amp_50K_Id, -0.8, -0.3, '50K')
        time.sleep(0.2)
        amp_biases = S.get_amplifier_biases()
        Vg_hemt, Vg_50K = amp_biases['hemt_Vg'], amp_biases['50K_Vg']
        print(f"Final hemt current = {amp_biases['hemt_Id']}")
        print(f"Desired hemt current = {amp_hemt_Id}")
        cprint(f"hemt current within range of desired value: "
                            f" {Id_hemt_in_range}",Id_hemt_in_range)
        print(f"Final hemt gate voltage is {amp_biases['hemt_Vg']}")

        print(f"Final 50K current = {amp_biases['50K_Id']}")
        print(f"Desired 50K current = {amp_50K_Id}")
        cprint(f"50K current within range of desired value:"
                            f"{Id_50K_in_range}", Id_50K_in_range)
        print(f"Final 50K gate voltage is {amp_biases['50K_Vg']}")
    else:
        cprint("Both amplifiers could not be biased... skipping bias voltage "
               "scan", False)
        Id_hemt_in_range = False
        Id_50K_in_range = False

    # Check timing is active.
    # Waiting for smurf timing card to be defined
    # Ask if there is a way to add 122.8 MHz external clock check

    # Check JESD connection on bay 0 and bay 1
    # Return connections for both bays, or passes if bays not active
    cprint("Checking JESD Connections", TermColors.HEADER)
    if bay0:
        jesd_tx0, jesd_rx0, status = S.check_jesd(0)
        if jesd_tx0:
            cprint(f"bay 0 jesd_tx connection working", True)
        else:
            cprint(f"bay 0 jesd_tx connection NOT working. "
                    "Rest of script may not function", False)
        if jesd_rx0:
            cprint(f"bay 0 jesd_rx connection working", True)
        else:
            cprint(f"bay 0 jesd_rx connection NOT working. "
                    "Rest of script may not function", False)
    else:
        jesd_tx0, jesd_rx0 = False, False
        print("Bay 0 not enabled. Skipping connection check")

    if bay1:
        jesd_tx1, jesd_rx1, status = S.check_jesd(1)
        if jesd_tx1:
            cprint(f"bay 1 jesd_tx connection working", True)
        else:
            cprint(f"bay 1 jesd_tx connection NOT working. Rest of script may "
                   "not function", False)
        if jesd_rx1:
            cprint(f"bay 1 jesd_rx connection working", True)
        else:
            cprint(f"bay 1 jesd_rx connection NOT working. Rest of script may "
                    "not function", False)
    else:
        jesd_tx1, jesd_rx1 = False, False
        print("Bay 1 not enabled. Skipping connection check")

    # Full band response. This is a binary test to determine that things are
    # plugged in.  Typical in-band noise values are around ~2-7, so here check
    # that average value of noise through band 0 is above 1.  

    # Check limit makes sense when through system
    cprint("Checking full-band response for band 0", TermColors.HEADER)
    band_cfg = cfg.dev.bands[0]
    S.set_att_uc(0, band_cfg['uc_att'])

    freq, response = S.full_band_resp(band=0)
    # Get the response in-band
    resp_inband = []
    band_width = 500e6  # Each band is 500 MHz wide
    for f, r in zip(freq, np.abs(response)):
        if -band_width/2 < f < band_width/2:
            resp_inband.append(r)
    # If the mean is > 1, say response received
    if np.mean(resp_inband) > 1: #LESS THAN CHANGE
        resp_check = True
        cprint("Full band response check passed", True)
    else:
        resp_check = False
        cprint("Full band response check failed - maybe something isn't "
               "plugged in?", False)

    # Check if ADC is clipping. Probably should be a different script, after
    # characterizing system to know what peak data amplitude to simulate
    # Should result in ADC_clipping = T/F
    # Iterate through lowest to highest band, stop when no clipping.
    # Find max value of output of S.read_adc_data(0), compare to pre-set threshold
    # Probably should have a 'good' 'warning', and 'failed' output
    # Above functions are 'startup_check", this is a seperate function

    cfg.dev.update_experiment({
        'amp_hemt_Vg': Vg_hemt,
        'amp_50k_Vg': Vg_50K,
    })

    cprint("Health check finished! Final status", TermColors.HEADER)
    cprint(f" - Hemt biased: \t{biased_hemt}", biased_hemt)
    cprint(f" - Hemt Id in range: \t{Id_hemt_in_range}", Id_hemt_in_range)
    print(f" - Hemt (Id, Vg): \t{(amp_biases['hemt_Id'], amp_biases['hemt_Vg'])}\n")
    cprint(f" - 50K biased: \t\t{biased_50K}", biased_50K)
    cprint(f" - 50K Id in range: \t{Id_50K_in_range}", Id_50K_in_range)
    print(f" - 50K (Id, Vg): \t{(amp_biases['50K_Id'], amp_biases['50K_Vg'])}\n")
    cprint(f" - Response check: \t{resp_check}", resp_check)

    if bay0:
        cprint(f" - JESD[0] TX, RX: \t{(jesd_tx0, jesd_rx0)}",
               jesd_tx0 and jesd_rx0)
    if bay1:
        cprint(f" - JESD[1] TX, RX: \t{(jesd_tx1, jesd_rx1)}",
               jesd_tx1 and jesd_rx1)

    status_bools = [biased_hemt, biased_50K, Id_hemt_in_range, Id_50K_in_range,
                    resp_check]
    if bay0:
        status_bools.extend([jesd_tx0, jesd_rx0])
    if bay1:
        status_bools.extend([jesd_tx1, jesd_rx1])

    return all(status_bools)

