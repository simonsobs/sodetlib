import time
import numpy as np

import typing
if typing.TYPE_CHECKING:
    from pysmurf.client.base.smurf_control import SmurfControl
    from sodetlib.det_config import DetConfig

def uxm_relock(S: SmurfControl, cfg: DetConfig, id_tolerance=0.5):
    """
    Relock steps:

        1. Reset state (all off, disable waveform, etc.)
        2. Set amps and check tolerance 
        3. load tune
        4. setup-notches if specified (new_master_assignment=False)
        5. Serial gradient descent and eta scan 
        6. Run tracking setup
        7. Measure noise
    """
    summary = {}

    S.all_off()  # Turn off Flux ramp, tones, and biases
    S.set_rtm_arb_waveform_enable(0)
    S.set_filter_disable(0)
    S.set_downsample_factor(20)
    S.set_mode_dc()

    exp = cfg.dev.exp

    # 1. Reset amplifiers
    S.set_50k_amp_gate_voltage(exp['amp_50k_Vg'])
    S.set_hemt_gate_voltage(exp['amp_hemt_Vg'])
    S.C.write_ps_en(3)
    time.sleep(0.1)
    biases = S.get_amplifier_biases()

    in_range_50k = np.abs(biases['50k_Id'] - exp['amp_50k_Id']) < id_tolerance
    in_range_hemt = np.abs(biases['hemt_Id'] - exp['amp_hemt_Id']) < id_tolerance

    if not (in_range_50k and in_range_hemt):
        S.log("Hemt or 50K Amp drain current not within tolerance")
        S.log(f"Target hemt Id: {exp['amp_hemt_Id']}")
        S.log(f"Target 50K Id: {exp['amp_50k_Id']}")
        S.log(f"tolerance: {id_tolerance}")
        S.log(f"biases: {biases}")
        return False

