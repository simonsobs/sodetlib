"""
Constants useful for general sodetlib ops.
"""
import numpy as np

NBGS = 12
NBANDS = 8
CHANS_PER_BAND= 512
pA_per_phi0 = 9e6
pA_per_rad = pA_per_phi0 / (2*np.pi)

# Max bias voltage is 20 V, and there are 20 bits (about 1.0973e-5)
rtm_bit_to_volt = 20/(2**20)
