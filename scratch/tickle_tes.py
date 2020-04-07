import matplotlib
matplotlib.use('Agg')

import pysmurf.client
import argparse
import numpy as np
import time
import sys


def tickle_tes(S, bias_group, duration=None, amp=0.005, freq=1):
    """
        Biases TES with a sine wave for specified duration.

        Args:
            S:  
                Pysmurf control object
            bias_group (int): 
                smurf bias group
            duration (float):
                duration of tickle (seconds)
            amp (float):
                amplitude of sine wave (Volts)
            freq (float):
                approximate frequency of sine wave (Hz)
    """
    bit_amp = amp / S._rtm_slow_dac_bit_to_volt

    # Array of times for daq signal in seconds
    # This is not a perfect conversion...
    t = np.arange(0, 2048) * 6.4e-9 * S.get_rtm_arb_waveform_timer_size()
    signal  = bit_amp*np.cos(2*np.pi*freq*t);                         

    S.play_tes_bipolar_waveform(bias_group, signal)

    if duration is not None:
        time.sleep(duration)
        S.stop_tes_bipolar_waveform(args.bias_group)


if __name__=='__main__':    
    parser = argparse.ArgumentParser()

    # Arguments that are needed to create Pysmurf object
    parser.add_argument('--setup', action='store_true')
    parser.add_argument('--config-file', required=True)
    parser.add_argument('--epics-root', default='smurf_server_s2')

    # Custom arguments for this script
    parser.add_argument('--bias-group', type=int, required=True, 
                        help='bias group (must be in range [0,11])')

    parser.add_argument('--freq', type=float, default=1, 
        help="Frequency of sine wave (approximately in hz)")
    parser.add_argument('--amp', type=float, default=.005,
        help="Amplitude of sine wave (Volts)"
    )
    parser.add_argument('--duration', type=float, default=5,
        help="Duration of tickle"
    )

    # Parse command line arguments
    args = parser.parse_args()

    if args.freq >= 1000:
        raise ValueError("Frequency must be less than 1 kHz ")

    S = pysmurf.client.SmurfControl(
            epics_root = args.epics_root,
            cfg_file = args.config_file,
            setup = args.setup, make_logfile=False,
    )

    S.stream_data_on()    
    try:                                                                          
        tickle_tes(S, args.bias_group, args.duration, args.amp, args.freq)
    finally:
        # Makes sure streaming is turned off if this funciton fails!
        S.stream_data_off()

