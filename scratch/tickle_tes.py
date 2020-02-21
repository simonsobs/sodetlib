import pysmurf.client
import argparse
import numpy as np
import time


if __name__=='__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--setup', action='store_true')
    parser.add_argument('--config-file', required=True)
    parser.add_argument('--epics-root', default='smurf_server_s2')

    parser.add_argument('--bias-group', type=int, required=True, 
                        help='bias group (must be in range [0,11])')

    #parser.add_argument('--step-or-tickle',required=True,default='tickle',
    #        help='Choose between injecting a tickle or step on the TES bias DAC')

    #parser.add_argument('--step-or-tickle-amp',type=float,required = True,
    #        help = 'Amplitude in volts of the step or tickle injected on the TES bias DAC')

    #parser.add_argumnet('--duration',type = float,default = 5,
    #        help = 'Duration in seconds of bias step or tickle')

    args = parser.parse_args()

    S = pysmurf.client.SmurfControl(
            epics_root = args.epics_root,
            cfg_file = args.config_file,
            setup = args.setup,make_logfile=False,
    )

    S.stream_data_on()                                                                             
    multiplier=1/10000.                                            
    scale = 2**17;                                                                               
    sig   = multiplier*scale*np.cos(2*np.pi*np.array(range(2048))/(2048));                         
    S.play_tes_bipolar_waveform(args.bias_group,sig)                                                             
    time.sleep(5)                                                                                 
    S.stop_tes_bipolar_waveform(args.bias_group) 
    
    S.stream_data_off()

