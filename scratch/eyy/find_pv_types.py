import numpy as np
import pysmurf.client

epics_prefix = 'smurf_server_s5'
config_file='/data/pysmurf_cfg/experiment_fp30_cc02-03_lbOnlyBay0.cfg' 

S = pysmurf.client.SmurfControl(epics_root=epics_prefix,cfg_file=config_file,setup=False,make_logfile=False,shelf_manager="shm-smrf-sp01")

skip_func = np.array(['get_tone_file_path', 'get_waveform_start_addr', 'get_waveform_end_addr',
                      'get_waveform_wr_addr', 'get_trigger_hw_arm', 'get_rtm_arb_waveform_lut_table',
                      'get_streaming_datafile'])

out_dict = {}

with open("./pysmurf/client/command/smurf_command.py") as search:
    for line in search:
        line = line.rstrip()
        if "def get_" in line:
            sp = line.split(' ')
            for l in sp:
                if "get_" in l:
                    func = l.split('(')[0]
                    print(func)
                    if func not in skip_func:
                        try:
                            out_dict[func] = getattr(S, func)()
                        except TypeError:
                            try:
                                out_dict[func] = getattr(S, func)(1)
                            except TypeError:
                                out_dict[func] = getattr(S, func)(0,1)
                        

t = np.array([])
for k in out_dict.keys():
    t = np.append(t, str(type(out_dict[k])))

print(np.unique(t))
