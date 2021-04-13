import struct
import numpy as np
import sys
sys.path.append('../../../../')
import pysmurf

# dat_fullpath = '/data/cpu-b000-hp01/cryo_data/data2/20180912/1536771265/'+\
# 	'outputs/1536771308.dat'
# 	# 'outputs/1536771308.dat'

dat_fullpath = '/home/common/data/cpu-b000-hp01/cryo_data/data2/20180912/1536762729.dat'

# with open(dat_fullpath, mode='rb') as file: # b is important -> binary
#     fileContent = file.read()

# n = int(len(fileContent)/4)
# print(n)
# x = np.zeros(n, dtype=int)
# for i in np.arange(n):
# 	# print(fileContent[i*8:(i+1)*8])
# 	# print(int.from_bytes(fileContent[i*8:(i+1)*8], byteorder='little'))
# 	x[i] = int.from_bytes(fileContent[i*4:(i+1)*4], byteorder='little')


S = pysmurf.SmurfControl()
S.initialize(cfg_file='/home/cryo/pysmurf/cfg_files/experiment_fp28.cfg', 
	make_logfile=False, output_dir_only=True)

I, Q = S.read_stream_data(dat_fullpath, 0)