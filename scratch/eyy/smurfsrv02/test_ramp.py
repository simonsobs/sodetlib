import sys
sys.path.append('/home/cryo/')
import pysmurf
import time

epics_root = 'dev_epics'
sd = epics_root + ':AMCc:FpgaTopLevel:AppTop:AppCore:StreamReg:StreamData[{}]'

S = pysmurf.SmurfControl(setup=False, epics_root=epics_root,
                         make_logfile=False)

ch = 0
step_size = 2**6
a = -2**15

S._caput(sd.format(ch), 0, write_log=True)
while True:
    write_log=False
    if a % (step_size*100) == 0:
        write_log=True
    S._caput(sd.format(ch), a, write_log=write_log)
    a += step_size
    time.sleep(.02)
    if a > 2**15:
        a = -2**15
