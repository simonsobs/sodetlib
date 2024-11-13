import epics
import sys
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('slot', type=int)
parser.add_argument('--debug', action='store_true')

args = parser.parse_args()

slot=args.slot
debug=args.debug

# levels
pcie_root='pcie_test0'
pcie_top='pcie:'
pcie_core='Core:'
pcie_udpgrp='UdpGrp:'
pcie_udpengine='UdpEngine[{}]:'
pcie_rssiclient='RssiClient[{}]:'

# regs
client_remote_ip0=pcie_root+':'+pcie_top+pcie_core+pcie_udpgrp+pcie_udpengine+'ClientRemoteIp[0]'
rssi_validcnt=pcie_root+':'+pcie_top+pcie_core+pcie_udpgrp+pcie_rssiclient+'ValidCnt'
rssi_restartconn=pcie_root+':'+pcie_top+pcie_core+pcie_udpgrp+pcie_rssiclient+'C_RestartConn'

# which slots are assigned to which udp engines
slot2udpe={}
nudpe=6
ipbyeng=[epics.PV(name).get(as_string=True) for name in [client_remote_ip0.format(iudpe) for iudpe in range(nudpe)]]

slots={}
for idx,ibe in enumerate(ipbyeng):
    s=int(ibe.split('.')[-1])-100
    if s>0:
        slots[s]={}
        slots[s]['engine']=idx

if slot not in slots.keys():
    print(f'Requested slot {slot} not one of the available slots (={[int(k) for k in slots.keys()]}).  Doing nothing!')
    sys.exit(1)
else:
    print(f'Restarting RSSI connection for slot {slot}.')
    validcntpv=epics.PV(rssi_validcnt.format(slots[slot]['engine']))
    restartconnpv=epics.PV(rssi_restartconn.format(slots[slot]['engine']))

    if debug:
        print(f'Pre-ValidCnt {validcntpv.get()}')
    restartconnpv.put(1,wait=True)
    if debug:
        time.sleep(2)
        print(f'Post-ValidCnt {validcntpv.get()}')
        time.sleep(2)        
        print(f'Post-ValidCnt {validcntpv.get()}')
        time.sleep(2)        
        print(f'Post-ValidCnt {validcntpv.get()}')