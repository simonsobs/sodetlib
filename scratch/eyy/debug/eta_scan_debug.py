import scipy.io as sio
import os
import sys
sys.path.append('../../../../')
import pysmurf
import numpy as np

S = pysmurf.SmurfControl(epics_root='mitch_epics')

test_filedir = '/home/common/data/cpu-b000-hp01/cryo_data/data2/20180819/'+ \
	'1534700282'

eta = np.ravel(sio.loadmat(os.path.join(test_filedir,
	'1534700282mitch_epics_etaOut.mat'))['etaOut'])

# for i in np.arange(len(eta))
for i in np.arange(150,155):
	resp = np.ravel(eta[i][3])
	freq = np.ravel(eta[i][4])

	if len(resp) > 0:
		I, Q, r, resid, eta_est = S.estimate_eta_parameter(freq, resp)
		print(I)
		print(Q)
		print(r)
		S.plot_eta_estimate(freq, resp, Ic=I, Qc=Q, r=r, eta=eta_est)


