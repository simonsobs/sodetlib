import pysmurf
import numpy as np
import matplotlib.pyplot as plt

S = pysmurf.SmurfControl(make_logfile=False, setup=True)

S.set_noise_select(3, 1, write_log=True)
S.get_noise_select(3, write_log=True)

n_sample = 2**19
n = 4

dat = np.zeros((n, n_sample), dtype=complex)
for i in np.arange(n):
    dat[i] = S.read_adc_data(band, n_samples, hw_trigger=True)
    time.sleep(.5)

S.set_noise_select(3, 0, write_log=True)
S.get_noise_select(3, write_log=True)

fig, ax = plt.subplots(1)
end = 250
for i in np.arange(n):
    ax.plot(dat[n,:end])

plt.show()