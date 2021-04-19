import pysmurf.client
import numpy as np
import matplotlib.pyplot as plt
import os

####
# Assumes you've already setup the system.
####

### System Configuration ###
epics_prefix = 'smurf_server_s5'
config_file = os.path.join('/data/pysmurf_cfg/experiment_fp30_cc02-03_lbOnlyBay0.cfg')

### Function variables ###
find_resonators = False
tune_filename = None
band = 2
subband = np.arange(13, 115)
noise_time = 10
data_rate = 200  # Hz

# reset_rate_khzs = np.array([4, 10, 15, 20, 25, 30])
# n_phi0s = np.array([4, 4, 4, 4, 4, 4, 4])
reset_rate_khzs = np.array([4, 20, 40, 60])
n_phi0s = np.array([4, 4, 4, 4])
lms_enable2 = False
lms_enable3 = False
lms_gain = 7
filter_order = 4

# Instatiate pysmurf object
S = pysmurf.client.SmurfControl(epics_root=epics_prefix, cfg_file=config_file,
    setup=False, make_logfile=False, shelf_manager='shm-smrf-sp01')

if find_resonators:
    S.find_freq(band, subband, make_plot=True)
    S.setup_notches(band, new_master_assignment=True)
    S.run_serial_gradient_descent(band)
    S.run_serial_eta_scan(band)
else:
    S.load_tune(filename=tune_filename)


f = {}
df = {}
noise_files = {}

n_steps = len(reset_rate_khzs)

# Take measurements
S.set_downsample_factor(1)
for i in np.arange(n_steps):
    S.relock(band)
    S.run_serial_gradient_descent(band)
    S.run_serial_eta_scan(band)
    f[i], df[i], sync = S.tracking_setup(band, reset_rate_khz=reset_rate_khzs[i],
        fraction_full_scale=.5, make_plot=True, show_plot=False, nsamp=2**18,
        lms_gain=lms_gain, lms_freq_hz=None, meas_lms_freq=False,
        feedback_start_frac=.25, feedback_end_frac=.98, meas_flux_ramp_amp=True,
        n_phi0=n_phi0s[i], lms_enable2=lms_enable2, lms_enable3=lms_enable3)

    # Cull bad detectors
    S.check_lock(band)

    # Take noise data
    # factor = int(reset_rate_khzs[i]/data_rate)
    # S.set_downsample_factor(factor)
    S.set_downsample_filter(filter_order, reset_rate_khzs[i]*1.0E3/4,
        write_log=False)
    noise_files[i] = S.take_stream_data(noise_time)


# Make plot
f_swing = {}
df_std = {}

fig, ax = plt.subplots(1, figsize=(6, 4.5))
cm = plt.get_cmap('gist_rainbow')
for i in np.arange(n_steps):
    channel = np.where(np.std(df[i], axis=0)!=0)[0]
    f_swing[i] = np.max(f[i][:,channel], axis=0) - \
        np.min(f[i][:,channel], axis=0)
    df_std[i] = np.std(df[i][:,channel], axis=0)

    # Convert to kHz
    f_swing[i] *= 1.0E3
    df_std[i] *= 1.0E3

    color = cm(i/n_steps)

    label = f'FR {reset_rate_khzs[i]} kHz ' + r'n$\phi_0$ ' + f'{n_phi0s[i]}'
    ax.plot(f_swing[i], df_std[i], '.', color=color, label=label)

timestamp = S.get_timestamp()

ax.legend()
ax.set_xlabel('Freq Swing [kHz]')
ax.set_ylabel('std(df) [kHz]')
ax.set_title(f'{timestamp} band {band} LMS2 {lms_enable2} LMS3 {lms_enable3} Gain {lms_gain}')
plt.tight_layout()
plt.savefig(os.path.join(S.plot_dir,
    f'{timestamp}_compare_tracking_band{band}.png'), bbox_inches='tight')
plt.close()

# Save data
np.save(os.path.join(S.output_dir, f'{timestamp}_compare_tracking_f'), f)
np.save(os.path.join(S.output_dir, f'{timestamp}_compare_tracking_df'), df)
np.save(os.path.join(S.output_dir, f'{timestamp}_compare_tracking_noise'),
    noise_files)
np.save(os.path.join(S.output_dir, f'{timestamp}_reset_rate_khz'),
    reset_rate_khzs)
np.save(os.path.join(S.output_dir, f'{timestamp}_n_phi0'),
    n_phi0s)