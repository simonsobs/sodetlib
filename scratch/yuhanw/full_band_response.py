# Runlike this exec(open("scratch/shawn/full_band_response.py").read())
# to use the pysmurf S object you've already initialized
import time
import numpy as np
import sys
import os
import matplotlib.pylab as plt
plt.ion()

n_scan_per_band=5
wait_btw_bands_sec=5

timestamp=S.get_timestamp()
bands=S.config.get('init').get('bands')

resp_dict={}
for band in bands:
    print(' ')
    print(' ')
    print(f'Band {band}')
    print(' ')
    print(' ')
    resp_dict[band]={}
    resp_dict[band]['fc']=S.get_band_center_mhz(band)

    f,resp=S.full_band_resp(band=band, make_plot=False, show_plot=False, n_scan=n_scan_per_band, timestamp=timestamp, save_data=True)
    resp_dict[band]['f']=f
    resp_dict[band]['resp']=resp
    
    time.sleep(wait_btw_bands_sec)

fig, ax = plt.subplots(2, figsize=(6,7.5), sharex=True)

# plt.suptitle(f'slot={S.slot_number} AMC0={S.get_amc_asset_tag(0)} AMC2={S.get_amc_asset_tag(1)}')

ax[0].set_title(f'Full band response {timestamp}')
last_angle=None
for band in bands:
    f_plot=resp_dict[band]['f']/1e6
    resp_plot=resp_dict[band]['resp']
    plot_idx = np.where(np.logical_and(f_plot>-250, f_plot<250))
    ax[0].plot(f_plot[plot_idx]+resp_dict[band]['fc'], np.log10(np.abs(resp_plot[plot_idx])),label=f'b{band}')
    angle = np.unwrap(np.angle(resp_plot))
    if last_angle is not None:
        angle-=(angle[0]-last_angle)
    ax[1].plot(f_plot[plot_idx]+resp_dict[band]['fc'], angle[plot_idx],label=f'b{band}')
    last_angle=angle[plot_idx][-1]
    
print('data taking done')

ax[0].legend(loc='lower left',fontsize=8)
ax[0].set_ylabel("log10(abs(Response))")
ax[0].set_xlabel('Frequency [MHz]')

ax[1].legend(loc='lower left',fontsize=8)
ax[1].set_ylabel("phase [rad]")
ax[1].set_xlabel('Frequency [MHz]')
    
save_name = '{}_full_band_resp_all.png'.format(timestamp)
plt.title(save_name)

plt.tight_layout()

print(f'Saving plot to {os.path.join(S.plot_dir, save_name)}')
plt.savefig(os.path.join(S.plot_dir, save_name), 
            bbox_inches='tight')
plt.show()

save_name = '{}_full_band_resp_all.npy'.format(timestamp)
print(f'Saving data to {os.path.join(S.output_dir, save_name)}')
full_resp_data = os.path.join(S.output_dir, save_name)
path = os.path.join(S.output_dir, full_resp_data) 
np.save(path, resp_dict)

# log plot file
logf=open('/data/smurf_data/smurf_loop.log','a+')
logf.write(f'{os.path.join(S.plot_dir, save_name)}'+'\n')
logf.close()

print('Done running full_band_response.py.')
