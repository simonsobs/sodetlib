# Runlike this exec(open("scratch/stephen/full_band_response_AMCatten.py").read())
# to use the pysmurf S object you've already initialized
import time
import numpy as np
import sys
import os
import matplotlib.pylab as plt
import re
plt.ion()
print('Beginning Attenuation Test')
timestamp = S.get_timestamp()
n_scan_per_band=1
wait_btw_bands_sec=.5
fig, Lax = plt.subplots(nrows=2, ncols=2) #one plot per Bay, per converter set
fig, Rax = plt.subplots(nrows=2, ncols=2) #one plot per Bay, per converter set
colors = ['r','y','g','c','b','m','k']
Z = 0 #for use with the response plots
z = 0 #for use with the attenuation plots

print('Getting Asset Tag for Bay 0')
lhsbay=S.get_amc_asset_tag(0)
if re.split('-|_',lhsbay)[1] == 'A02':
   LBoardType = 'high'
else:
   LBoardType = 'low'

print('Getting Asset Tag for Bay 1')   
rhsbay=S.get_amc_asset_tag(1)
if re.split('-|_',rhsbay)[1] == 'A02':
   RBoardType = 'high'
else:
   RBoardType = 'low'
"""
rhsbay='C03-A01-21'
lhsbay='C03-A01-21'
LBoardType = 'low'
RBoardType = 'low'
"""


print(f'{lhsbay},{rhsbay}')
att=[0,1,2,4,8,16,31]

uattplt = [[0 for x in range(len(att))] for y in range(8)]
dattplt = [[0 for x in range(len(att))] for y in range(8)]

print('Initilizing Boards')
#Set all attenuators to 0 just to be safe
for x in range(8):
    S.set_att_uc(x,0)
    S.set_att_dc(x,0)

time.sleep(1)  
print('STARTING TEST')
#Test for Up Converters
for uatt in att:
    for x in range(8):
        S.set_att_uc(x, uatt)
        S.set_att_dc(x, 0)
    time.sleep(.1) #wait a sec for attenuators to change
    resp_dict={}
    Luxplot = []
    Luyplot = []
    Ruxplot = []
    Ruyplot = []

    for band in range(8):
        print(' ')
        print(' ')
        print(f'Up Converters, Attenuation {uatt}, Band {band}')
        print(' ')
        print(' ')
        resp_dict[band]={}
        if band<4:
            resp_dict[band]['fc']=(4250+(band%4)*500) + (2000 if LBoardType=='high' else 0) #sets Bay 0 band center based on board type
        else:
            resp_dict[band]['fc']=(4250+(band%4)*500) + (2000 if RBoardType=='high' else 0) #sets Bay 1 band center based on board type
        
        f,resp=S.full_band_resp(band=band, make_plot=False, save_plot=False, show_plot=False,save_data=True, n_scan=n_scan_per_band, timestamp=timestamp, correct_att=False) #returns band response around band center
        
        resp_dict[band]['f']=f 
        resp_dict[band]['resp']=resp 
        
        f_plot=resp_dict[band]['f']/1e6 #response X-axis for band for attenuation setting
        resp_plot=resp_dict[band]['resp'] #response y-axis for band for attenuation setting
        plot_idx = np.where(np.logical_and(f_plot>-250, f_plot<250))
        uattplt[band][Z]=20*np.log10(np.mean(np.abs((resp_plot[plot_idx]))))             #used to check attenuation
        if band<4:
            Luxplot = np.concatenate([Luxplot,f_plot[plot_idx]+resp_dict[band]['fc']])       #Bay 0 full band response x-axis for attenuation setting  
            Luyplot = np.concatenate([Luyplot,20*np.log10(np.abs(resp_plot[plot_idx]))])     #Bay 0 full band response y-axis for attenuation setting 
        else:
            Ruxplot = np.concatenate([Ruxplot,f_plot[plot_idx]+resp_dict[band]['fc']])       #Bay 1 full band response x-axis for attenuation setting
            Ruyplot = np.concatenate([Ruyplot,20*np.log10(np.abs(resp_plot[plot_idx]))])     #Bay 1 full band response y-axis for attenuation setting
        
        time.sleep(wait_btw_bands_sec)
     
    Lax[0,0].plot(Luxplot,Luyplot,color=colors[Z],label=f'att{uatt}') #plot for Bay 0, up conveters
    Rax[0,0].plot(Ruxplot,Ruyplot,color=colors[Z],label=f'att{uatt}') #plot for Bay 1, up conveters
    Z = Z + 1

for x in range(8):
    S.set_att_uc(x, 0)
    S.set_att_dc(x, 0)
Z = 0
#Test for Down Converters
for datt in att:
    for x in range(8):
        S.set_att_uc(x, 0)
        S.set_att_dc(x, datt)
    time.sleep(.1) #wait a sec for attenuators to change
    resp_dict={}
    Ldxplot = []
    Ldyplot = []
    Rdxplot = []
    Rdyplot = []

    for band in range(8):
        print(' ')
        print(' ')
        print(f'Down Converters, Attenuation {datt}, Band {band}')
        print(' ')
        print(' ')
        resp_dict[band]={}
        if band<4:
            resp_dict[band]['fc']=(4250+(band%4)*500) + (2000 if LBoardType=='high' else 0) #sets Bay 0 band center based on board type
        else:
            resp_dict[band]['fc']=(4250+(band%4)*500) + (2000 if RBoardType=='high' else 0) #sets Bay 1 band center based on board type
        
        f,resp=S.full_band_resp(band=band, make_plot=False, show_plot=False, n_scan=n_scan_per_band,save_plot=False, timestamp=timestamp,correct_att=False) #returns band response around band center
        
        resp_dict[band]['f']=f 
        resp_dict[band]['resp']=resp 
        
        f_plot=resp_dict[band]['f']/1e6 #response X-axis for band for attenuation setting
        resp_plot=resp_dict[band]['resp'] #response y-axis for band for attenuation setting
        plot_idx = np.where(np.logical_and(f_plot>-250, f_plot<250))
        dattplt[band][Z]=20*np.log10(np.mean(np.abs((resp_plot[plot_idx]))))             #used to check attenuation
        if band<4:
            Ldxplot = np.concatenate([Ldxplot,f_plot[plot_idx]+resp_dict[band]['fc']])       #Bay 0 full band response x-axis for attenuation setting  
            Ldyplot = np.concatenate([Ldyplot,20*np.log10(np.abs(resp_plot[plot_idx]))])     #Bay 0 full band response y-axis for attenuation setting 
        else:
            Rdxplot = np.concatenate([Rdxplot,f_plot[plot_idx]+resp_dict[band]['fc']])       #Bay 1 full band response x-axis for attenuation setting
            Rdyplot = np.concatenate([Rdyplot,20*np.log10(np.abs(resp_plot[plot_idx]))])     #Bay 1 full band response y-axis for attenuation setting
        
        time.sleep(wait_btw_bands_sec)
     
    Lax[1,0].plot(Ldxplot,Ldyplot,color=colors[Z],label=f'att{datt}') #plot for Bay 0, down conveters
    Rax[1,0].plot(Rdxplot,Rdyplot,color=colors[Z],label=f'att{datt}') #plot for Bay 1, down conveters
    Z = Z + 1


for band in range(4):
    Lax[0,1].plot(np.divide(att,2),np.abs(uattplt[band]-uattplt[band][0]), color=colors[z],label=f'band{band}')  #up attenuation check for Bay 0
    Lax[1,1].plot(np.divide(att,2),np.abs(dattplt[band]-dattplt[band][0]), color=colors[z],label=f'band{band}')  #down attenuation check for Bay 0
    
    Rax[0,1].plot(np.divide(att,2),np.abs(uattplt[band+4]-uattplt[band+4][0]), color=colors[z],label=f'band{band}')   #up attenuation check for Bay 1
    Rax[1,1].plot(np.divide(att,2),np.abs(dattplt[band+4]-dattplt[band+4][0]), color=colors[z],label=f'band{band}')   #down attenuation check for Bay 1
    z = z+1
    
for x in range(8):
    S.set_att_uc(x, 0)
    S.set_att_dc(x, 0)

#setting up response plots
Lax[0,0].set_title(f'UC band response for Bay 0(Board {lhsbay})')
Lax[0,0].legend(loc='lower left',fontsize=8)
Lax[0,0].set_ylabel("20*log10(abs(Response))")
Lax[0,0].set_xlabel('Frequency [MHz]')

Lax[0,1].set_title('Bay 0 UC Attenuation Check Plot')
Lax[0,1].legend(loc='lower left',fontsize=8)
Lax[0,1].set_ylabel("Average Band Normalized Attenuation")
Lax[0,1].set_xlabel('Attenuation Setpoint') 

Lax[1,0].set_title(f'DC band response for Bay 0(Board {lhsbay})')
Lax[1,0].legend(loc='lower left',fontsize=8)
Lax[1,0].set_ylabel("20*log10(abs(Response))")
Lax[1,0].set_xlabel('Frequency [MHz]')

Lax[1,1].set_title(f'Bay 0 DC Attenuation Check Plot')
Lax[1,1].legend(loc='lower left',fontsize=8)
Lax[1,1].set_ylabel("Average Band Normalized Attenuation")
Lax[1,1].set_xlabel('Attenuation Setpoint')

#setting up attenuation plots
Rax[0,0].set_title(f'UC band response for Bay 1(Board {rhsbay})')
Rax[0,0].legend(loc='lower left',fontsize=8)
Rax[0,0].set_ylabel("20*log10(abs(Response))")
Rax[0,0].set_xlabel('Frequency [MHz]')

Rax[0,1].set_title('Bay 1 UC Attenuation Check Plot')
Rax[0,1].legend(loc='lower left',fontsize=8)
Rax[0,1].set_ylabel("Average Band Normalized Attenuation")
Rax[0,1].set_xlabel('Attenuation Setpoint') 

Rax[1,0].set_title(f'DC band response for Bay 1(Board {rhsbay})')
Rax[1,0].legend(loc='lower left',fontsize=8)
Rax[1,0].set_ylabel("20*log10(abs(Response))")
Rax[1,0].set_xlabel('Frequency [MHz]')

Rax[1,1].set_title(f'Bay 1 DC Attenuation Check Plot')
Rax[1,1].legend(loc='lower left',fontsize=8)
Rax[1,1].set_ylabel("Average Band Normalized Attenuation")
Rax[1,1].set_xlabel('Attenuation Setpoint')


save_name = '{}_full_band_resp_atten.png'.format(timestamp)

fig.suptitle(save_name)

print(f'Saving plot to {os.path.join(S.plot_dir, save_name)}')
plt.savefig(os.path.join(S.plot_dir, save_name),
            bbox_inches='tight')

plt.show()

# log plot file
logf=open('/data/smurf_data/smurf_loop.log','a+')
logf.write(f'{os.path.join(S.plot_dir, save_name)}'+'\n')
logf.close()
timeend = S.get_timestamp()
print(f'Test started: {timestamp}')
print(f'Test started: {timeend}')
print(f'RF Test Complete')

