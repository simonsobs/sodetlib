import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
band = 2
fname = open('/data/sodetlib_data/20200623_IV_NIST_SPB_3_D_UHF_r3_105_to_200_mK_5mk_step.txt','r')
Lines = fname.readlines()

Temps = []
Psat = {}

def gfit(T,k,n,Tc):
    return k*(Tc**n - T**n)

for i,line in enumerate(Lines):
    long_fn = 
    temp_dict = np.load(line.split('iv_file : ')[-1].split('\n')[0],allow_pickle = True)
    temp_dict = temp_dict.item()
    chans = np.fromiter(temp_dict[band].keys(),dtype = int)
    Temps.append(float(line.split(' mK')[0].split('= ')[-1])/1000)
    for chan in chans:
        if i == 0:
            Psat[chan] = []
        Psat[chan].append(temp_dict[band][chan]['p_trans'])

copy_temps = np.asarray(Temps)
fit_temp = np.linspace(105,200,100)/1000
for chan in chans:
    iter_temp = copy_temps
    Psat_copy  = np.asarray(Psat[chan])
    idx = np.argmax(np.diff(Psat_copy))
    iter_temp=iter_temp[0:idx+1]
    Psat_fit=Psat_copy[0:idx+1]
    Psat_fit[-1] = 0
    plt.figure(figsize = (12,10))
    plt.plot(iter_temp,Psat_fit*1e-12,'o')
    popt,pcov = opt.curve_fit(gfit,iter_temp,Psat_fit*1e-12,maxfev = 3000,p0 = [10e-9,3,0.18])
    plt.plot(fit_temp,gfit(fit_temp,popt[0],popt[1],popt[2]))
    plt.title(f'Channel {chan}',fontsize = 24)
    plt.xlabel('Temperature [K]',fontsize = 16)
    plt.ylabel('$P_{sat}$ [W]',fontsize = 16)
    plt.xlim([0.085,0.185])
    plt.ylim([0.9*np.min(Psat_fit*1e-12),1.1*np.max(Psat_fit*1e-12)])
    G = popt[0]*popt[1]*popt[2]**(popt[1]-1)
    textstr = '\n'.join((
        r'Fit to $P_{sat} = k(T_c^n - T_{bath}^n)$',
        r'$\kappa=%.2f$e-9' % (popt[0]/1e-9, ),
        r'$n=%.2f$' % (popt[1], ),
        r'$T_c=%.2f$' % (popt[2]/1e-3, ),
        r'$G = \kappa nT_c^{n-1}$ = %.2fe-10' % (G/1e-10, ) ))
    props = dict(boxstyle = 'round',facecolor = 'wheat',alpha = 0.5)
    ax = plt.gca()
    ax.text(0.65,0.95,textstr,transform=ax.transAxes,fontsize = 16,verticalalignment = 'top',bbox = props)
    plt.savefig('/home/cryo/Pictures/20200429_Fit_For_G_From_IV.png')
    plt.close()
