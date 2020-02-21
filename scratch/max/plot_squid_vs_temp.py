import sys
sys.path.append('/home/cryo/docker/pysmurf/dev/pysmurf/pysmurf')
import squid_fit as sf
import numpy as np
import matplotlib.pylab as plt
import os
from scipy.optimize import curve_fit
from matplotlib.ticker import MaxNLocator

main_dir='/data/smurf_data/20191205'

ctimes = ['1575509988','1575510308','1575510632','1575510958','1575511283','1575511605','1575511996','1575512404','1575512799','1575513193','1575513587','1575513909','1575514236','1575514561','1575514886','1575515211','1575515535','1575515866']

band = 2

outdirname=main_dir + '/squid_curves_vs_temp'
if not os.path.exists(outdirname):
    os.mkdir(outdirname)

Temps = np.arange(60,420,20)

plot_dict = {}
for i, ctime in enumerate(ctimes):
    plot_dict[Temps[i]] = {}
    datafn = main_dir + '/' + ctime + '/outputs/' + ctime + '_fr_sweep_data.npy'
    frdata=np.load(datafn)
    frdata=frdata.item()
    if i == 0:
        plot_dict['bias'] = frdata['bias']
    channels = frdata[band][(None,None)]['channels']
    for j, chan in enumerate(channels):
        plot_dict[Temps[i]][chan] = frdata[band][(None,None)]['fvsfr'][j]

for chan in channels:
    z = np.zeros((len(Temps),len(plot_dict['bias'])))
    for t in range(len(Temps)):
        z[t,:] = plot_dict[Temps[t]][chan]
    x,y = np.meshgrid(Temps,plot_dict['bias'])
    plt.figure()
    levels = MaxNLocator(nbins = 50).tick_values(z.min(),z.max())
    plt.contourf(x,y,np.transpose(z),levels = levels)
    plt.colorbar(label = 'Frequency Shift MHz')
    plt.xlabel('Temperature [mK]')
    plt.ylabel('Flux Bias')
    plt.savefig(outdirname + '/band' + str(band) + '_chan' + str(chan)+'_contourplot.png')
    plt.close()
    plt.plot(plot_dict['bias'],plot_dict[100][chan])
    plt.title('Cut at 100mK for Channel '+str(chan))
    plt.xlabel('Flux Bias')
    plt.ylabel('Frequency Shift MHz')
    plt.savefig(outdirname + '/band' + str(band) + '_chan' + str(chan)+'_TempCut100mK.png')
    plt.close()



