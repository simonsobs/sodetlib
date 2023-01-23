from sodetlib.legacy.analysis import squid_fit as sf
import numpy as np
import matplotlib.pylab as plt
import os
from scipy.optimize import curve_fit
import sys
from optparse import OptionParser

#Script written by SWH: shawn@slac.stanford.edu
#Example usage: 
#python3 -i fit_squid_data.py -rg1 /data/smurf_data/20201009/1602222586/outputs/1602222586_fr_sweep_data.npy -d -e 0.55

parser = OptionParser()
parser.add_option("-g", "--plot-guess", action="store_true",
                  dest="plot_guess", help="Include to plot initial guess.  " +
                  "Does not plot initial guess by default.",
                  default=False)
parser.add_option("-d", "--debug", action="store_true",
                  dest="debug", help="Debug mode; extra plotting and " +
                  "printout to help diagnose problems.  Includes the " +
                  "-r1g plotting options.",
                  default=False)
parser.add_option("-1", "--plot-first-harmonic", action="store_true",
                  dest="plot_first_harmonic", help="Include to overplot " +
                  "fitted first harmonic.  Does not plot first harmonic " +
                  "of fit by default.",
                  default=False)
parser.add_option("-r", "--plot-residuals", action="store_true",
                  dest="plot_residuals", help="Include to overplot " +
                  "fit residuals.  Does not plot fit residuals by " +
                  "default.",
                  default=False)
parser.add_option("-u", "--phi0-units", action="store_true",
                  dest="phi0_units", help="Plot x-axes in phi0 units.  " +
                  "Does not plot in phi0 units by default.",
                  default=False)
parser.add_option("-n", "--nharmonics", action="store",
                  dest="nharmonics", help="Total number of harmonics to include in fit.",
                  type='int',
                  default=7)
parser.add_option("-e", "--extend-frac", action="store",
                  dest="extend_frac", help="Increases y-span of plot " +
                  "by +/- the specified fraction times the peak to " +
                  "peak default plotting y-span.  Gives additional " +
                  "room so that annotation boxes with legend and fit " +
                  "results don't obscure the plots.  Default is 0.5.",
                  type='float',
                  default=0.5)

(options, args) = parser.parse_args()

# enable all optional plotting in debug mode
if options.debug:
    options.plot_first_harmonic = True
    options.plot_residuals = True
    options.plot_guess = True

if len(args) != 1:
    parser.error("Must provide a data file to analyze.")

# path to data file
datafp=args[0]
# name of data file (without path)
datafn=os.path.basename(args[0])
# data directory
datadir=os.path.dirname(args[0])

if not os.path.exists(datafp):
    parser.error(f"{datafp} does not exist on disk.")

# load the data
frdata=np.load(datafp,allow_pickle=True)
frdata=frdata.item()

# what bands are present in the files?
bands=sorted([k for k in list(frdata.keys()) if k != 'bias'])
    
outdirname=datafn.split('.')[0]
outdirname+='_fits'

# right now, puts results directory in same directory as data
#outdir=outdirname
outdir = os.path.join(datadir.split('outputs')[0]+'plots',outdirname)
if not os.path.exists(outdir):
    os.mkdir(outdir)

resultsoutfilename=outdirname+'.dat'
resultsoutfile=open(os.path.join(outdir,resultsoutfilename),'w')
fmt_str='{0[0]:<15}{0[1]:<15}{0[2]:<15}{0[3]:<20}{0[4]:<20}{0[5]:<15}{0[6]:<15}{0[7]:<15}{0[8]:<15}{0[9]:<25}\n'
hdr=fmt_str.format(['band', 'channel', 'fres_mhz', 'max_fit_fres_mhz',
                    'min_fit_fres_mhz', 'phi0_ff', 'phi0_offset',
                    'df_khz', 'hhpwr', 'avg_dfdphi_Hzperuphi0'])
resultsoutfile.write(hdr)
resultsoutfile.flush()

for band in bands:
    #channels = frdata[band][(None,None)]['channels']
    channels=frdata[band]['channels']
    for idx in range(len(channels)):

        ch=channels[idx]
        if options.debug:
            print(f'* b{band}ch{ch}')

        squid_data=frdata[band]['fvsfr'][idx]
        fres=frdata[band]['fres'][idx]        
        bias=frdata['bias']
        #squid_data=frdata[band][(None,None)]['fvsfr'][idx]
        #fres=frdata[band][(None,None)]['fres'][idx]        

        est=sf.estimate_fit_parameters(
            bias,squid_data,nharmonics_to_estimate=options.nharmonics,debug=options.debug)

        plt.figure()

        #fit
        popt, pcov = curve_fit(sf.model, bias, squid_data, p0=est)
        fit_curve=sf.model(bias,*popt)
        # some fit curve statistics
        fit_curve_span=np.max(fit_curve)-np.min(fit_curve)        
        fit_curve_center=fit_curve_span/2.+np.min(fit_curve)        

        ## fit results
        # popt[0] is phi0 in radians
        # popt[1] is the phi0 offset, in units of radians
        # popt[2] is the DC offset
        # popt[3] is 2x the amplitude of the 1st harmonic
        # and for N>3
        #   popt[N] is 2x the amplitude of (N-2)th harmonic
        phi0_ff=popt[0]
        phi0_offset_ff=popt[1]

        xvals=bias
        if options.phi0_units:
            xvals=bias/phi0_ff
        
        plt.plot(
            xvals,fit_curve,'r--',lw=2,
            label=f'fit (nharmonics={options.nharmonics})')
        
        plt.plot(xvals,squid_data,
                 label=f'rf-SQUID curve ch{ch}',
                 lw=2,c='k')
        
        if options.plot_guess:
            plt.plot(xvals,sf.model(bias,*est),
                     'm--',label='guess')

        # evaluate model over just one cycle
        phi_over_one_cycle=np.linspace(0,phi0_ff,10000)
        phi_over_one_cycle+=phi0_offset_ff
        fit_curve_over_one_cycle=sf.model(phi_over_one_cycle,*popt)        
        phi_over_one_cycle/=phi0_ff
        # only useful to plot to make sure it's right.  Plot will only look
        # right if plot_in_phi0=True
        #plt.plot(phi_over_one_cycle,fit_curve_over_one_cycle,'o',lw=4,label='one cycle')
        # multiply by 1e6 to convert from MHz->Hz but then by 1e-6 to convert from per phi0
        # to per uphi0
        avg_dfdphi_Hzperuphi0=np.nanmean(np.abs(np.gradient(fit_curve_over_one_cycle))/(np.gradient(phi_over_one_cycle)))
        
        popt1=[0]*(options.nharmonics+2)
        # phi0, phi0_offset, and dc_offset
        popt1[:3]=popt[:3]
        # first harmonic
        popt1[3]=popt[3]
        # second harmonic
        #popt1[4]=popt[4]

        harmonic_amplitudes=popt[3:]
        # takes into account fact that odd terms do not contribute to
        # peak to peak amplitude
        amplitude_sum=np.sum(harmonic_amplitudes*np.array([(1+(-1)**n)/2. for
                             n in range(len(harmonic_amplitudes))]))
        hhpwr=( ( np.square(popt[3]) )/( np.sum(np.square(popt[4:])) ) )**-1
        
        fit_curve1=sf.model(bias,*popt1)
        # some first harmonic fit curve statistics
        fit_curve1_span=np.max(fit_curve1)-np.min(fit_curve1)        
        fit_curve1_center=fit_curve1_span/2.+np.min(fit_curve1)                    

        # compute average sensitivity contributed by only the first harmonic.
        fit_curve1_over_one_cycle=sf.model(phi_over_one_cycle*phi0_ff,*popt1)
        avg_dfdphi1_Hzperuphi0=np.nanmean(np.abs(np.gradient(fit_curve1_over_one_cycle))/(np.gradient(phi_over_one_cycle)))
        # end attempt to estimate sensitivity from first harmonic alone

        if options.debug:
            print(f'-> avg_dfdphi1_Hzperuphi0={avg_dfdphi1_Hzperuphi0}')
            print(f'-> avg_dfdphi_Hzperuphi0={avg_dfdphi_Hzperuphi0}')
        
        if options.plot_first_harmonic:

            # may need to investigate why the centering step is necessary.           
            plt.plot(xvals,fit_curve1+(fit_curve_center-fit_curve1_center),'c--',lw=1,label='first harmonic fit')

        if options.plot_residuals:
            # residuals - add median so it falls on curve
            yoffset_for_plotting_residuals=(np.max(squid_data)+np.min(squid_data))/2.
            residuals=(squid_data-fit_curve)
            #plt.plot(bias,residuals*10.+yoffset_for_plotting_residuals,'k--',label='residuals x 10')
            plt.plot(xvals,residuals+yoffset_for_plotting_residuals,color='gray',ls='--',label='residuals')

        plt.title(datafn)
        plt.ylabel('Resonator Frequency (MHz)')
        if options.phi0_units:
            plt.xlabel('Flux ramp flux (Phi0)')                        
        else:
            plt.xlabel('Flux ramp flux (fraction full scale x2)')                        

        plt.legend(loc='upper right',fontsize=8)
        figoutfilename='%s_frfit_b%dch%d.png'%(datafn.split('_')[0],band,ch)
        ax=plt.gca()

        ymin,ymax=ax.get_ylim()
        yspan=(ymax-ymin)
        plt.ylim(ymin-yspan*options.extend_frac,ymax+yspan*options.extend_frac)       
        
        df_khz=(np.max(fit_curve)-np.min(fit_curve))*1000.
        fitresulttxt=(
            f'fres = {fres:.1f} MHz\n' +
            f'Phi0 = {phi0_ff:.3f} ff\n' +
            f'Phi_offset = {phi0_offset_ff/phi0_ff:.3f} Phi0\n' +
            f'df = {df_khz:.1f} kHz\n' +
            f'<dfdphi> = {avg_dfdphi_Hzperuphi0:.3f} Hz/uPhi0\n' +
            f'hhpwr = {hhpwr:.3f}')
        plt.text(0.0175,0.975,fitresulttxt,horizontalalignment='left',
                 verticalalignment='top',transform=ax.transAxes,fontsize=10,bbox=dict(facecolor='white',alpha=0.9))

        # also print first harmonic sensitivity fraction
        fitresulttxt=(
            f'$<dfdphi_{1}>$ = {avg_dfdphi1_Hzperuphi0:.3f} Hz/uPhi0\n' +
            f'$<dfdphi_{1}>/<dfdphi>$ = {100*avg_dfdphi1_Hzperuphi0/avg_dfdphi_Hzperuphi0:.0f}%')
        plt.text(0.98,0.125,fitresulttxt,horizontalalignment='right',
                 verticalalignment='top',transform=ax.transAxes,fontsize=10,bbox=dict(facecolor='white',alpha=0.9))        
        
        plt.savefig(os.path.join(outdir,figoutfilename))

        chisq=np.power((squid_data-fit_curve),2)/fit_curve
        
        plt.plot(
            xvals,squid_data,label=f'rf-SQUID curve ch{ch}',
            lw=2,c='k')        
        plt.tight_layout()
        
        max_fit_fres_mhz=np.max(fit_curve)
        min_fit_fres_mhz=np.min(fit_curve)
        fitresult=fmt_str.format([
            str(band),
            str(ch),
            f'{fres:.4f}',
            f'{max_fit_fres_mhz:.4f}',
            f'{min_fit_fres_mhz:.4f}',
            f'{phi0_ff:.4f}',
            f'{phi0_offset_ff/phi0_ff:.4f}',
            f'{df_khz:.4f}',
            f'{hhpwr:.4e}',
            f'{avg_dfdphi_Hzperuphi0:4e}'
        ]) 
        resultsoutfile.write(fitresult)
        resultsoutfile.flush()

        # next channel
        if options.debug:
            print(f'-------------------------')

resultsoutfile.close()
