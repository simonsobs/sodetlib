import pysmurf.client
import pysmurf
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import scipy.signal as signal
from scipy import optimize
import scipy.optimize as opt
from scipy import fftpack
import argparse
import sys

#example: get_1hz.py -data /data/smurf_data/20210219/1613770613/outputs/xxxx.dat -out /xxxx/xxxx/xxx -plot true

S = pysmurf.client.SmurfControl(offline=True)
parser = argparse.ArgumentParser(description='taking target datafile')
parser.add_argument('-data',  nargs='?',
                    help='datafile path')

parser.add_argument('-out', nargs='?',
                    help='output path')

parser.add_argument('-plot', nargs='?',
                    help='whether plot each channel')

parser.add_argument('-chan', nargs='?',
                    help='which chan')

parser.add_argument('-band', nargs='?',
                    help='which band')

parser.add_argument('-freq', nargs='?',
                    help='which freq')

if len(sys.argv)==1:
    parser.print_help(sys.stderr)
    sys.exit(1)
args = parser.parse_args()
filename = args.data
output = args.out
chan_plot = args.plot
target_chan = args.chan
target_band = args.band
target_freq = float(args.freq)
print(filename)
print('plot',chan_plot)

def ctime_from_dat(iv_file):
    file=iv_file.split('/')[-1]
    ctime=file.split('.')[0]
    return ctime

ctime = ctime_from_dat(filename)


def noise_model(freq, wl, n, f_knee):
            """
            Crude model for noise modeling.
            Args
            ----
            wl : float
                White-noise level.
            n : float
                Exponent of 1/f^n component.
            f_knee : float
                Frequency at which white noise = 1/f^n component
            """
            A = wl*(f_knee**n)

            # The downsample filter is at the flux ramp frequency
            w, h = signal.freqz(filter_b, filter_a, worN=freq,
                fs=flux_ramp_freq)
            tf = np.absolute(h) # filter transfer function
#             print(A/(freq**n) + wl)
            return (A/(freq**n) + wl)*tf

timestamp, phase, mask, tes_bias = S.read_stream_data(filename,
            return_tes_bias=True)
bands, channels = np.where(mask!=-1)
S._pA_per_phi0 = 9000000.0
phase *= S._pA_per_phi0/(2.*np.pi)
flux_ramp_freq = S.get_flux_ramp_freq() * 1.0E3
filter_b = S.get_filter_b()
filter_a = S.get_filter_a()
downsample_factor = S.get_downsample_factor()
# flux ramp rate returns in kHz
fs = flux_ramp_freq/downsample_factor
downsample_freq, downsample_transfer = signal.freqz(filter_b,filter_a, worN=np.arange(.01, fs/2, .01), fs=flux_ramp_freq)
downsample_transfer = np.abs(downsample_transfer)
print()
nperseg=2**16
detrend='constant'

target_noise_level_list = []
target_noise_freq_list = []
# target_freq = 1
for c, (b, ch) in enumerate(zip(bands, channels)):
    if ch < 0:
        continue
    ch_idx = mask[b, ch]
    sampleNums = np.arange(len(phase[ch_idx]))
    t_array = sampleNums / fs
    f, Pxx = signal.welch(phase[ch_idx], nperseg=nperseg,
        fs=fs, detrend=detrend)
    Pxx = np.sqrt(Pxx)
    try:
        popt, pcov, f_fit, Pxx_fit = S.analyze_psd(f, Pxx,
                        fs=fs, flux_ramp_freq=flux_ramp_freq)
        wl, n, f_knee = popt
    except:
        continue

    freq_mask = ((np.abs(f) < target_freq + 0.1) & (np.abs(f) > target_freq - 0.1))
    range_peaks, _ = find_peaks(Pxx[freq_mask], height=500)
    f_range = f[freq_mask]
    Pxx_range = Pxx[freq_mask]
    peak_freq = np.float(f_range[Pxx_range.argmax()])
    peak_height = np.float(Pxx_range[Pxx_range.argmax()])
    
    target_peak_mask = (np.abs(f) == peak_freq)
    target_noise_level_list.append(peak_height)
    target_noise_freq_list.append(peak_freq)
    
    plt.figure(figsize=(12,8))
    plt.subplot(2, 1, 1)
    plt.plot(t_array,phase[ch_idx] - np.mean(phase[ch_idx]),'.-')
    plt.xlabel('Time [s]')
    plt.ylabel('Phase [pA]')
    plt.subplot(2, 1, 2)
    plt.loglog(f,Pxx,'.-')
    try:
        expect_target_noise = np.float(noise_model([peak_freq],wl,n,f_knee))
        plt.plot(f_fit,wl + np.zeros(len(f_fit)), linestyle=':',label=r'$\mathrm{wl} = $'+ f'{wl:0.2f},' +
                                r'$\mathrm{pA}/\sqrt{\mathrm{Hz}}$')
        plt.plot(f_fit, Pxx_fit, linestyle='--')
        plt.plot(peak_freq,expect_target_noise,'o',label = r'$\mathrm{Expected\ from\ fit\ }  $'+ f'{peak_freq:0.2f}'+r'$\mathrm{Hz} = $'+ f'{expect_target_noise:0.2f}' +
                            r'$\mathrm{pA}/\sqrt{\mathrm{Hz}}$',color = 'r')
    except:
        continue
    plt.plot(f[target_peak_mask],Pxx[target_peak_mask],'x',label = r'$\mathrm{Measured\ } $'+ f'{peak_freq:0.2f}'+r'$\mathrm{Hz} = $'+ f'{peak_height:0.2f}' +
                            r'$\mathrm{pA}/\sqrt{\mathrm{Hz}}$',color = 'r')
    
    plt.legend()
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Amp [pA/rtHz]')
    plt.savefig('{}/{}_band_{}_chan_{}_{}Hz.png'.format(output,ctime,b,ch,target_freq),bbox_inches='tight', transparent=False,facecolor=(1,1,1,1))
    print(target_freq)



# plt.figure(figsize=(12,8))
# plt.suptitle('1Hz pick up')
# plt.subplot(2, 1, 1)
# plt.hist(target_noise_freq_list, color = 'g', edgecolor='gray', alpha=0.3,range = (0,2), bins=100);
# med_freq = np.median(target_noise_freq_list)
# plt.axvline(med_freq, label = '$\mathrm{median\ peak\ freq\ }\ \mathrm{%0.2f~Hz}$'% med_freq, color = 'r', linestyle='--')
# plt.grid(linestyle = '--')
# plt.legend(fontsize=14, loc=1)
# plt.ylabel('$\mathrm{Channel\ count}$')
# plt.xlabel('Frequency [Hz]')
# plt.tight_layout()
# plt.subplot(2, 1, 2)
# plt.hist(target_noise_level_list, color = 'b', edgecolor='gray', alpha=0.3, bins=30);
# med_level = np.median(target_noise_level_list)
# plt.axvline(med_level, label = r'$\mathrm{median\ amp\ }  $'+ f'{med_level:0.2f}' +
#                         r'$\mathrm{pA}/\sqrt{\mathrm{Hz}}$', color = 'r', linestyle='--')
# plt.grid(linestyle = '--')
# plt.legend(fontsize=14, loc=1)
# plt.ylabel('$\mathrm{Channel\ count}$')
# plt.xlabel('Amp [pA/rtHz]')

# plt.tight_layout()
# plt.savefig('{}/{}_1_Hz_summary.png'.format(output,ctime),bbox_inches='tight', transparent=False,facecolor=(1,1,1,1))


