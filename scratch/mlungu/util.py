import os
import time
import numpy as np
import matplotlib as mpl
import scipy.signal as signal
import scipy.optimize as opt

# Read extended ADC data

def read_adc_data(S, band, nsamp):
  bay = S.band_to_bay(band)
  adc = band%4

  S.setup_daq_mux('adc', adc, nsamp, band = band)

  timestamp = S.get_timestamp()
  datafn = os.path.join(S.output_dir, timestamp + '.dat')

  datafnbuffer = np.zeros(300, dtype = int)
  datafnbuffer[:len(datafn)] = [ord(c) for c in datafn]

  S.set_streamdatawriter_datafile(datafnbuffer)
  S.set_streamdatawriter_open('True')
  S.set_trigger_daq(bay, 1, write_log = True)

  S.get_waveform_end_addr(bay, engine=0)
  empty = False

  while not empty:
    empty = True
    time.sleep(1)

    for i in range(4):
      S.get_waveform_wr_addr(bay, engine=0)
      empty *= S.get_waveform_empty(bay, engine = i)

  S.set_streamdatawriter_open('False')

  return decode_adc_data(S,datafn)
  
# Decode ADC data from file

def decode_adc_data(S, datafn):
  header, rawdata = S.process_data(datafn)

  # Decode Strobes
  strobes = np.floor(rawdata / (2**30)).astype(np.uint32)
  strobe = np.remainder(strobes, 2)
  rawdata -= (2**30)*strobes

  # Decode data
  data = np.zeros(len(rawdata),dtype=complex)
  data.real = np.double(rawdata[:,0])
  data.imag = np.double(rawdata[:,1])
  data.real[rawdata[:,0] >= 2**29] -= 2**30
  data.imag[rawdata[:,1] >= 2**29] -= 2**30

  return data

# Test new ADC read function

def test_adc_read(S, freqs_mhz, drive):
  band = int((np.mean(freqs_mhz)-4000)/500)

  S.band_off(band)

  for freq in freqs_mhz:
    S.set_fixed_tone(freq, drive)

  mpl.pyplot.ion()

  ctime = S.get_timestamp()
  data = S.read_adc_data(band, data_length=2**19, save_data = False)

  fs = S.get_digitizer_frequency_mhz()*1E6
  freq, pspec = signal.welch(data, fs=fs, nperseg=len(data)/2, \
                             return_onesided = False, detrend = False)
  freq_mhz = freq/1E6+4250+band*500

  pspec = pspec[np.argsort(freq_mhz)]
  freq_mhz = freq_mhz[np.argsort(freq_mhz)]

  mpl.pyplot.plot(freq_mhz,pspec,'b')

  data = read_adc_data(S,band, 2**19)

  S.band_off(band)

  freq, pspec = signal.welch(data, fs=fs, nperseg=len(data)/2, \
                             return_onesided = False, detrend = False)
  freq_mhz = freq/1E6+4250+band*500

  pspec = pspec[np.argsort(freq_mhz)]
  freq_mhz = freq_mhz[np.argsort(freq_mhz)]

  mpl.pyplot.plot(freq_mhz,pspec,'r')

  mpl.pyplot.grid()
  mpl.pyplot.yscale('log', nonposy = 'clip')
  mpl.pyplot.xlim(4250+band*500-fs*1E-6/2,4250+band*500+fs*1E-6/2)
  mpl.pyplot.xlabel('Frequency (MHz)', fontsize = 16)
  mpl.pyplot.ylabel('Power (ADC Units)', fontsize = 16)
  mpl.pyplot.title('Band %d (Drive = %d): %s' %(band,drive,ctime), \
                   fontsize = 20)
  mpl.pyplot.gca().tick_params(labelsize = 16)
  mpl.pyplot.tight_layout()

# Generate multi-tone file

def generate_tone_file(S, freqs_mhz, drive):
  fs = S.get_digitizer_frequency_mhz()*1E6
  band = int((np.mean(freqs_mhz)-4000)/500)

  nsamp = 2**14

  t = 1/fs*np.arange(nsamp)
  d = np.zeros(nsamp).astype(complex)

  maxt = 1/fs*nsamp

  for freq in freqs_mhz:
    freq -= 4250+band*500
    fif = round(maxt*freq*1E6)/maxt
    phase = 2*np.pi*np.random.rand()
    d += np.exp(1j*(2*np.pi*fif*t + phase))

  d *= (2**15-1)/np.max(abs(d))
  d = np.round(np.vstack([d.real,d.imag]).T)
  np.savetxt(S.output_dir+'/test.csv', d, delimiter = ',', fmt = '%6d')

# Test playing multi-tone file

def test_multi_tone(S, freqs_mhz, drive):
  band = int((np.mean(freqs_mhz)-4000)/500)

  mpl.pyplot.ion()

  ncolors = ['g','m','c','y']
  norders = ['Third','Fifth','Seventh','Ninth']
  nlabels = ['Primary'] + norders

  fmin = min(freqs_mhz)-4.5*abs(np.diff(freqs_mhz)[0])
  fmax = max(freqs_mhz)+4.5*abs(np.diff(freqs_mhz)[0])

  adc2mW = 10**(-3.3)/1.2389E6
  drive2dBm = { 1:-61.4,  2:-57.6,  3:-54.9,  4:-51.3,  5:-48.7,  6:-45.1, \
                7:-42.6,  8:-39.0,  9:-36.5, 10:-33.0, 11:-30.5, 12:-27.0, \
               13:-24.5, 14:-21.0, 15:-18.5}

  S.band_off(band)

  print()

  S.play_tone_file(band,S.output_dir+'/test.csv')

  ctime = S.get_timestamp()
  data = S.read_adc_data(band, data_length=2**19, save_data = False)

  S.stop_tone_file(band)

  S.band_off(band)

  fs = S.get_digitizer_frequency_mhz()*1E6
  freq, pspec = signal.welch(data, fs = fs, nperseg = len(data), \
                             return_onesided = False, detrend = False, \
                             window = 'flattop', scaling = 'spectrum')
  freq = freq/1E6+4250+band*500
  pspec = 10*np.log10(adc2mW*pspec)

  pspec = pspec[np.argsort(freq)]
  freq = freq[np.argsort(freq)]

  primary = np.zeros(len(freq)).astype(bool)
  third = np.zeros(len(freq)).astype(bool)

  print()
  print('Drive Level = %d (%.1f dBm)'  %(drive,drive2dBm[drive]))
  print('='*60)

  nlines = []

  mask = np.ones(len(pspec)).astype(bool)
  fmask = np.ones(len(pspec)).astype(bool)

  for f in freqs_mhz:
    primary = (freq > f-0.02)*(freq < f+0.02)
    mask *= ~primary
    fmask *= (freq < f-0.5*abs(np.diff(freqs_mhz))) + \
             (freq > f+0.5*abs(np.diff(freqs_mhz)))

    nlines.append(mpl.pyplot.plot(freq[primary],pspec[primary],'r')[0])

    print('Primary Tone at %.1f MHz: %.1f dBm' %(f,pspec[primary].max()))

  for i in range(2,6):
    fn = (i*min(freqs_mhz)-(i-1)*max(freqs_mhz), \
          i*max(freqs_mhz)-(i-1)*min(freqs_mhz))

    print()

    for f in fn:
      higher = (freq > f-0.02)*(freq < f+0.02)
      mask *= ~higher
      fmask *= (freq < f-0.1*abs(np.diff(freqs_mhz))) + \
               (freq > f+0.1*abs(np.diff(freqs_mhz)))

      nlines.append(mpl.pyplot.plot(freq[higher], pspec[higher], \
                                    ncolors[i-2])[0])

      print('%s-Order Product at %.1f MHz: %.1f dBm' \
             %(norders[i-2],f,pspec[higher].max()))

  fmask = mask*(freq > fmin)*(freq < fmax)
  noise = np.median(pspec[fmask])

  print()
  print('Median Noise Level %.1f - %.1f MHZ: %.1f dBm' %(fmin,fmax,noise))

  print('='*60)
  print()

  mpl.pyplot.plot(freq[mask],pspec[mask],'b')

  mpl.pyplot.grid()
  #mpl.pyplot.xlim(fmin,fmax)
  #mpl.pyplot.ylim(-140,0)
  mpl.pyplot.ylim(-120,0)
  mpl.pyplot.xlabel('Frequency (MHz)', fontsize = 16)
  mpl.pyplot.ylabel('Power (dBm)', fontsize = 16)
  mpl.pyplot.title('Band %d (Drive = %d): %s' %(band,drive,ctime), \
                   fontsize = 20)
  mpl.pyplot.gca().tick_params(labelsize = 16)
  mpl.pyplot.legend(nlines[::2], nlabels, fontsize = 12, loc = 'upper right')
  mpl.pyplot.tight_layout()
  mpl.pyplot.show()
  #mpl.pyplot.savefig(S.plot_dir+'/%s_adc%d_%02d.png' %(ctime,band,d))
  #mpl.pyplot.close(mpl.pyplot.gcf())

# Lorentzina Peak Model

def lorentzian(f,p,c = None):
  pm = np.repeat(np.nan,3)

  if 'A' in c.keys():
    pm[0] = c['A']
  if 'w' in c.keys():
    pm[1] = c['w']
  if 'f0' in c.keys():
    pm[2] = c['f0']

  pm[np.isnan(pm)] = p

  return pm[0]/(1+((pm[2]-f)/(p[1]/2.0))**2)

# Plot an IP3 measuement

def plot_ip3(S,band,Pi,P1o,P3o):
  ctime = S.get_timestamp()

  mpl.pyplot.plot(Pi, P1o, 'ob', label = 'Primary', ms = 10)
  mpl.pyplot.plot(Pi, P3o, 'og', label = 'Third', ms = 10)

  mpl.pyplot.grid()
  mpl.pyplot.ylim(-100,0)
  mpl.pyplot.xlabel('Input Power (dBm)', fontsize = 16)
  mpl.pyplot.ylabel('Output Power (dBm)', fontsize = 16)
  mpl.pyplot.title('IP3 (Band %d): %s' %(band,ctime), fontsize = 20)
  mpl.pyplot.gca().tick_params(labelsize = 16)
  mpl.pyplot.legend(numpoints = 2, fontsize = 14, loc = 'upper left')
  mpl.pyplot.tight_layout()
  mpl.pyplot.savefig(S.plot_dir+'/%s_adc%d_ip3.png' %(ctime,band))
  mpl.pyplot.close(mpl.pyplot.gcf())

# Measure IP3 using SMuRF

def measure_ip3(S, freqs_mhz, drive):
  band = int((np.mean(freqs_mhz)-4000)/500)

  mpl.pyplot.ioff()
  
  ncolors = ['g','m','c','y']
  norders = ['Third','Fifth','Seventh','Ninth']
  nlabels = ['Primary'] + norders

  fmin = min(freqs_mhz)-4.5*abs(np.diff(freqs_mhz)[0])
  fmax = max(freqs_mhz)+4.5*abs(np.diff(freqs_mhz)[0])

  adc2mW = 10**(-3.3)/1.2389E6
  drive2dBm = { 1:-61.4,  2:-57.6,  3:-54.9,  4:-51.3,  5:-48.7,  6:-45.1, \
                7:-42.6,  8:-39.0,  9:-36.5, 10:-33.0, 11:-30.5, 12:-27.0, \
               13:-24.5, 14:-21.0, 15:-18.5}

  if type(drive) not in [list,tuple,range,np.ndarray]:
    drive = [drive]

  tone1 = np.array([])
  tone3 = np.array([])

  for d in drive:
    S.band_off(band)

    print()

    for freq in freqs_mhz:
      S.set_fixed_tone(freq, d)

    ctime = S.get_timestamp()
    data = S.read_adc_data(band, data_length=2**19, save_data = True)

    S.band_off(band)

    fs = S.get_digitizer_frequency_mhz()*1E6
    freq, pspec = signal.welch(data, fs = fs, nperseg = len(data), \
                               return_onesided = False, detrend = False, \
                               window = 'flattop', scaling = 'spectrum')
    freq = freq/1E6+4250+band*500
    pspec = 10*np.log10(adc2mW*pspec)

    pspec = pspec[np.argsort(freq)]
    freq = freq[np.argsort(freq)]

    primary = np.zeros(len(freq)).astype(bool)
    third = np.zeros(len(freq)).astype(bool)

    print()
    print('Drive Level = %d (%.1f dBm)'  %(d,drive2dBm[d]))
    print('='*60)

    nlines = []

    mask = np.ones(len(pspec)).astype(bool)
    fmask = np.ones(len(pspec)).astype(bool)

    for f in freqs_mhz:
      primary = (freq > f-0.02)*(freq < f+0.02)
      mask *= ~primary
      fmask *= (freq < f-0.5*abs(np.diff(freqs_mhz))) + \
               (freq > f+0.5*abs(np.diff(freqs_mhz)))

      nlines.append(mpl.pyplot.plot(freq[primary],pspec[primary],'r')[0])

      tone1 = np.append(tone1,pspec[primary].max())
      print('Primary Tone at %.1f MHz: %.1f dBm' %(f,pspec[primary].max()))

    for i in range(2,6):
      fn = (i*min(freqs_mhz)-(i-1)*max(freqs_mhz), \
            i*max(freqs_mhz)-(i-1)*min(freqs_mhz))

      print()

      for f in fn:
        higher = (freq > f-0.02)*(freq < f+0.02)
        mask *= ~higher
        fmask *= (freq < f-0.1*abs(np.diff(freqs_mhz))) + \
                 (freq > f+0.1*abs(np.diff(freqs_mhz)))

        nlines.append(mpl.pyplot.plot(freq[higher], pspec[higher], \
                                      ncolors[i-2])[0])

        if i == 2:
          tone3 = np.append(tone3,pspec[higher].max())

        print('%s-Order Product at %.1f MHz: %.1f dBm' \
               %(norders[i-2],f,pspec[higher].max()))

    fmask = mask*(freq > fmin)*(freq < fmax)
    noise = np.median(pspec[fmask])

    print()
    print('Median Noise Level %.1f - %.1f MHZ: %.1f dBm' %(fmin,fmax,noise))

    print('='*60)
    print()

    mpl.pyplot.plot(freq[mask],pspec[mask],'b')

    mpl.pyplot.grid()
    mpl.pyplot.xlim(fmin,fmax)
    mpl.pyplot.ylim(-140,0)
    mpl.pyplot.xlabel('Frequency (MHz)', fontsize = 16)
    mpl.pyplot.ylabel('Power (dBm)', fontsize = 16)
    mpl.pyplot.title('Band %d (Drive = %d): %s' %(band,d,ctime), fontsize = 20)
    mpl.pyplot.gca().tick_params(labelsize = 16)
    mpl.pyplot.legend(nlines[::2], nlabels, fontsize = 12, loc = 'upper right')
    mpl.pyplot.tight_layout()
    mpl.pyplot.savefig(S.plot_dir+'/%s_adc%d_%02d.png' %(ctime,band,d))
    mpl.pyplot.close(mpl.pyplot.gcf())

  Pin = np.array([drive2dBm[d] for d in drive])
  tone1 = (tone1[:-1:2]+tone1[1::2])/2.0
  tone3 = (tone3[:-1:2]+tone3[1::2])/2.0

  if len(drive) > 1:
    plot_ip3(S,band,Pin,tone1,tone3)

  return Pin,tone1,tone3
