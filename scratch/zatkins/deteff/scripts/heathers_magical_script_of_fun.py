#!/usr/bin/env python3
# author: zatkins
# a quick script to demonstrate getting some optical powers from setup files
# ASSUMES 100% optical efficiency -- multiply powers by your expected efficiency!!

import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

import sys
sys.path.append('/home/zatkins/repos/simonsobs/sodetlib/scratch/zatkins/deteff/modules')#('/path/to/deteff/modules')
import detector as det

# get parameters of setup
setupname = 'M_D_016b'       # folder holds setup files (cold load, filters, array orientation)
assem = 'Cv4'               # name of the assembly within that setup folder

# get parameters for doing all the angular integrals, and 
# integrals over frequency
beam_N = int(1e4)       # number of monte carlo points to integrate the full beam

beam_phi = 'both'       # global polarization of the beam for each feedhorn. 'both' is a concentric average beam, can
                        # also do '0' or '90'. better code would do the correct polarization for every bolometer 

det_N = int(2e3)        # number of monte carlo points to integrate the cold-load-stopped beam for every feedhorn
                        
                        # some kwargs to help do integrals over frequency (setting limits by band, defining
get_power_kwargs = {    # which frequency is the low band ('1') and high band ('2'))
    90: {'band': '1', 'min': 70e9, 'max': 120e9},
    150: {'band': '2', 'min': 120e9, 'max': 180e9},
} 

# get assembly object
#
# This reads the geometry from the setup parameters to precompute
# the angular integrals for each frequency for each feedhorn.
# pixel_ids=None will precomute terms for *every* feedhorn
a = det.Assembly(setupname, assem, pixel_ids=None, \
    beam_kwargs=dict(N = beam_N, phi = beam_phi), pixel_kwargs=dict(N = det_N))

# get a particular pixel object
#
# Pixel objects support calculating incident optical power as function of temperature, given
# the band ('1' or '2'), and the limits of integration, which depend on band.
# There are also other methods for plotting or optical efficiency which you don't need here,
# but feel free to try them.
# 
# pixel_ids look like (rhombus, row, column), e.g. ('A', 1, 10),
# see http://phy-stslab.princeton.edu/filepull.php?fileid=376 for how this maps, specifically
# columns DTGoupsection, DTPixelrow, DTPixelcolumn
my_pixel_id = ('A', 1, 10)
pix = a.pixels[my_pixel_id]
temperatures = np.arange(8, 16)

print(f'plotting power vs. temp for {my_pixel_id}')
power, power_err = pix.get_power(temperatures, **get_power_kwargs[90]) # do the integral for a 90GHz detector
plt.errorbar(temperatures, power, yerr=power_err, fmt = '.')
plt.show()

print(f'These are my powers:', power)
print(f'These are my monte carlo integration errors:', power_err)

# Can also sanity check the orientation of the objects in the cryostat
print(f'plotting the apertures, cold load for {my_pixel_id}')
pix.plot_apertures()

# a.pixels is a dict of Pixel objects, so you can iterate over all of them if you want,
# or just give whichever you're interested in.
# The Assembly object also supports doing some things in bulk -- just one temperature 
# at a time, though. This can take a minute, since there are so many horns.
T = 20

print(f'plotting power across wafer at temp of {T}K')
a.plot_power(T, 'wafer_scatter', get_power_kwargs=get_power_kwargs[90]) # look at 90GHz detectors



