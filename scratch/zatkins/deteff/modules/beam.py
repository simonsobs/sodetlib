#!/usr/bin/env python3

# author: zatkins
# desc: collection of classes and methods related to a horn or lenslet beam 

import numpy as np
from scipy.interpolate import interp1d
from scipy.constants import pi
import matplotlib.pyplot as plt

import pathlib
import csv
import re

import monte_carlo as mc

# repo data folder for loading
# TODO: this is basically a proxy for a full installation
fpath = pathlib.Path(__file__).parent.parent / 'data/beams'

# properties of beam files
fname_key = {
    'MF-F' : 'MF_SO_mag.csv',
    'UHF-F' : 'UHF_SO_mag.csv',
    'deprecated-lenslet': 'deprecated_lenslet.csv'
}

regex_key = {
    'MF-F': r"mag\(rEL3X\) \[V\] - Freq='(.+)GHz' Phi='(.+)deg'",
    'UHF-F': r"mag\(rEL3X\) \[V\] - Freq='(.+)GHz' Phi='(.+)deg'",
    'deprecated-lenslet': r"RealizedGainTotal \[\] - Freq='(.+)GHz' LensMachineAngle='90' LR='0.48' Phi='(.+)deg'"
}

kwargs_key = {
    'MF-F': dict(phi = 'both', mag = True, freq_mult = 1e9),
    'UHF-F': dict(phi = 'both', mag = True, freq_mult = 1e9),
    'deprecated-lenslet': dict(phi = 'both', mag = False, freq_mult = 1e9)
}

# static methods
def sincos(x):
    return np.sin(x)*np.cos(x)

class Beam:
    '''Beam objects load a simulated feedhorn or lenslet beam and calculate the total solid angle of the
    beam via monte carlo integration. 
    
    Notes
    -----
    Intermediate calculations -- such as the monte carlo samples over
    theta -- are stored as instance variables to be used later on by Detector objects.
    '''

    def __init__(self, name, N = int(4e5), kind = 'linear', mc_dist = mc.beta_uniform(), \
        correlated = True, det_cos_factor = False, **kwargs):
        '''A Beam instance, with metadata loaded according to specified name, and statistical properties
        given by the passed monte carlo integration random variable object.

        Parameters
        ----------
        name : str
            The key associated with the desired metadata, e.g. 'MF-F' for MF feedhorn.
        N : int, optional
            The number of samples to draw over theta, for each frequency in the beam simulation,
            by default int(4e5)
        kind : str, optional
            The type of linear interpolation to use for representing the beam(theta; freq) functions,
            by default 'linear'. Passed as the 'kind' kwarg to scipy.interpolate.interp1d
        mc_dist : scipy.stats.rv_continuous or monte_carlo.mixture object, optional
            A statistical sampler supporting at least the scipy.stats.rv_continuous pdf() and rvs() methods,
            by default mc.beta_uniform()
        correlated: bool, optional
            Whether to NOT draw new samples for the beam statistics to be used later on by Detector objects,
            by default True
        det_cos_factor : bool, optional
            Whether to apply an additional cos(theta) factor to the beam(theta) function passed to
            detectors, by default False

        Notes
        -----
        More accurate beam statistics can be achieved by specifying a higher-order 'kind' kwarg at the
        cost of speed. For densely-modeled beams (e.g. >= 1 data point/degree), the gains are marginal.

        If correlated = True (default), the Detector-instance beam statistics are simply copied 
        from the Beam-instance statistics, resulting in correlated errors. This improves computation 
        speed and results in a decreased error in ratios of Detector-instance and Beam-instance statistics,
        such as the Detector.fill_fracs. See the Detector object in detectors.py.

        det_cos_factor should only be True if one has a load shining on detectors that emits with a 
        "normal"/collimated flux unit vector. If det_cos_factor = True is supplied, the detector solid angles 
        and other intermediate products are statistically independent from the same products for the beam, 
        which is never convolved with a cos(theta) function.
        '''
        self.name = name
        self.fname = fname_key[name]
        self.regex = regex_key[name]
        
        self.N = N
        self.kind = kind
        self.mc_dist = mc_dist
        self.correlated = correlated
        self.cos = det_cos_factor

        default_kwargs = kwargs_key[name]
        self.phi = kwargs.get('phi', default_kwargs['phi'])
        self.mag = kwargs.get('mag', default_kwargs['mag'])
        self.freq_mult = kwargs.get('freq_mult', default_kwargs['freq_mult'])

        print('loading beam file')
        self.beams, self.thetas, self.freq_keys, self.freqs_SI = \
            self.init_beams(phi = self.phi, mag = self.mag, kind = self.kind)

        print('fitting sin-convolved beam fit params')
        self.beam_fit_params = self.init_fit_params()

        print('getting sin-weighted-pdf theta samples')
        self.beam_theta_samples = self.init_theta_samples()

        print('evaluating sin-convolved beam solid angles as a function of freq')
        self.sas, self.sas_err, self.beam_ys = self.init_solid_angles()

        if det_cos_factor: det_jacobian = sincos
        else: det_jacobian = np.sin

        det_fit_params = self.beam_fit_params
        det_theta_samples = self.beam_theta_samples
        det_ys = self.beam_ys

        if det_cos_factor:
            print(f'fitting new {det_jacobian.__name__}-convolved beam fit params (for detectors)')
            det_fit_params = self.init_fit_params(jacobian = det_jacobian)

        if not correlated or det_cos_factor:
            print(f'getting new {det_jacobian.__name__}-weighted-pdf theta samples (for detectors)')
            det_theta_samples = self.init_theta_samples(params_dict = det_fit_params)
        
            print(f'evaluating new {det_jacobian.__name__}-convolved beam solid angles \
                 as a function of freq (for detectors)')
            _, _, det_ys = self.init_solid_angles(jacobian = det_jacobian, \
                params_dict = det_fit_params, thetas_dict = det_theta_samples)


        self.det_fit_params = det_fit_params
        self.det_theta_samples = det_theta_samples
        self.det_ys = det_ys

    def _load_beams(self):
        '''A private function for loading beam simulation data from disk. Returns a layered dict
        with such that a user can recover the data point by supplying a freq, phi, and theta key, in order.

        Only uses self.name to look up necessary data.

        Returns
        -------
        dict
            A layered dict such that supplying freq, phi, and theta (as strings, in order) will give
            the data point from the on-disk file.
        '''
        beam_out = {}
        
        with open (fpath / self.fname) as csv_data:
            reader_data = csv.DictReader(csv_data)
            
            for row in reader_data:
                theta = row["Theta [deg]"]
                
                for col in row:
                    
                    if col != "Theta [deg]":
                        searcher = re.search(self.regex, col)
                        
                        if searcher is not None:
                            freq, phi = searcher.groups()
                            
                            if freq not in beam_out:
                                beam_out[freq] = {}
                            
                            if phi not in beam_out[freq]:
                                beam_out[freq][phi] = {}
                            
                            beam_out[freq][phi][theta] = float(row[col])

        return beam_out

    def _make_beam_fs(self, beam, kind = 'linear'):
        '''Returns a linear interpolant for the beam, with specified interpolation kind. Normalizes
        the beam by its power at the pole, i.e. where theta = 0.

        Parameters
        ----------
        beam : dict
            A dict of theta:beam(theta) pairs only, i.e. at one frequency. Theta data must be
            in degrees!
        kind : str, optional
            kwarg to pass to scipy.interpolate.interp1d fixing the order of interpolation, 
            by default 'linear'

        Returns
        -------
        function
            A python function which takes theta (in radians) as an argument and returns the
            normalized beam power at that angle.
        '''
        theta = np.array(list(beam.keys())).astype(float) * pi/180
        power = np.array(list(beam.values())).astype(float)
        interp =  interp1d(theta, power, kind = kind, fill_value = 0.0, bounds_error=False)
        interp_0 = interp(0)
        assert interp(0) > 0
        def to_return(x):
            return interp(x) / interp_0
        return to_return

    def init_beams(self, phi = 'both', mag = True, kind = 'linear'):
        '''A wrapper function to build beam(theta) power functions to be saved as instance variables for the
        object constructor.

        Parameters
        ----------
        phi : str, optional
            Which beam component to extract: '0' (E) or '90' (H) or 'both' (sqrt(E**2 + H**2)), by default 'both'
        mag : bool, optional
            Whether on-disk data given by EM-field magnitude, by default True. If True, the magnitude is 
            squared to get the beam power. If False, the data is assumed to be in units of EM power already.
        kind : str, optional
            kwarg to pass to scipy.interpolate.interp1d fixing the order of interpolation, 
            by default 'linear', by default 'linear'

        Returns
        -------
        4-tuple of dict, dict, array of str, array of float
            1. A dict containing the result of self._make_beam_fs, indexed by each frequency key in the on-disk data
            2. A dict containing the array of thetas in the on-disk data (in radians), indexed by each frequency key
            in the on-disk data
            3. An array of of the frequency keys in the on disk data, as strings
            4. An array of the SI values (in Hz) of those frequency keys, as floats
        '''
        beam_in = self._load_beams()
        beam_out = {}
        
        for freq in beam_in: 
            if phi == 'both':
                if mag:
                    beam_E = {k: v**2 for k, v in beam_in[freq]['0'].items()}
                    beam_H = {k: v**2 for k, v in beam_in[freq]['90'].items()}
                else:
                    beam_E = beam_in[freq]['0']
                    beam_H = beam_in[freq]['90']
                
                assert beam_E.keys() == beam_H.keys(), 'beam_E keys != beam_H keys'
                
                beam_out[freq] = {k: beam_E[k] + beam_H[k] for k in beam_E}
            
            else:
                if mag:
                    beam_out[freq] = {k: v**2 for k, v in beam_in[freq][phi].items()}
                else:
                    beam_out[freq] = beam_in[freq][phi]
                
        beams = {freq: self._make_beam_fs(beam_out[freq], kind = kind) for freq in beam_out}
        thetas = {freq: np.array(list(beam_out[freq].keys())).astype(float) * pi/180 for freq in beam_out}
        freq_keys = np.array(list(beam_out.keys()))
        freqs = freq_keys.astype(float) * self.freq_mult
        return beams, thetas, freq_keys, freqs

    def get_fit_params(self, freq, min = 0, max = pi/2, jacobian = np.sin):
        x = self.thetas[freq]
        x = x[np.logical_and(min <= x, x <= max)]
        y = self.beams[freq](x) * jacobian(x)
        popt, _ = self.mc_dist.fit_params(x, y)
        return popt

    def init_fit_params(self, *args, **kwargs):
        out = {}

        for freq in self.freq_keys:
            out[freq] = self.get_fit_params(freq, *args, **kwargs)

        return out

    def get_theta_sample(self, freq, size = None, params_dict = None):
        if size == None: size = self.N
        if params_dict == None: params_dict = self.beam_fit_params

        params = params_dict[freq][1:]
        assert len(params) == len(self.mc_dist.p0) - 1, \
            print(f'mc_dist expected {len(self.mc_dist.p0)-1} params, got {len(params)}')

        return self.mc_dist.rvs(*params, size = size)

    def init_theta_samples(self, *args, **kwargs):
        out = {}

        for freq in self.freq_keys:
            out[freq] = self.get_theta_sample(freq, *args, **kwargs)

        return out

    def get_solid_angle(self, freq, jacobian = np.sin, mask = False, params_dict = None, thetas_dict = None):
        if params_dict == None: params_dict = self.beam_fit_params
        if thetas_dict == None: thetas_dict = self.beam_theta_samples

        params = params_dict[freq][1:] # don't want to pass A, the overall multiplicative constant
        thetas = thetas_dict[freq]
        assert len(params) == len(self.mc_dist.p0) - 1, \
            print(f'mc_dist expected {len(self.mc_dist.p0)-1} params, got {len(params)}')
        
        y = 2*pi*self.beams[freq](thetas)*jacobian(thetas)/self.mc_dist.pdf(thetas, *params) #2pi for uniform phi
        est, err = mc.mc_integrate(y, mask = mask)
        return est, err, y

    def init_solid_angles(self, *args, **kwargs):
        out = np.zeros(len(self.beams))
        sig_out = np.zeros(len(out))
        y_dict = {}

        for i, freq in enumerate(self.freq_keys):
            out[i], sig_out[i], y_dict[freq] = self.get_solid_angle(freq, *args, **kwargs)

        return out, sig_out, y_dict

def test_integral(f = np.cos, max_theta = pi/4, slice_frac = 3/8, size = int(4e5), jacobian = np.sin):
    # get fit params
    x_fit = np.linspace(0, pi/2)
    y_fit = f(x_fit) * jacobian(x_fit)
    mc_dist = mc.beta_uniform()
    params, _ = mc_dist.fit_params(x_fit, y_fit)

    # get theta samples
    params = params[1:]
    theta = mc_dist.rvs(*params, size = size)

    # get phi samples
    phi = 2*pi * np.random.rand(size)

    # get mask
    mask = np.logical_or(theta > max_theta, phi/2/pi > slice_frac)

    # get integral
    y = 2*pi * f(theta) * jacobian(theta) / mc_dist.pdf(theta, *params)
    est, err = mc.mc_integrate(y, mask = mask)
    return est, err