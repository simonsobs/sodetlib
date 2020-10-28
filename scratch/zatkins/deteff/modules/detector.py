#!/usr/bin/env python3

# author: zatkins
# desc: collection of classes and methods related to a detector optical efficiency experimental setup.
# imports callables from beam.py and solid_angle_.py

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.path as path
import matplotlib.patches as patches
from matplotlib.ticker import FuncFormatter

from scipy.stats import uniform
from scipy.interpolate import interp1d
from scipy.constants import pi, c, h, k

import pathlib
import copy

import monte_carlo as mc
from beam import Beam
import efficiency as eff

import time

freq_mult_key = {
    'W1223_scaled.txt': 100*c,
    'ACTPol_X.txt': 100*c,
    'K1706.txt': 100*c,
    'K1674.txt': 100*c,
    'W1180.txt': 100*c,
    'W1319_scaled.txt': 100*c,
    'W1236.txt': 100*c,
    'MF_1.txt': 1e9,
    'MF_2.txt': 1e9
}

# repo data folder for loading
# TODO: this is basically a proxy for a full installation
fpath = pathlib.Path(__file__).parent.parent / 'data'

# static functions
def S(v, T):
    v = np.atleast_1d(v)
    T = np.atleast_1d(T)

    if len(v) > 1 and len(T) > 1:
        shape = (len(T), len(v)) # Temps along first axis, freqs along second
        v = np.broadcast_to(v, shape)
        T = np.broadcast_to(np.atleast_2d(T).T, shape)

    # there is a factor of 1/2*B(v, T) built in here because of polarization
    return h * v / (np.exp(h * v / (k * T)) - 1)

# classes
class circle:
    # a faster version of matplotlib.path.Patch.circle

    def __init__(self, center = (0, 0), radius = 0):
        self.x, self.y = center
        self.r = radius

    def contains_points(self, points):
        dx = points[..., 0] - self.x
        dy = points[..., 1] - self.y

        return dx**2 + dy**2 < self.r**2

    def to_polygons(self):
        phi = np.linspace(0, 2*pi, 100)
        x = self.x + self.r * np.cos(phi)
        y = self.y + self.r * np.sin(phi)
        return [np.stack((x, y)).T]

class ColdLoad:
    
    def __init__(self, setupname, **kwargs):
        self.apertures = []
        self.zs = []

        with open(fpath / 'setups'/ setupname / 'coldload.txt') as f:
            lines = f.readlines()
            i = 0
            while i < len(lines):
                if not lines[i].isspace():
                    path, z, i = self._setupfile_to_Path(lines, i)
                    self.apertures.append(path)
                    self.zs.append(z)
                i += 1
    
    def _setupfile_to_Path(self, lines, starting_line):
        i = starting_line

        if lines[i].split(':')[1].strip().lower() == 'circle':
            x, y, z = np.array(lines[i+1].split(':')[1].strip().split(',')).astype(float)
            D = float(lines[i+2].split(':')[1].strip())

            return circle(center = (x, y), radius = D/2), z, i+2

        elif lines[i].split(':')[1].strip().lower() == 'polygon':
            z = float(lines[i+1].split(':')[1].strip())
            x0, y0 = np.array(lines[i+2].split(':')[1].strip().split(',')).astype(float)
            vertices = np.array([[x0, y0]])
            
            x, y = np.array(lines[i+3].split(':')[1].strip().split(',')).astype(float)
            vertex = np.array([[x, y]])
            vertices = np.append(vertices, vertex, axis = 0)
            i = i+3
            while not np.all(vertices[0] == vertex):
                i += 1
                x, y = np.array(lines[i].split(':')[1].strip().split(',')).astype(float)
                vertex = np.array([[x, y]])
                vertices = np.append(vertices, vertex, axis = 0)

            return path.Path(vertices, closed = True), z, i

        else: assert False

class FilterStack:

    def __init__(self, setupname, kind = 'linear', bounds_error = True, fill_value = 0.0):
        self.spectra = {}

        with open(fpath / 'setups' / setupname / 'filters.txt') as f:
            lines = f.readlines()
            i = 0
            while i < len(lines):
                if not lines[i].isspace():
                    fname, freq_mult, i = self._setupfile_to_metadata(lines, i)
                    func = self.get_filter(fname, freq_mult, kind = kind, \
                        bounds_error = bounds_error, fill_value = fill_value)
                    self.spectra[fname.split('.')[0]] = func
                i += 1

    def _setupfile_to_metadata(self, lines, starting_line):
        i = starting_line

        if lines[i].split(':')[0].strip().lower() == 'file':
            fname = lines[i].split(':')[1].strip()
            freq_mult = freq_mult_key[fname]

        return fname, freq_mult, i

    def get_filter(self, fname, freq_mult, kind = 'linear', bounds_error = True, fill_value = 0.0):
        x, y = np.loadtxt(fpath / 'filters' / fname).T
        x *= freq_mult
        return interp1d(x, y, kind = kind, bounds_error = bounds_error, fill_value = fill_value)

class Array:

    def __init__(self, setupname, arrayname, pixel_ids = None, beam_kwargs = None, coldload_kwargs = None, \
        filterstack_kwargs = None, detector_kwargs = None):
        if beam_kwargs is None: beam_kwargs = {}
        if coldload_kwargs is None: coldload_kwargs = {}
        if filterstack_kwargs is None: filterstack_kwargs = {}
        if detector_kwargs is None: detector_kwargs = {}

        t0 = time.time()
        arrayname = arrayname + '.txt'

        with open(fpath / 'setups' / setupname / arrayname) as f:
            lines = f.readlines()
            i = 0
            while i < len(lines):
                if not lines[i].isspace():
                    fname, beam_name, x_c, y_c, z_c, phi, i = self._setupfile_to_metadata(lines, i)
                    self.wafer_locations = self.get_wafer_locations(fname)
                i += 1

        self.x_c = x_c
        self.y_c = y_c

        self.beam = Beam(beam_name, **beam_kwargs)
        self.coldload = ColdLoad(setupname, **coldload_kwargs)
        self.filters = FilterStack(setupname, **filterstack_kwargs)
        self.type = self.beam.name.split('_')[0]
        
        self.bandfunc_dict = {}
        self.add_to_bandfunc_dict('1', **filterstack_kwargs)
        self.add_to_bandfunc_dict('2', **filterstack_kwargs)

        self.pixels = {}
        self.efficiency_data = {}
        
        if pixel_ids is None: pixel_ids = self.wafer_locations.keys()
        print(f'load time = {time.time() - t0}')
        for pix in pixel_ids:
            x = self.x_c + self.wafer_locations[pix][0]*np.cos(phi) - self.wafer_locations[pix][1]*np.sin(phi)
            y = self.y_c + self.wafer_locations[pix][0]*np.sin(phi) + self.wafer_locations[pix][1]*np.cos(phi)
            self.pixels[pix] = Pixel(x, y, z_c - self.wafer_locations[pix][2], \
                self, **detector_kwargs) # minus sign in z cause of + z values in spreadsheet
        print(f'total time = {time.time() - t0}')

    def _setupfile_to_metadata(self, lines, starting_line):
        i = starting_line

        if lines[i].split(':')[0].strip().lower() == 'file':
            fname = lines[i].split(':')[1].strip()
            beam_name = lines[i+1].split(':')[1].strip()
            x, y, z = np.array(lines[i+2].split(':')[1].strip().split(',')).astype(float)
            slot_x, slot_y = np.array(lines[i+3].split(':')[1].strip().split(',')).astype(float)[:2]
            phi = np.arctan2(slot_y, slot_x) - 2*pi/3

            return fname, beam_name, x, y, z, phi, i+3

    def get_wafer_locations(self, fname):
        out = {}
        id, x, y, z, _, _ = np.genfromtxt(fpath / 'arrays' / fname, skip_header = 1, delimiter = ',').T
        for i in range(len(id)):
            out[int(id[i])] = (x[i], y[i], z[i])
        return out

    def add_to_bandfunc_dict(self, band_id, **filterstack_kwargs):
        band_id = str(band_id)
        assert band_id in ('1', '2')

        band_name = f'{self.type}_{band_id}'
        band_fn = band_name + '.txt'
        band_freq_mult = freq_mult_key[band_fn]

        self.bandfunc_dict[band_id] = {}
        self.bandfunc_dict[band_id]['func'] = self.filters.get_filter(band_fn, band_freq_mult, \
                        **filterstack_kwargs)
        self.bandfunc_dict[band_id]['label'] = band_name

    def get_efficiency(self, pixel_eff_dict, pixel_ids = None):
        '''Calculates the optical efficiency parameters for all the entries in the dictionary
        passed as the argument.

        Parameters
        ----------
        pixel_eff_dict : dict
            A dictionary of {pixel_id: args_kwargs} pairs, where pixel_id is the pixel number
            and args_kwargs is a dict with Ts, p_sats args and potentially T_errs, other_errs,
            and get_power_kwargs to pass to the get_efficiency method of each pixel.

        pixel_ids : iterable, optional
            The pixel ids to perform the calculation on, by default the keys of pixel_eff_dict
        '''
        if pixel_ids is None: pixel_ids = self.pixels.keys()
        
        for i in pixel_ids:
            Ts = pixel_eff_dict[i]['Ts']
            p_sats = pixel_eff_dict[i]['p_sats']
            T_errs = pixel_eff_dict[i].get('T_errs')
            other_errs = pixel_eff_dict[i].get('other_errs')
            get_power_kwargs = pixel_eff_dict[i].get('get_power_kwargs')

            self.efficiency_data[i] = self.pixels[i].get_efficiency(Ts, p_sats, \
                T_errs = T_errs, other_errs = other_errs, get_power_kwargs = get_power_kwargs)

    def _wafer_scatter(self, vals, pixel_ids, label, title, **kwargs):
        x = np.array([self.wafer_locations[i][0] for i in pixel_ids])
        y = np.array([self.wafer_locations[i][1] for i in pixel_ids])
        sc = plt.scatter(x, y, c = vals, cmap = plt.cm.inferno, **kwargs)
        bar = plt.colorbar(sc)
        bar.set_label(label)
        plt.xlabel('x [mm]')
        plt.ylabel('y [mm]')
        plt.title(title)
        plt.show()

    def _histogram(self, vals, label, title, **kwargs):
        plt.hist(vals, **kwargs)
        ax = plt.gca()
        x = np.mean(vals)
        ax.axvline(x, color = 'r', linestyle = '--', label = f'mean = {np.round(x, 3)}')
        plt.xlabel(label)
        plt.ylabel('Count [a.u.]')
        plt.legend()
        plt.title(title)
        plt.show()

    def _unordered_scatter(self, vals, errvals, label, title, **kwargs):
        plt.errorbar(np.arange(len(vals)), vals, yerr = errvals, **kwargs)
        ax = plt.gca()
        x = np.mean(vals)
        ax.axhline(x, color = 'r', linestyle = '--', label = f'mean = {np.round(x, 3)}')
        plt.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
        plt.ylabel(label)
        plt.legend()
        plt.title(title)
        plt.show()

    def plot_sa(self, freq, plot_type, pixel_ids = None, **kwargs):
        freq = np.atleast_1d(freq)
        assert len(freq) == 1
        if pixel_ids is None: pixel_ids = self.pixels.keys()

        sas = np.array([self.pixels[i].sa(freq) for i in pixel_ids])
        label = '$\Omega_{p}$ [rad]'
        title = f'Cold Load Solid Angle, freq = {freq[0]/1e9}GHz'
        
        if plot_type == 'wafer_scatter':
            self._wafer_scatter(sas, pixel_ids, label, title, **kwargs)
        if plot_type == 'histogram':
            self._histogram(sas, label, title, **kwargs)
        if plot_type == 'unordered_scatter':
            print('Not yet implemented for solid angle')

    def plot_fill_frac(self, freq, plot_type, pixel_ids = None, **kwargs):
        freq = np.atleast_1d(freq)
        assert len(freq) == 1
        if pixel_ids is None: pixel_ids = self.pixels.keys()

        fracs = np.array([self.pixels[i].fill_frac(freq) for i in pixel_ids])
        label = '$\Omega_{p}/\Omega_{beam}$ [rad]'
        title = f'Cold Load Beam Filling Fraction, freq = {freq[0]/1e9}GHz'

        if plot_type == 'wafer_scatter':
            self._wafer_scatter(fracs, pixel_ids, label, title, **kwargs)
        if plot_type == 'histogram':
            self._histogram(fracs, label, title, **kwargs)
        if plot_type == 'unordered_scatter':
            print('Not yet implemented for fill frac')        

    def plot_power(self, T, plot_type, pixel_ids = None, get_power_kwargs = None, **kwargs):
        T = np.atleast_1d(T)
        assert len(T) == 1 # only one plot for one temperature
        if pixel_ids is None: pixel_ids = self.pixels.keys()
        if get_power_kwargs is None: get_power_kwargs = {}

        # with one T, out and sig_out are len(1) arrays
        data = 1e12*np.array([self.pixels[i].get_power(T, **get_power_kwargs) for i in pixel_ids])[..., 0]
        p_opts, p_opts_err = data.T
        label = '$P_{opt}$ [pW]'
        title = f'Optical Power, T = {T[0]}K'

        if plot_type == 'wafer_scatter':
            self._wafer_scatter(p_opts, pixel_ids, label, title, **kwargs)
        if plot_type == 'histogram':
            self._histogram(p_opts, label, title, **kwargs)
        if plot_type == 'unordered_scatter':
            self._unordered_scatter(p_opts, p_opts_err, label, title, **kwargs)    

    def plot_efficiency(self, plot_type, param = 'eta', pixel_ids = None, **kwargs):
        param_dict = {
            'eta': {
                'idx': 0,
                'label': '$\eta$ [a.u.]',
                'mult': 1
            },
            'C': {
                'idx': 1,
                'label': '$P_{sat}(0)$ [pW]',
                'mult': 1e12
            }
        }

        idx = param_dict[param]['idx']
        mult = param_dict[param]['mult']
        label = param_dict[param]['label']
        if pixel_ids is None: pixel_ids = self.pixels.keys()

        vals = mult*np.array([self.efficiency_data[i][0][idx] for i in pixel_ids]) # 0 gets the 'mean' data
        errs = mult*np.array([np.sqrt(np.diag(self.efficiency_data[i][1]))[idx] for i in pixel_ids]) # 1 gets the 'cov' data
        title = f'Optical Efficiency Fit Parameter {label}'
        
        if plot_type == 'wafer_scatter':
            self._wafer_scatter(vals, pixel_ids, label, title, **kwargs)
        if plot_type == 'histogram':
            self._histogram(vals, label, title, **kwargs)
        if plot_type == 'unordered_scatter':
            self._unordered_scatter(vals, errs, label, title, **kwargs)

class SPB(Array):

    def __init__(self, setupname, location, beam_name, pixel_ids = None, beam_kwargs = None, coldload_kwargs = None, \
        filterstack_kwargs = None, detector_kwargs = None):
        if beam_kwargs is None: beam_kwargs = {}
        if coldload_kwargs is None: coldload_kwargs = {}
        if filterstack_kwargs is None: filterstack_kwargs = {}
        if detector_kwargs is None: detector_kwargs = {}

        x, y, z = location

        self.x_c = x
        self.y_c = y
        
        self.beam = Beam(beam_name, **beam_kwargs)
        self.coldload = ColdLoad(setupname, **coldload_kwargs)
        self.filters = FilterStack(setupname, **filterstack_kwargs)
        self.type = self.beam.name.split('_')[0]

        self.bandfunc_dict = {}
        self.add_to_bandfunc_dict('1', **filterstack_kwargs)
        self.add_to_bandfunc_dict('2', **filterstack_kwargs)
        
        self.pixels = {}
        self.wafer_locations = {}
        self.efficiency_data = {}

        pixel = Pixel(x, y, z, self, **detector_kwargs)
        if pixel_ids is None: pixel_ids = (1,)
        for pix in pixel_ids:
            self.pixels[pix] = pixel # Many ids, one pixel
            
            # dummy locations for plotting only
            y = pix // 21
            x = pix - 21*y
            self.wafer_locations[pix] = (x, y, z)
        
class Pixel:
    
    def __init__(self, x, y, z, array, N = int(5e4), kind = 'linear'):
        self.x = x
        self.y = y
        self.z = z
        self.array = array
        self.r = np.sqrt((self.x - self.array.x_c)**2 + (self.y - self.array.y_c)**2)
        self.N = N
        self.kind = kind

        self.idxs = np.random.choice(np.arange(self.array.beam.N), size = self.N, replace = False)

        self.sas, self.sas_err, self.sa = self.init_solid_angles()
        self.fill_fracs, self.fill_fracs_err, self.fill_frac = self.init_fill_frac()

    def get_mask(self, freq, coords = False):
        thetas = self.array.beam.det_theta_samples[freq][self.idxs]
        phis = 2*pi*np.random.rand(self.N)

        out = np.full(self.N, True)
        tan = np.tan(thetas)
        cos = np.cos(phis)
        sin = np.sin(phis)

        for i, ap in enumerate(self.array.coldload.apertures):
            z = self.z - self.array.coldload.zs[i]
            r = z * tan
            x = self.x + r * cos
            y = self.y + r * sin
            points = np.stack((x, y)).T
            out = np.logical_and(out, ap.contains_points(points))

        if not coords: return ~out
        else: return ~out, thetas, phis

    def get_solid_angle(self, freq):
        y = self.array.beam.det_ys[freq][self.idxs]
        mask = self.get_mask(freq)
        return mc.mc_integrate(y, mask = mask)

    def init_solid_angles(self):
        out = np.zeros(len(self.array.beam.freq_keys))
        sig_out = np.zeros(len(out))

        for i, freq in enumerate(self.array.beam.freq_keys):
            out[i], sig_out[i] = self.get_solid_angle(freq)

        f = interp1d(self.array.beam.freqs_SI, out, kind = self.kind, bounds_error = True)

        return out, sig_out, f

    # TODO: implement correlated errors
    def init_fill_frac(self):
        out = self.sas / self.array.beam.sas
        
        rel_err_det = self.sas_err / self.sas
        rel_err_beam = self.array.beam.sas_err / self.array.beam.sas
        sig_out = out * np.sqrt(rel_err_det**2 + rel_err_beam**2)

        f = interp1d(self.array.beam.freqs_SI, out, kind = self.kind, bounds_error = True)

        return out, sig_out, f

    def get_power(self, T, min = None, max = None, N = int(4e5), band = None):
        if min == None: min = np.min(self.array.beam.freqs_SI)
        if max == None: max = np.max(self.array.beam.freqs_SI)
        
        dist = uniform(loc = min, scale = max - min)
        x = dist.rvs(size = N)
        p = dist.pdf(x)

        if band is None: bandfunc = lambda x: 1
        else: bandfunc = self.array.bandfunc_dict[band]['func']

        f_det = np.prod([bandfunc(x), S(x, T), self.fill_frac(x)], axis = 0)
        f_filts = np.prod([filt_func(x) for filt_func in self.array.filters.spectra.values()], axis = 0)
        f = np.prod([f_det, f_filts], axis = 0)

        out, sig_out = mc.mc_integrate(f/p)
        return out, sig_out

    def get_efficiency(self, Ts, p_sats, T_errs = None, other_errs = None, get_power_kwargs = None):
        assert len(Ts) == len(p_sats)
        if T_errs is None: T_errs = np.zeros(len(Ts))
        if other_errs is None: other_errs = np.zeros(len(Ts))
        if get_power_kwargs is None: get_power_kwargs = {}

        assert len(Ts) == len(T_errs)
        assert len(Ts) == len(other_errs)
        do_T_err = True
        if np.allclose(T_errs, np.zeros(len(T_errs)), rtol=0, atol=1e-8): do_T_err = False
        
        p_opts, p_opts_errs = self.get_power(Ts, **get_power_kwargs)

        if do_T_err:
            p_opts_low = self.get_power(Ts - T_errs, **get_power_kwargs)[0]
            p_opts_high = self.get_power(Ts + T_errs, **get_power_kwargs)[0]
            p_opts_T_errs = (p_opts_high - p_opts_low)/2
            p_opts_errs = np.sqrt(p_opts_errs**2 + p_opts_T_errs**2)

        p_opts_errs = np.sqrt(p_opts_errs**2 + other_errs**2)

        mean, cov = eff.eff_fit(p_opts, p_sats, p_opts_errs = p_opts_errs)
        return mean, cov, p_opts, p_opts_errs

    def plot_efficiency_curve(self, Ts, p_sats, T_errs = None, other_errs = None, get_power_kwargs = None):
        if get_power_kwargs is None: get_power_kwargs = {}

        mean, cov, p_opts, p_opts_errs = self.get_efficiency(Ts, p_sats, \
            T_errs = T_errs, other_errs = other_errs, get_power_kwargs = get_power_kwargs)

        plt.errorbar(p_opts*1e12, p_sats*1e12, xerr = p_opts_errs*1e12, fmt = 'o', capsize = 5)
        
        eta, C = mean
        eta_err, _ = np.sqrt(np.diag(cov))
        plt.plot(p_opts*1e12, eff.eff_fit_func(p_opts, eta, C)*1e12, \
            label = f'$\eta = {np.round(eta, 3)} \pm {np.round(eta_err, 3)}$')

        plt.xlabel('$P_{opt}$ [pW]')
        plt.ylabel('$P_{sat}$ [pW]')
        plt.legend()
        plt.show()

    def _hz2ghz_tickformatter(self, val, loc):
        return val / 1e9

    def plot_spectra(self, min = None, max = None, band = None):
        if min == None: min = np.min(self.array.beam.freqs_SI)
        if max == None: max = np.max(self.array.beam.freqs_SI)

        x = np.linspace(min, max, 1000)
        for filt_name, filt_func in self.array.filters.spectra.items():
            plt.plot(x, filt_func(x), label = filt_name)

        if band is None: bandfunc = lambda x: 1
        else:
            bandfunc = self.array.bandfunc_dict[band]['func']
            bandlabel = self.array.bandfunc_dict[band]['label']
            plt.plot(x, bandfunc(x), label = bandlabel)

        plt.plot(x, self.fill_frac(x), label = 'Beam fill fraction')

        f_filts = np.prod([filt(x) for filt in self.array.filters.spectra.values()], axis = 0)
        f = np.prod([bandfunc(x), self.fill_frac(x), f_filts], axis = 0)
        plt.plot(x, f, label = 'Combined')

        plt.xlabel('Frequency [GHz]')
        plt.ylabel('Transmission')
        plt.legend()

        ax = plt.gca()
        ax.xaxis.set_major_formatter(FuncFormatter(self._hz2ghz_tickformatter))
        plt.show()

    def _rad2deg_tickformatter(self, val, loc):
        new_val = int(np.round(np.rad2deg(val)))
        return f'${new_val}^0$'

    def plot_apertures(self, N = 5000):
        for i in range(len(self.array.coldload.apertures)):
            ap = self.array.coldload.apertures[i]
            z = self.array.coldload.zs[i]
            x, y = ap.to_polygons()[0].T

            # get interpolated points
            t = np.linspace(0, 1, len(x))
            t_fill = np.linspace(0, 1, 500)

            x_fill = np.interp(t_fill, t, x)
            y_fill = np.interp(t_fill, t, y)

            dx = x_fill - self.x
            dy = y_fill - self.y
            dz = z - self.z
            
            theta = np.arccos(-dz/np.sqrt(dx**2 + dy**2 + dz**2))
            phi = np.arctan2(dy, dx+1e-14)

            plt.polar(phi, theta)
            plt.ylim(0,pi/2)

        ax = plt.gca()
        ax.yaxis.set_major_formatter(FuncFormatter(self._rad2deg_tickformatter))
        
        min_freq = str(np.min(self.array.beam.freq_keys.astype(int)))
        wide_beam = self.array.beam.beams[min_freq]
        
        mask, thetas, phis = self.get_mask(min_freq, coords = True)

        thetas = thetas[~mask][:N]
        phis = phis[~mask][:N]

        c_values = wide_beam(thetas)
        sc = plt.scatter(phis, thetas, c = c_values, marker = '.', cmap = plt.cm.inferno,
            vmin = 0, vmax = 1)
        plt.colorbar(sc, pad = .1)

        plt.show()