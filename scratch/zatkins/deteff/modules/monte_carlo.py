#!/usr/bin/env python3

# author: zatkins
# desc: a collection of classes and methods related to integrals over solid angle

import numpy as np
from scipy.stats import rv_continuous, uniform, beta
from scipy.optimize import curve_fit
from scipy.constants import pi

# define static methods

def unity_filter(x, min=0, max=pi/2):
    '''Return 1 if data between two bounds, inclusive; return 0 otherwise.

    Parameters
    ----------
    x : ndarray
        Data to condition the filter upon.
    min : scalar, optional
        Lower bound, by default 0
    max : scalar, optional
        Upper bound, by default pi/2

    Returns
    -------
    ndarray
        An array with same shape as x, valued as 1 where x between bounds, otherwise 0.
    '''
    return np.where(np.logical_and(x >= min, x <= max), 1, 0)

def mc_integrate(y, mask=False):
    '''Perform a monte carlo integral of the transformed data.

    Parameters
    ----------
    y : ndarray
        The function to be integrated divided by the pdf of each sample. May have any 
        dimension, integration is performed along the last axis
    mask : bool, optional
        Where True, replace the data point with 0, by default False (broadcast to y.shape)

    Returns
    -------
    2-tuple
        The estimate of the integral and the standard error of the estimate.

    Notes
    -----
    The data, y, must represent the function to be integrated divided by
    the numeric value of the pdf of each sample where the function is evaluated.
    
    The samples must be drawn from the same pdf for the result to be unbiased.

    Examples
    --------
    >>> x = norm.rvs(100)
    >>> f = lambda x: 1/(1+x**2)
    >>> y = f(x)/norm.pdf(x)
    >>> integral, error = mc_integrate(y)
    '''
    # prepare the data, broadcast mask shape if necessary
    y = np.atleast_2d(y)
    mask = np.atleast_2d(mask)
    if mask.size == 1:
        mask = np.broadcast_to(mask, y.shape)
    N = y.shape[-1]  # size of each "integrand" array

    # replace masked values with 0, perform integral along last axis
    filt_y = np.where(~mask, y, 0)
    est = np.mean(filt_y, axis=-1)
    return est, np.std(filt_y, ddof=1, axis=-1) / np.sqrt(N)

# TODO: unpack this function a bit so it can do arbitrary mc sims, not
# just mc_ints (eg, good for ratios of mc_ints etc)
def mcmc_integrate(y, y2=None, N_sims=100, mask=False, do_std=True):
    mask = np.atleast_1d(mask)

    idxs = np.arange(len(y))
    np.random.shuffle(idxs)
    y_shuffled = y[idxs]

    y_reshaped = y_shuffled.reshape(N_sims, -1)
    if mask.size == 1:
        mask_reshaped = np.broadcast_to(mask, y_reshaped.shape)
    else:
        mask_shuffled = mask[idxs]
        mask_reshaped = mask_shuffled.reshape(N_sims, -1)

    assert y_reshaped.shape == mask_reshaped.shape

    est, _ = mc_integrate(y_reshaped, mask=mask_reshaped)
    if do_std:
        return np.mean(est), np.std(est) / np.sqrt(N_sims)
    else:
        return np.mean(est), np.zeros(est.shape)

# define distributions

class omega(rv_continuous):
    '''An rv_continuous custom subclass for a random variable distributed uniformly
    on the unit half-sphere (theta only)
    '''

    def _pdf(self, x):
        return unity_filter(x) * np.sin(x)

    def _ppf(self, q):
        return np.arccos(1 - q)

# only way to offer speed advantage over stats.norm is to force pdf, rvs to
# not inherit from rv_continuous
class norm(rv_continuous):
    '''Overwrite stats.norm with the only two methods needed in this module: pdf() and rvs(). 
    
    Implementation is faster than the base class. Method signatures are
    the same as the base class.
    '''
    def pdf(self, x, loc=0, scale=1):
        x = (x - loc) / scale
        return 1/np.sqrt(2*pi)/scale * np.exp(-x**2/2)

    def rvs(self, loc=0, scale=1, size=1):
        size = np.atleast_1d(size)
        return loc + scale * np.random.randn(*size)

    def __call__(self, loc=0, scale=1):
        '''Return a "frozen" normal distribution with location and scale fixed

        Parameters
        ----------
        loc : scalar, optional
            Mean of the normal distribution, by default 0
        scale : int, optional
            Standard deviation of the normal distribution, by default 1

        Returns
        -------
        rv_continuous instance
            A "frozen" normal distribution with location and scale fixed
        '''
        return _norm_frozen(loc, scale)

class _norm_frozen(norm):
    '''A private subclass of norm that fixes the location and scale parameters
    '''
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale

    def pdf(self, x):
        return super().pdf(x, loc=self.loc, scale=self.scale)

    def rvs(self, size=1):
        return super().rvs(loc=self.loc, scale=self.scale, size=size)

# rv_continuous does not play nicely with non-numeric args
class mixture:
    '''A base class representing a "mixture" distribution: a weighted linear combination of underlying 
    statistical distributions, with positive semi-definite weights adding to 1.
    '''

    def __init__(self, list_of_dists=None):
        '''A mixture instance, with same pdf(), rvs() API as scipy.stats.rv_continuous instances

        Parameters
        ----------
        list_of_dists : list, optional
            The underlying stastical distributions of type rv_continuous, by default [uniform]
        '''
        if list_of_dists is None: list_of_dists = [uniform]

        self.dists = list_of_dists
        self.k = len(list_of_dists)

    def pdf(self, x, *weights, dist_args=None, dist_kwargs=None):
        '''Return the pdf of the mixture distribution evaluated at locations given by x-array

        Parameters
        ----------
        x : ndarray
            Data points at which to evaluate the pdf
        dist_args : list, optional
            Argument tuples to pass to underlying distributions' pdf() methods, by default [()]
        dist_kwargs : list, optional
            Keyword dictionaries to pass to underlying distributions' pdf() methods, by default [{}]

        Returns
        -------
        ndarray
            The numerical value of the pdf evaluated at x

        Notes
        -----
        Weights are passed as args following the samples x
        '''
        if dist_args is None: dist_args = [()]
        if dist_kwargs is None: dist_kwargs=[{}]

        weights = np.atleast_1d(weights)
        dist_args = np.atleast_1d(dist_args)
        dist_kwargs = np.atleast_1d(dist_kwargs)

        # must have same number of dists and weights
        assert len(weights) == self.k, \
            print(f'expecting {self.k} weights but got {len(weights)}')

        # weights must add up to really close to 1
        assert np.isclose(sum(weights), 1, rtol=0, atol=1e-8), \
            print(f'expecting 1 but got {sum(weights)}')

        # weights must be positive semi-definite
        assert np.all(np.logical_and(0 <= weights, weights <= 1)), \
            print(f'weights {weights} not positive semi-definite')

        # check the lengths of dist_args and dist_kwargs
        assert self.k == len(dist_args) and self.k == len(dist_kwargs), \
            print(f'expecting {self.k} dist_args, dist_kwargs but got {len(dist_args), len(dist_kwargs)}')

        # get the pdf of each underlying distribution and multiply by its weight
        partial_pdfs = np.array([weights[i]*self.dists[i].pdf(x, *dist_args[i], **dist_kwargs[i])
                                 for i in range(self.k)])

        # return the sum of the pdfs across distributions (along the 0th axis)
        return np.sum(partial_pdfs, axis=0)

    def rvs(self, *weights, dist_args=None, dist_kwargs=None, size=1):
        '''Return samples from the mixture distribution with given pdf

        Parameters
        ----------
        dist_args : list, optional
            Argument tuples to pass to underlying distributions' pdf() methods, by default [()]
        dist_kwargs : list, optional
            Keyword dictionaries to pass to underlying distributions' pdf() methods, by default [{}]
        size : tuple-of-ints, optional
            The shape of samples to draw, by default 1. Can by n-dimensional.

        Returns
        -------
        ndarray
           An unordered ndarray whose pdf is given by self.pdf(x, *weights, ...)

        Notes
        -----
        Weights are passed as the only args in this method
        '''
        if dist_args is None: dist_args = [()]
        if dist_kwargs is None: dist_kwargs=[{}]

        weights = np.atleast_1d(weights)
        dist_args = np.atleast_1d(dist_args)
        dist_kwargs = np.atleast_1d(dist_kwargs)
        size = np.atleast_1d(size)

        # must have same number of dists and weights
        assert len(weights) == self.k, \
            print(f'expecting {self.k} weights but got {len(weights)}')

        # weights must add up to really close to 1
        assert np.isclose(sum(weights), 1, rtol=0, atol=1e-8), \
            print(f'expecting 1 but got {sum(weights)}')

        # weights must be positive semi-definite
        assert np.all(np.logical_and(0 <= weights, weights <= 1)), \
            print(f'weights {weights} not positive semi-definite')

        # check the lengths of dist_args and dist_kwargs
        assert self.k == len(dist_args) and self.k == len(dist_kwargs), \
            print(f'expecting {self.k} dist_args, dist_kwargs but got {len(dist_args), len(dist_kwargs)}')

        # get the number of samples for each underlying distribution
        N = np.prod(size)
        draws = np.random.choice(np.arange(self.k), size=N, p=weights)
        Ns = np.bincount(draws, minlength=self.k) # ensure correct shape even if some Ns are 0

        # check no off-by-1 errors in rounding
        assert np.round(weights * N).sum() == N
        assert Ns.sum() == N, f'expected {N}, got {Ns.sum()}'

        # sample each underlying distribution and concatenate result
        out = np.concatenate([self.dists[i].rvs(size=Ns[i], *dist_args[i], **dist_kwargs[i])
                              for i in range(self.k)])
        assert out.shape == (N,) # because paranoia
        np.random.shuffle(out) # shuffle samples
        return out.reshape(size) # return in correct shape

class norm_uniform(mixture):
    '''A mixture subclass of a normal distribution and a uniform distribution

    Notes
    -----
    Only the weight associated with the normal distribution is needed. The uniform
    distribution is fixed to location = 0, scale = pi/2
    '''
    def __init__(self):
        super().__init__(list_of_dists=[norm(), uniform])
        self.bounds = ((0, 0, -np.inf, 0), (np.inf, 1, np.inf, np.inf))
        self.p0 = (1, 0.8, .25, .25)

    def pdf(self, x, w_norm, *args, **kwargs):
        '''Return the pdf of the normal + [0, pi/2] uniform distribution

        Parameters
        ----------
        x : ndarray
            Data points at which to evaluate the pdf
        w_norm : scalar
            Weight assigned to the normal distribution

        Returns
        -------
        ndarray
            The pdf of the normal + [0, pi/2] uniform distribution evaluated at x

        Notes
        -----
        Args and kwargs are passed to the normal distribution as args and kwargs. The uniform
        distribution is fixed to location = 0, scale = pi/2
        '''
        return super().pdf(x, w_norm, 1-w_norm, dist_args=[args, ()],
                           dist_kwargs=[kwargs, dict(loc=0, scale=pi/2)])

    def rvs(self, w_norm, *args, size=1, **kwargs):
        '''Sample from the pdf of the normal + [0, pi/2] uniform distribution

        Parameters
        ----------
        w_norm : scalar
            Weight assigned to the normal distribution
        size : tuple-of-ints, optional
            The shape of samples to draw, by default 1. Can by n-dimensional.

        Returns
        -------
        ndarray
            An unordered ndarray whose pdf is given by w_norm*norm.pdf(x, ...)+(1-w_norm)*uniform.pdf(x, ...)

        Notes
        -----
        Args and kwargs are passed to the normal distribution as args and kwargs. The uniform
        distribution is fixed to location = 0, scale = pi/2
        '''
        return super().rvs(w_norm, 1-w_norm, dist_args=[args, ()],
                           dist_kwargs=[kwargs, dict(loc=0, scale=pi/2)], size=size)

    def f(self, x, A, w_norm, loc, scale):
        return A * self.pdf(x, w_norm, loc=loc, scale=scale)

    def fit_params(self, x, y):
        return curve_fit(self.f, x, y, p0 = self.p0, bounds = self.bounds)

class beta_uniform(mixture):
    '''A mixture subclass of a beta distribution and a uniform distribution

    Notes
    -----
    Only the weight associated with the beta distribution is needed. The uniform
    distribution is fixed to location = 0, scale = pi/2
    '''

    def __init__(self):
        super().__init__(list_of_dists=[beta, uniform])
        self.bounds = ((0, 0, 0, 0), (np.inf, 1, np.inf, np.inf))
        self.p0 = (.25, 0.95, 2, 10)

    def pdf(self, x, w_beta, *args, **kwargs):
        '''Return the pdf of the beta + [0, pi/2] uniform distribution

        Parameters
        ----------
        x : ndarray
            Data points at which to evaluate the pdf
        w_beta : scalar
            Weight assigned to the beta distribution

        Returns
        -------
        ndarray
            The pdf of the beta + [0, pi/2] uniform distribution evaluated at x

        Notes
        -----
        Args and kwargs are passed to the beta distribution as args and kwargs. The uniform
        distribution is fixed to location = 0, scale = pi/2
        '''
        return super().pdf(x, w_beta, 1-w_beta, dist_args=[args, ()],
                           dist_kwargs=[kwargs, dict(loc=0, scale=pi/2)])

    def rvs(self, w_beta, *args, size=1, **kwargs):
        '''Sample from the pdf of the beta + [0, pi/2] uniform distribution

        Parameters
        ----------
        w_beta : scalar
            Weight assigned to the normal distribution
        size : tuple-of-ints, optional
            The shape of samples to draw, by default 1. Can by n-dimensional.

        Returns
        -------
        ndarray
            An unordered ndarray whose pdf is given by w_beta*beta.pdf(x, ...)+(1-w_beta)*uniform.pdf(x, ...)

        Notes
        -----
        Args and kwargs are passed to the beta distribution as args and kwargs. The uniform
        distribution is fixed to location = 0, scale = pi/2
        '''
        return super().rvs(w_beta, 1-w_beta, dist_args=[args, ()],
                           dist_kwargs=[kwargs, dict(loc=0, scale=pi/2)], size=size)

    def f(self, x, A, w_beta, a, b):
        return A * self.pdf(x, w_beta, a, b)

    def fit_params(self, x, y):
        return curve_fit(self.f, x, y, p0 = self.p0, bounds = self.bounds)