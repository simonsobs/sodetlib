import numpy as np
from lmfit import Model

from timer_wrap import timing


def linear_resonator(f, f_0, Q, Q_e_real, Q_e_imag):
    Q_e = Q_e_real + 1j * Q_e_imag
    return (1 - (Q * Q_e ** (-1) / (1 + 2j * Q * (f - f_0) / f_0)))


def cable_delay(f, delay, phi, f_min):
    return np.exp(1j * (-2 * np.pi * (f - f_min) * delay + phi))


def general_cable(f, delay, phi, f_min, A_mag, A_slope):
    phase_term = cable_delay(f, delay, phi, f_min)
    magnitude_term = ((f - f_min) * A_slope + 1) * A_mag
    return magnitude_term * phase_term


def resonator_cable(f, f_0, Q, Q_e_real, Q_e_imag, delay, phi, f_min, A_mag, A_slope):
    # combine above functions into our full fitting functions
    resonator_term = linear_resonator(f, f_0, Q, Q_e_real, Q_e_imag)
    cable_term = general_cable(f, delay, phi, f_min, A_mag, A_slope)
    return cable_term * resonator_term


@timing
def full_fit(freqs, real, imag):
    # takes numpy arrays of freq, real and imag values

    # turn real and imag s21 into a single complex array
    s21_complex = np.vectorize(complex)(real, imag)

    argmin_s21 = np.abs(s21_complex).argmin()
    fmin = freqs.min()
    fmax = freqs.max()
    f_0_guess = freqs[argmin_s21]
    Q_min = 0.1 * (f_0_guess / (fmax - fmin))
    delta_f = np.diff(freqs)
    min_delta_f = delta_f[delta_f > 0].min()
    Q_max = f_0_guess / min_delta_f
    Q_guess = np.sqrt(Q_min * Q_max)
    s21_min = np.abs(s21_complex[argmin_s21])
    s21_max = np.abs(s21_complex).max()
    Q_e_real_guess = Q_guess / (1 - s21_min / s21_max)
    A_slope, A_offset = np.polyfit(freqs - fmin, np.abs(s21_complex), 1)
    A_mag = A_offset
    A_mag_slope = A_slope / A_mag
    phi_slope, phi_offset = np.polyfit(freqs - fmin, np.unwrap(np.angle(s21_complex)), 1)
    delay = -phi_slope / (2 * np.pi)
    # Q_i=1/(1/Q_guess-1/Q_e_real_guess)

    # make our model
    totalmodel = Model(resonator_cable)
    params = totalmodel.make_params(f_0=f_0_guess,
                                    Q=Q_guess,
                                    Q_e_real=Q_e_real_guess,
                                    Q_e_imag=0,
                                    delay=delay,
                                    phi=phi_offset,
                                    f_min=fmin,
                                    A_mag=A_mag,
                                    A_slope=A_mag_slope)
    # set some bounds
    params['f_0'].set(min=freqs.min(), max=freqs.max())
    params['Q'].set(min=Q_min, max=Q_max)
    params['Q_e_real'].set(min=1, max=1e7)
    params['Q_e_imag'].set(min=-1e7, max=1e7)
    params['phi'].set(min=phi_offset - np.pi, max=phi_offset + np.pi)

    # fit it
    result = totalmodel.fit(s21_complex, params, f=freqs)
    return result


def fine_s21_model(freqs_fine, fit_params):
    # use this after fitting the data to get a prettier model
    totalmodel = Model(resonator_cable)
    params = totalmodel.make_params(**fit_params)
    fine_model = totalmodel.eval(params, f=freqs_fine)
    return fine_model


def get_qi(Q, Q_e_real):
    return (Q ** -1 - Q_e_real ** -1) ** -1


def get_br(Q, f_0):
    return f_0 * (2 * Q) ** -1


@timing
def reduced_chi_squared(ydata, ymod, n_param=9, sd=None):
    # red chi squared in lmfit does not return something reasonable
    # so here is a handwritten function
    # you want sd to be the complex error

    chisq = np.sum((np.real(ydata) - np.real(ymod)) ** 2 / ((np.real(sd)) ** 2)) + np.sum(
        (np.imag(ydata) - np.imag(ymod)) ** 2 / ((np.imag(sd)) ** 2))
    nu = 2 * ydata.size - n_param  # multiply  the usual by 2 since complex
    red_chisq = chisq / nu
    return chisq, red_chisq


def residuals(ydata, ymod):
    return ydata - ymod
