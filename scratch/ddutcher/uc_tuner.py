# uc_tuner.py

import os
import time
import numpy as np
from sodetlib import noise

class UCTuner:
    """
    UCTuner is the object used to perform and hold the information from
    uc_att and tone power optimization, designed for use in the
    uxm_optimize_quick function.

    Args
    ----
    S : SmurfControl
        Pysmurf instance
    cfg : DetConfig
        DetConfig instance
    band : int
        SMuRF band on which to operate.

    Other Parameters
    ----------------
    stream_time, fmin, fmax, detrend:
        Used for noise taking and white noise calculation.

    Attributes
    ----------
    current_tone_power : int
        The SMuRF RF tone power used for `band`.
    estimate_att : int
        The uc_att setting resulting in the lowest noise thus far.
    lowest_wl_index : int
        Index corresponding to estimate_att from a list of uc_att settings.
    wl_median : float
        Median white noise level in pA/rtHz
    wl_length : int
        Number of channels used in wl_median determination.
    status : str
        Status string containing wl_median, wl_length, and channel_length.
    """
    def __init__(self, S, cfg, band=None):
        self._S = S
        self._cfg = cfg
        self.band = band
        self.current_tone_power = None
        self.estimate_att = None
        self.lowest_wl_index = None
        self.wl_median = None
        self.wl_length = None
        self.status = None
        self.wl_list = None
        self.uc_attens = None

        if band is None:
            raise ValueError("Must specify `band` as int [0-7]")

        self.current_uc_att = S.get_att_uc(band)
        self.current_tone_power = S.amplitude_scale[band]

    
    def uc_tune(self, uc_attens):
        """
        Find the uc_att from those provided that yields the lowest noise.
        """
        uc_attens = np.atleast_1d(uc_attens)
        self.uc_attens = uc_attens
        wl_list = []
        wl_len_list = []
        for atten in uc_attens:
            self._S.set_att_uc(self.band, atten)
            self._S.tracking_setup(
                self.band,
                reset_rate_khz=self._cfg.dev.bands[self.band]["flux_ramp_rate_khz"],
                fraction_full_scale=self._cfg.dev.bands[self.band]["frac_pp"],
                make_plot=False,
                save_plot=False,
                show_plot=False,
                channel=self._S.which_on(self.band),
                nsamp=2 ** 18,
                lms_freq_hz=self._cfg.dev.bands[self.band]["lms_freq_hz"],
                meas_lms_freq=self._cfg.dev.bands[self.band]["meas_lms_freq"],
                feedback_start_frac=self._cfg.dev.bands[self.band]["feedback_start_frac"],
                feedback_end_frac=self._cfg.dev.bands[self.band]["feedback_end_frac"],
                lms_gain=self._cfg.dev.bands[self.band]["lms_gain"],
            )
            am, outdict = noise.take_noise(self._S, self._cfg, acq_time=30, fit=False,
                                           plot_band_summary=False, show_plot=False)

            noise_param = outdict['noisedict']['noise_pars'][:,0]

            wl_list.append(round(np.nanmedian(noise_param), 1))
            wl_len_list.append(len(noise_param))

        lowest_wl_index = wl_list.index(min(wl_list))
        if lowest_wl_index == len(wl_list):
            lowest_wl_index = -1
        estimate_att = uc_attens[lowest_wl_index]
        wl_median = wl_list[lowest_wl_index]

        self.wl_list = wl_list
        self.estimate_att = estimate_att
        self.lowest_wl_index = lowest_wl_index
        self.wl_median = wl_median
        self.wl_length= wl_len_list[lowest_wl_index]
        self.status = (f"WL: {self.wl_median:.1f} pA/rtHz with"
                       + f" {self.wl_length} channels.")


    def rough_tune(self):
        """
        Takes noise with current uc_att and uc_att +/-5 and +/- 10.
        """
        uc_attens = np.array(
            [
                self.current_uc_att - 10,
                self.current_uc_att - 5,
                self.current_uc_att,
                self.current_uc_att + 5,
                self.current_uc_att + 10,
            ]
        )
        uc_attens = uc_attens[np.where((uc_attens <= 30) & (uc_attens >= 0))]

        self.uc_tune(uc_attens=uc_attens)


    def fine_tune(self):
        """
        Takes noise with current uc_att and uc_att +/-2 and +/- 4.
        """
        uc_attens = np.array(
            [
                self.current_uc_att - 4,
                self.current_uc_att - 2,
                self.current_uc_att,
                self.current_uc_att + 2,
                self.current_uc_att + 4,
            ]
        )
        uc_attens = uc_attens[np.where((uc_attens <= 30) & (uc_attens >= 0))]

        self.uc_tune(uc_attens=uc_attens)


    def set_tone_and_uc(self, tone_power, uc_att):
        """
        Apply new tone power and uc_att settings.
        Re-runs find_freqs, setup notches, and serial algs.
        """
        self._S.amplitude_scale[self.band] = tone_power
        self._S.set_att_uc(self.band, uc_att)
        self._S.find_freq(self.band, tone_power=tone_power, make_plot=True)
        self._S.setup_notches(self.band, tone_power=tone_power,
                              new_master_assignment=True)
        self._S.run_serial_gradient_descent(self.band)
        self._S.run_serial_eta_scan(self.band)
        self.current_uc_att = uc_att
        self.current_tone_power = tone_power


    def increase_tone_power(self):
        """
        Increases tone power by 2 and uc_att by 11. Should result in
        approximately the same power at the resonators.
        """
        self.set_tone_and_uc(
            tone_power=self.current_tone_power + 2,
            uc_att=np.min([self.current_uc_att + 11, 30])
        )


    def decrease_tone_power(self):
        """
        Decreases tone power by 2 and uc_att by 11. Should result in
        approximately the same power at the resonators.
        """
        self.set_tone_and_uc(
            tone_power=self.current_tone_power - 2,
            uc_att=np.max([self.current_uc_att - 11, 0])
        )
