# optimize.py
# Module for various functions optimizing the setup of the resonators.
# Currently just holds one main function for adjusting uc_att and tone.

import time
import os, sys
import numpy as np
import scipy.signal as signal
from sodetlib import noise
from sodetlib.operations import tracking


def tone_uc_att(
    S,
    cfg,
    bands=None,
    uxm="ufm",
    low_noise_thresh=None,
    med_noise_thresh=None,
    high_noise_thresh=None,
):
    """
    Optimize tone power and uc atten for the specified bands.

    Parameters
    ----------
    S :
        Pysmurf control instance
    cfg :
        DetConfig instance
    bands : array-like
        List of SMuRF bands to optimize
    uxm : ["umm", "ufm"]
        Assembly type. Determines the relevant default noise thresholds.
    low_noise_thresh : float, default None
        If white noise below this, do one fine adjustment to uc atten then stop.
        If specified, overrides default setting for specified uxm.
    med_noise_thresh : float, default None
        If white noise below this, do one rough adjustment to uc atten.
        If white noise above this, do up to two rough adjustments to uc atten.
        If specified, overrides default setting for specified uxm.
    high_noise_thresh : float, default None
        If white noise above this, raise ValueError. Inspect manually.
        If specified, overrides default setting for specified uxm.
    """
    if bands is None:
        bands = S.config.get("init").get("bands")
    bands = np.atleast_1d(bands)

    # Default noise thresholds based on Phase 2 targets
    if low_noise_thresh is None:
        low_noise_thresh = 45 if uxm == "umm"  else 135
    if med_noise_thresh is None:
        med_noise_thresh = 65 if uxm == "umm" else 150
    if high_noise_thresh is None:
        high_noise_thresh = 300

    for opt_band in bands:
        S.log(f"Estimating attens for band {band}")
        uctuner = UCTuner(S, cfg, band=opt_band)
        uctuner.uc_tune(uc_attens=uctuner.current_uc_att)
        S.log(f"UC attens: {uctuner.uc_attens}")
        S.log(f"Noise levels: {uctuner.wl_list}")
        S.log(f"Number of active channels: {uctuner.wl_length}")

        if uctuner.wl_median >= high_noise_thresh:
            summary = (
                f"wl_median={uctuner.wl_median} is too high. "
                + "Power level might be really off, please investigate."
            )
            S.log(summary)
            return False, summary

        elif uctuner.wl_median <= low_noise_thresh:
            # Do one fine tune and stop.
            uctuner.status_tune("fine")

        elif low_noise_thresh < uctuner.wl_median <= med_noise_thresh:
            # Do a rough tune followed by a fine tune.
            uctuner.status_tune("rough")

            # If needed, adjust the tone powers so the attenuations can have
            # some dynamic range.
            if uctuner.estimate_att < 4:
                uctuner.increase_tone_power()
                S.log(f"Adjusting tone power to {uctuner.current_tone_power}"
                            + f" and uc_att to {uctuner.current_uc_att}"
                )

            if uctuner.estimate_att > 26:
                uctuner.decrease_tone_power()
                S.log(f"Adjusting tone power to {uctuner.current_tone_power}"
                            + f" and uc_att to {uctuner.current_uc_att}"
                )

            uctuner.status_tune("fine")

            if uctuner.lowest_wl_index == 0 or uctuner.lowest_wl_index == -1:
                # Best noise was found at the edge of uc_att range explored;
                # re-center and repeat.
                uctuner.current_uc_att = uctuner.estimate_att
                uctuner.status_tune("fine")

        elif med_noise_thresh < uctuner.wl_median < high_noise_thresh:
            # Do up to two rough tunes followed by one or more fine tunes.
            uctuner.status_tune("rough")

            if uctuner.wl_median < low_noise_thresh:
                # Do one fine tune and stop.
                uctuner.status_tune("fine")

            else:
                # Do another rough tune.
                uctuner.status_tune("rough")

                # If needed, adjust the tone powers so the attenuations can have
                # some dynamic range.
                if uctuner.estimate_att < 4:
                    uctuner.increase_tone_power()
                    S.log(f"Adjusting tone power to {uctuner.current_tone_power}"
                                + f" and uc_att to {uctuner.current_uc_att}"
                    )

                if uctuner.estimate_att > 26:
                    uctuner.decrease_tone_power()
                    S.log(f"Adjusting tone power to {uctuner.current_tone_power}"
                                + f" and uc_att to {uctuner.current_uc_att}"
                    )

                uctuner.status_tune("fine")

                if uctuner.lowest_wl_index == 0 or uctuner.lowest_wl_index == -1:
                    # Best noise was found at the edge of uc_att range explored;
                    # re-center and repeat.
                    uctuner.current_uc_att = uctuner.estimate_att
                    uctuner.status_tune("fine")

        else:
            # wl_median above high_noise_thresh
            summary = (
                f"WL={uctuner.wl_median:.1f} is too high. "
                +"Power level might be really off, please investigate."
            )
            S.log(summary)
            return False, summary
            
        S.log(uctuner.status)
        S.log(f"Best noise {uctuner.best_wl:.1f} pA/rtHz achieved at"
              + f" uc att {uctuner.best_att} tone power {uctuner.best_tone}.")
        cfg.dev.update_band(
            opt_band,
            {"uc_att": uctuner.best_att, "tone_power": uctuner.best_tone},
            update_file=True,
        )
        return True

class UCTuner:
    """
    UCTuner is the object used to perform and hold the information from
    uc_att and tone power optimization, designed for use in the
    optimize_setup.tone_uc_att function.

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
        self.best_wl = None
        self.best_att = None
        self.best_tone = None

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
            am, outdict = noise.take_noise(self._S, self._cfg, acq_time=30, fit=False,
                                           plot_band_summary=False, show_plot=False)

            noise_param = outdict['noise_pars'][outdict['bands']==self.band][:,0]

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
        if self.best_wl is None or self.wl_median < self.best_wl:
            self.best_wl = wl_median
            self.best_tone = self.current_tone_power
            self.best_att = estimate_att
        self.wl_length = wl_len_list[lowest_wl_index]
        self.status = (f"WL: {self.wl_median:.1f} pA/rtHz with"
                       + f" {self.wl_length} channels.")

    def rough_tune(self):
        """
        Takes noise with current uc_att and uc_att +/-5 and +/- 10.
        """
        uc_attens = np.array(
            [
                self.estimate_att - 10,
                self.estimate_att - 5,
                self.estimate_att,
                self.estimate_att + 5,
                self.estimate_att + 10,
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
                self.estimate_att - 4,
                self.estimate_att - 2,
                self.estimate_att,
                self.estimate_att + 2,
                self.estimate_att + 4,
            ]
        )
        uc_attens = uc_attens[np.where((uc_attens <= 30) & (uc_attens >= 0))]

        self.uc_tune(uc_attens=uc_attens)

    def set_tone_and_uc(self, tone_power, uc_att):
        """
        Apply new tone power and uc_att settings.
        Re-runs find_freqs, setup notches, serial algs, and tracking.
        """
        self._S.amplitude_scale[self.band] = tone_power
        self._S.set_att_uc(self.band, uc_att)
        self._S.find_freq(self.band, tone_power=tone_power, make_plot=True)
        self._S.setup_notches(self.band, tone_power=tone_power,
                              new_master_assignment=True)
        self._S.run_serial_gradient_descent(self.band)
        self._S.run_serial_eta_scan(self.band)
        tracking_res = tracking.relock_tracking_setup(self._S, self._cfg, self.band)
        self.current_uc_att = uc_att
        self.current_tone_power = tone_power

    def increase_tone_power(self):
        """
        Increases tone power by 2 and uc_att by 11. Should result in
        approximately the same power at the resonators.
        """
        self.set_tone_and_uc(
            tone_power=self.current_tone_power + 2,
            uc_att=np.min([self.estimate_att + 11, 30])
        )

    def decrease_tone_power(self):
        """
        Decreases tone power by 2 and uc_att by 11. Should result in
        approximately the same power at the resonators.
        """
        self.set_tone_and_uc(
            tone_power=self.current_tone_power - 2,
            uc_att=np.max([self.estimate_att - 11, 0])
        )

    def status_tune(self, rof):
        ''' Declares rough/fine tunability and uctuner status, then tunes.
        rof = string, either "rough" or "fine", indicating tuning type.'''
        self._S.log(f"Can be {rof}-tuned")
        if rof == "fine":
            self.fine_tune()
        elif rof == "rough":
            self.rough_tune()
        else:
            raise ArgumentError(f'tune type must be "rough" or "fine"; was given "{rof}"')
        self._S.log(f"UC attens: {self.uc_attens}")
        self._S.log(f"Noise levels: {self.wl_list}")
        self._S.log(f"Number of active channels: {self.wl_length}")
