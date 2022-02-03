# uc_tuner.py
# The core code is from Princeton, created by Yuhan Wang and Daniel Dutcher Oct/Nov 2021.
# The code was refactored by Caleb Wheeler Nov 2021.

import os
import time
import numpy as np
import scipy.signal


class UCTuner:
    """
    This is a description of UCTuner

    Args
    ----

    Attributes
    ----------
    """
    def __init__(self, S, cfg, band=None, stream_time=20,
                 fmin=5., fmax=50., detrend='constant'):
        self._S = S
        self._cfg = cfg
        self.band = band
        self.stream_time = stream_time
        self.fmin = fmin
        self.fmax = fmax
        self.detrend = detrend
        self.current_tone_power = None
        self.estimate_att = None
        self.lowest_wl_index = None
        self.wl_median = None
        self.wl_length = None
        self.channel_length = None
        self.status = None

        if band is None:
            raise ValueError("Must specify `band` as int [0-7]")

        self.current_uc_att = S.get_att_uc(band)
        self.current_tone_power = S.amplitude_scale[band]

    
    def uc_tune(self, uc_attens, initial_attempt=False):
        """
        Find the uc_att from those provided that yields the lowest noise.
        """
        uc_attens = np.atleast_1d(uc_attens)
        wl_list = []
        wl_len_list = []
        noise_floors_list = []
        for atten in uc_attens:
            if not initial_attempt:
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

            dat_path = self._S.stream_data_on()
            # collect stream data
            time.sleep(self.stream_time)
            # end the time stream
            self._S.stream_data_off()
            fs = self._S.get_sample_frequency()
            wl_list_temp = []
            timestamp, phase, mask, tes_bias = self._S.read_stream_data(
                dat_path, return_tes_bias=True
            )

            bands, channels = np.where(mask != -1)
            phase *= self._S.pA_per_phi0 / (2.0 * np.pi)  # uA

            for c, (b, ch) in enumerate(zip(bands, channels)):
                if ch < 0:
                    continue
                ch_idx = mask[b, ch]
                nsamps = len(phase[ch_idx])
                f, Pxx = scipy.signal.welch(
                    phase[ch_idx], nperseg=nsamps, fs=fs, detrend=self.detrend
                )
                Pxx = np.sqrt(Pxx)
                fmask = (self.fmin < f) & (f < self.fmax)

                wl = np.median(Pxx[fmask])
                wl_list_temp.append(wl)

            noise_param = wl_list_temp

            wl_list.append(np.nanmedian(noise_param))
            wl_len_list.append(len(noise_param))
            noise_floors_list.append(np.median(noise_param))

        lowest_wl_index = wl_list.index(min(wl_list))
        if lowest_wl_index == len(wl_list):
            lowest_wl_index = -1
        estimate_att = uc_attens[lowest_wl_index]
        wl_median = wl_list[lowest_wl_index]

        self.estimate_att = estimate_att
        self.lowest_wl_index = lowest_wl_index
        self.wl_median = wl_median
        self.wl_length= wl_len_list[lowest_wl_index]
        if self.channel_length is None:
            self.channel_length = wl_len_list[lowest_wl_index]
        self.status = (f"WL: {self.wl_median:.1f} pA/rtHz with"
                       + f" {self.wl_length} channels out of {self.channel_length}")


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

