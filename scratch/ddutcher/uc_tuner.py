# uc_tuner.py
# The core code is from Princeton, created by Yuhan Wang and Daniel Dutcher Oct/Nov 2021.
# The code was refactored by Caleb Wheeler Nov 2021.

import os
import time
import numpy as np


def uc_tune(
    S,
    cfg,
    band,
    uc_attens,
    current_tone_power,
    stream_time=20.0,
    fmin=5,
    fmax=50,
    detrend="constant",
):
    """
    Find the uc_att from those provided that yields the lowest noise.
    """
    wl_list = []
    wl_len_list = []
    noise_floors_list = []
    channel_length = None
    for atten in uc_attens:
        S.set_att_uc(band, atten)
        S.tracking_setup(
            band,
            reset_rate_khz=cfg.dev.bands[band]["flux_ramp_rate_khz"],
            fraction_full_scale=cfg.dev.bands[band]["frac_pp"],
            make_plot=False,
            save_plot=False,
            show_plot=False,
            channel=S.which_on(band),
            nsamp=2 ** 18,
            lms_freq_hz=cfg.dev.bands[band]["lms_freq_hz"],
            meas_lms_freq=cfg.dev.bands[band]["meas_lms_freq"],
            feedback_start_frac=cfg.dev.bands[band]["feedback_start_frac"],
            feedback_end_frac=cfg.dev.bands[band]["feedback_end_frac"],
            lms_gain=cfg.dev.bands[band]["lms_gain"],
        )

        dat_path = S.stream_data_on()
        # collect stream data
        time.sleep(stream_time)
        # end the time stream
        S.stream_data_off()
        fs = S.get_sample_frequency()
        wl_list_temp = []
        timestamp, phase, mask, tes_bias = S.read_stream_data(
            dat_path, return_tes_bias=True
        )

        bands, channels = np.where(mask != -1)
        phase *= S.pA_per_phi0 / (2.0 * np.pi)  # uA
        import scipy.signal as signal

        for c, (b, ch) in enumerate(zip(bands, channels)):
            if ch < 0:
                continue
            ch_idx = mask[b, ch]
            nsamps = len(phase[ch_idx])
            f, Pxx = signal.welch(
                phase[ch_idx], nperseg=nsamps, fs=fs, detrend="constant"
            )
            Pxx = np.sqrt(Pxx)
            fmask = (fmin < f) & (f < fmax)

            wl = np.median(Pxx[fmask])
            wl_list_temp.append(wl)

        noise_param = wl_list_temp

        wl_list.append(np.nanmedian(noise_param))
        wl_len_list.append(len(noise_param))
        noise_floors_list.append(np.median(noise_param))
        # Only set this once in the first loop
        if channel_length is None:
            channel_length = len(noise_param)

    lowest_wl_index = wl_list.index(min(wl_list))
    if lowest_wl_index == len(wl_list):
        lowest_wl_index = -1
    estimate_att = uc_attens[lowest_wl_index]
    wl_median = wl_list[lowest_wl_index]

    return estimate_att, current_tone_power, lowest_wl_index, wl_median


def uc_rough_tune(
    S,
    cfg,
    band,
    current_uc_att,
    current_tone_power,
    stream_time=20.0,
    fmin=5,
    fmax=50,
    detrend="constant",
):
    """
    Takes noise with current uc_att and uc_att +/-5 and +/- 10.
    """
    uc_attens = np.array(
        [
            current_uc_att - 10,
            current_uc_att - 5,
            current_uc_att,
            current_uc_att + 5,
            current_uc_att + 10,
        ]
    )
    uc_attens = uc_attens[np.where((uc_attens <= 30) & (uc_attens >= 0))]

    estimate_att, current_tone_power, lowest_wl_index, wl_median = uc_tune(
        S=S,
        cfg=cfg,
        band=band,
        uc_attens=uc_attens,
        current_tone_power=current_tone_power,
        stream_time=stream_time,
        fmin=fmin,
        fmax=fmax,
        detrend=detrend,
    )
    return estimate_att, current_tone_power, lowest_wl_index, wl_median


def uc_fine_tune(
    S,
    cfg,
    band,
    current_uc_att,
    current_tone_power,
    stream_time=20.0,
    fmin=5,
    fmax=50,
    detrend="constant",
):
    """
    Takes noise with current uc_att and uc_att +/-2 and +/- 4.
    """
    uc_attens = np.array(
        [
            current_uc_att - 4,
            current_uc_att - 2,
            current_uc_att,
            current_uc_att + 2,
            current_uc_att + 4,
        ]
    )
    uc_attens = uc_attens[np.where((uc_attens <= 30) & (uc_attens >= 0))]

    estimate_att, current_tone_power, lowest_wl_index, wl_median = uc_tune(
        S=S,
        cfg=cfg,
        band=band,
        uc_attens=uc_attens,
        current_tone_power=current_tone_power,
        stream_time=stream_time,
        fmin=fmin,
        fmax=fmax,
        detrend=detrend,
    )
    return estimate_att, current_tone_power, lowest_wl_index, wl_median
