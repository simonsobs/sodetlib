'''
Code written in early 2021 by Yuhan Wang
only suitable for optimizing one band of none TES coupled resonator channels
different noise levels here are based on phase 2 noise target
'''

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pysmurf.client
import argparse
import numpy as np
import pickle as pkl
from scipy import signal
import os
import time
from sodetlib.det_config import DetConfig


band = 3
slot_num = 3

cfg = DetConfig()
cfg.load_config_files(slot=slot_num)
S = cfg.get_smurf_control()

print("plotting directory is:")
print(S.plot_dir)

S.all_off()
S.set_rtm_arb_waveform_enable(0)
S.set_filter_disable(0)
S.set_downsample_factor(20)
S.set_mode_dc()


print("setting up band {}".format(band))

S.set_att_dc(band, cfg.dev.bands[band]["dc_att"])
print("band {} dc_att {}".format(band, S.get_att_dc(band)))

S.set_att_uc(band, cfg.dev.bands[band]["uc_att"])
print("band {} uc_att {}".format(band, S.get_att_uc(band)))

S.amplitude_scale[band] = cfg.dev.bands[band]["drive"]
print("band {} tone power {}".format(band, S.amplitude_scale[band]))

print("estimating phase delay")
S.estimate_phase_delay(band)
print("setting synthesis scale")
# hard coding it for the current fw
S.set_synthesis_scale(band, 1)
print("running find freq")
S.find_freq(band, tone_power=cfg.dev.bands[band]["drive"], make_plot=True)
print("running setup notches")
S.setup_notches(
    band, tone_power=cfg.dev.bands[band]["drive"], new_master_assignment=True
)
print("running serial gradient descent and eta scan")
S.run_serial_gradient_descent(band)
S.run_serial_eta_scan(band)
print("running tracking setup")
S.set_feedback_enable(band, 1)
S.tracking_setup(
    band,
    reset_rate_khz=cfg.dev.bands[band]["flux_ramp_rate_khz"],
    fraction_full_scale=cfg.dev.bands[band]["frac_pp"],
    make_plot=False,
    save_plot=False,
    show_plot=False,
    channel=S.which_on(band),
    nsamp=2 ** 18,
    lms_freq_hz=None,
    meas_lms_freq=True,
    feedback_start_frac=cfg.dev.bands[band]["feedback_start_frac"],
    feedback_end_frac=cfg.dev.bands[band]["feedback_end_frac"],
    lms_gain=cfg.dev.bands[band]["lms_gain"],
)
print("checking tracking")
S.check_lock(
    band,
    reset_rate_khz=cfg.dev.bands[band]["flux_ramp_rate_khz"],
    fraction_full_scale=cfg.dev.bands[band]["frac_pp"],
    lms_freq_hz=None,
    feedback_start_frac=cfg.dev.bands[band]["feedback_start_frac"],
    feedback_end_frac=cfg.dev.bands[band]["feedback_end_frac"],
    lms_gain=cfg.dev.bands[band]["lms_gain"],
)

print("taking 20s timestream")
datafile, noise_param = S.take_noise_psd(
    20,
    nperseg=2 ** 16,
    save_data=True,
    make_channel_plot=False,
    return_noise_params=True,
)

wl_median = np.median(noise_param[3])
wl_length = len(noise_param[3])
channel_length = len(noise_param[0])
noise_floors = np.median(noise_param[1])


def rough_tune(current_uc_att, current_tune_power, band):

    attens = [
        current_uc_att - 10,
        current_uc_att - 5,
        current_uc_att,
        current_uc_att + 5,
        current_uc_att + 10,
    ]
    wl_list = []
    wl_len_list = []
    noise_floors_list = []

    for atten in attens:
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
            lms_freq_hz=None,
            meas_lms_freq=True,
            feedback_start_frac=cfg.dev.bands[band]["feedback_start_frac"],
            feedback_end_frac=cfg.dev.bands[band]["feedback_end_frac"],
            lms_gain=cfg.dev.bands[band]["lms_gain"],
        )

        datafile, noise_param = S.take_noise_psd(
            20,
            nperseg=2 ** 16,
            save_data=True,
            make_channel_plot=False,
            return_noise_params=True,
        )
        wl_list.append(np.median(noise_param[3]))
        wl_len_list.append(len(noise_param[3]))
        noise_floors_list.append(np.median(noise_param[1]))
        channel_length = len(noise_param[0])

    lowest_wl_index = wl_list.index(min(wl_list))
    estimate_att = attens[lowest_wl_index]
    wl_median = wl_list[lowest_wl_index]
    print(
        "lowest WL: {} with {} channels out of {}".format(
            wl_median, wl_len_list[lowest_wl_index], channel_length
        )
    )

    return estimate_att, current_tune_power, lowest_wl_index, wl_median


def fine_tune(current_uc_att, current_tune_power, band):
    band = band
    attens = [
        current_uc_att - 4,
        current_uc_att - 2,
        current_uc_att,
        current_uc_att + 2,
        current_uc_att + 4,
    ]
    wl_list = []
    wl_len_list = []
    noise_floors_list = []

    for atten in attens:
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
            lms_freq_hz=None,
            meas_lms_freq=True,
            feedback_start_frac=cfg.dev.bands[band]["feedback_start_frac"],
            feedback_end_frac=cfg.dev.bands[band]["feedback_end_frac"],
            lms_gain=cfg.dev.bands[band]["lms_gain"],
        )

        datafile, noise_param = S.take_noise_psd(
            20,
            nperseg=2 ** 16,
            save_data=True,
            make_channel_plot=False,
            return_noise_params=True,
        )
        wl_list.append(np.median(noise_param[3]))
        wl_len_list.append(len(noise_param[3]))
        noise_floors_list.append(np.median(noise_param[1]))
        channel_length = len(noise_param[0])

    lowest_wl_index = wl_list.index(min(wl_list))
    estimate_att = attens[lowest_wl_index]
    wl_median = wl_list[lowest_wl_index]
    print(
        "lowest WL: {} with {} channels out of {}".format(
            wl_median, wl_len_list[lowest_wl_index], channel_length
        )
    )

    return estimate_att, current_tune_power, lowest_wl_index, wl_median


if wl_median > 200:
    print(
        "WL: {} with {} channels out of {}".format(wl_median, wl_length, channel_length)
    )
    print(
        "something might be wrong, power level might be really off, please investigate"
    )


if wl_median < 45:
    print(
        "WL: {} with {} channels out of {}".format(wl_median, wl_length, channel_length)
    )
    print("can be fine tuned")

    current_uc_att = S.get_att_uc(band)
    current_tune_power = S.amplitude_scale[band]

    estimate_att, current_tune_power, lowest_wl_index, wl_median = fine_tune(
        current_uc_att, current_tune_power, band
    )

    print("achieved at uc att {} drive {}".format(estimate_att, current_tune_power))


if wl_median > 45 and wl_median < 65:

    print(
        "WL: {} with {} channels out of {}".format(wl_median, wl_length, channel_length)
    )
    print("can be fine tuned")

    current_uc_att = S.get_att_uc(band)
    current_tune_power = S.amplitude_scale[band]

    estimate_att, current_tune_power, lowest_wl_index,wl_median = rough_tune(
        current_uc_att, current_tune_power, band
    )

    if estimate_att < 16:
        print("adjusting tune power and uc att")
        new_tune_power = current_tune_power + 2
        adjusted_uc_att = current_uc_att + 11
        S.set_att_uc(band, adjusted_uc_att)
        S.find_freq(band, tone_power=new_tune_power, make_plot=True)
        S.setup_notches(band, tone_power=new_tune_power, new_master_assignment=True)
        S.run_serial_gradient_descent(band)
        S.run_serial_eta_scan(band)
        current_uc_att = adjusted_uc_att
        current_tune_power = new_tune_power

    if estimate_att > 26:
        print("adjusting tune power and uc att")
        new_tune_power = current_tune_power + 2
        adjusted_uc_att = current_uc_att - 11
        S.set_att_uc(band, adjusted_uc_att)
        S.find_freq(band, tone_power=new_tune_power, make_plot=True)
        S.setup_notches(band, tone_power=new_tune_power, new_master_assignment=True)
        S.run_serial_gradient_descent(band)
        S.run_serial_eta_scan(band)
        current_uc_att = adjusted_uc_att
        current_tune_power = new_tune_power

    estimate_att, current_tune_power, lowest_wl_index,wl_median = fine_tune(
        current_uc_att, current_tune_power, band
    )
    print("achieved at uc att {} drive {}".format(estimate_att, current_tune_power))
    step2_index = lowest_wl_index

    if step2_index == 0:
        print("can be fine tuned")
        estimate_att, current_tune_power, lowest_wl_index,wl_median = fine_tune(
            current_uc_att - 4, current_tune_power, band
        )

    print("achieved at uc att {} drive {}".format(estimate_att, current_tune_power))


if wl_median > 65 and wl_median < 200:

    print(
        "WL: {} with {} channels out of {}".format(wl_median, wl_length, channel_length)
    )
    print("can be fine tuned")

    current_uc_att = S.get_att_uc(band)
    current_tune_power = S.amplitude_scale[band]

    estimate_att, current_tune_power, lowest_wl_index,wl_median = rough_tune(
        current_uc_att, current_tune_power, band
    )

    if wl_median < 45:
        print(
            "WL: {} with {} channels out of {}".format(
                wl_median, wl_length, channel_length
            )
        )
        print("can be fine tuned")

        current_uc_att = S.get_att_uc(band)
        current_tune_power = S.amplitude_scale[band]

        estimate_att, current_tune_power, lowest_wl_index,wl_median = fine_tune(
            current_uc_att, current_tune_power, band
        )

        print("achieved at uc att {} drive {}".format(estimate_att, current_tune_power))

    if wl_median > 45:

        print(
            "WL: {} with {} channels out of {}".format(
                wl_median, wl_length, channel_length
            )
        )
        print("can be fine tuned")

        current_uc_att = S.get_att_uc(band)
        current_tune_power = S.amplitude_scale[band]

        estimate_att, current_tune_power, lowest_wl_index,wl_median = rough_tune(
            current_uc_att, current_tune_power, band
        )
        step1_index = lowest_wl_index

        if estimate_att < 16:
            print("adjusting tune power and uc att")
            new_tune_power = current_tune_power + 2
            adjusted_uc_att = current_uc_att + 12
            S.set_att_uc(band, adjusted_uc_att)
            S.find_freq(band, tone_power=new_tune_power, make_plot=True)
            S.setup_notches(band, tone_power=new_tune_power, new_master_assignment=True)
            S.run_serial_gradient_descent(band)
            S.run_serial_eta_scan(band)
            current_uc_att = adjusted_uc_att
            current_tune_power = new_tune_power

        if estimate_att > 26:
            print("adjusting tune power and uc att")
            new_tune_power = current_tune_power + 2
            adjusted_uc_att = current_uc_att - 11
            S.set_att_uc(band, adjusted_uc_att)
            S.find_freq(band, tone_power=new_tune_power, make_plot=True)
            S.setup_notches(band, tone_power=new_tune_power, new_master_assignment=True)
            S.run_serial_gradient_descent(band)
            S.run_serial_eta_scan(band)
            current_uc_att = adjusted_uc_att
            current_tune_power = new_tune_power

        estimate_att, current_tune_power, lowest_wl_index, wl_median= fine_tune(
            current_uc_att, current_tune_power, band
        )
        print("achived at uc att {} drive {}".format(estimate_att, current_tune_power))
        step2_index = lowest_wl_index

        if step2_index == 0 and step1_index == 0:
            print("can be fine tuned")
            estimate_att, current_tune_power, lowest_wl_index,wl_median = fine_tune(
                current_uc_att - 4, current_tune_power, band
            )

        if step2_index == 4 and step1_index == 4:
            print("can be fine tuned")
            estimate_att, current_tune_power, lowest_wl_index,wl_median = fine_tune(
                current_uc_att + 4, current_tune_power, band
            )

        print("achived at uc att {} drive {}".format(estimate_att, current_tune_power))

try:
    print("WL: {} with {} channels out of {}".format(wl_median, wl_length, channel_length))
    print("achived at uc att {} drive {}".format(estimate_att, current_tune_power))
except:
    print('WL is off, please investigate')
print("plotting directory is:")
print(S.plot_dir)

