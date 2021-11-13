'''
Code written in Oct 2021 by Yuhan Wang
only suitable for UFMs when TESes are in normal stage
instead of fitting to noise model, this takes median noise from 5Hz to 50Hz
different noise levels here are based on phase 2 noise target and noise model after considering johnson noise at 100mK
'''
import matplotlib
matplotlib.use('Agg')

import pysmurf.client
import argparse
import numpy as np
import os
import time
import glob
from sodetlib.det_config  import DetConfig
import numpy as np
from scipy.interpolate import interp1d
import argparse
import time
import csv
import scipy.signal as signal



band = 4
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
    lms_freq_hz=cfg.dev.bands[band]["lms_freq_hz"],
    meas_lms_freq=cfg.dev.bands[band]["meas_lms_freq"],
    feedback_start_frac=cfg.dev.bands[band]["feedback_start_frac"],
    feedback_end_frac=cfg.dev.bands[band]["feedback_end_frac"],
    lms_gain=cfg.dev.bands[band]["lms_gain"],
)
print("checking tracking")
S.check_lock(
    band,
    reset_rate_khz=cfg.dev.bands[band]["flux_ramp_rate_khz"],
    fraction_full_scale=cfg.dev.bands[band]["frac_pp"],
    lms_freq_hz=cfg.dev.bands[band]["lms_freq_hz"],
    feedback_start_frac=cfg.dev.bands[band]["feedback_start_frac"],
    feedback_end_frac=cfg.dev.bands[band]["feedback_end_frac"],
    lms_gain=cfg.dev.bands[band]["lms_gain"],
)
S.overbias_tes_all(bias_groups = [0,1,2,3,4,5,6,7,8,9,10,11], overbias_wait=1, tes_bias= 12, cool_wait= 3, high_current_mode=False, overbias_voltage= 12)
print("waiting for thermal environment get stablized")
time.sleep(120)

print("taking 20s timestream")
stream_time = 20


# non blocking statement to start time stream and return the dat filename
dat_path = S.stream_data_on()
# collect stream data
time.sleep(stream_time)
# end the time stream
S.stream_data_off()

fmin=5
fmax=50
wl_list_temp = []
timestamp, phase, mask, tes_bias = S.read_stream_data(dat_path,
        return_tes_bias=True)

bands, channels = np.where(mask != -1)
phase *= S.pA_per_phi0 / (2.0 * np.pi)  # uA
sample_nums = np.arange(len(phase[0]))
fs = 200
nperseg=2**12
detrend='constant'
t_array = sample_nums / fs
for c, (b, ch) in enumerate(zip(bands, channels)):
    if ch < 0:
        continue
    ch_idx = mask[b, ch]
    sampleNums = np.arange(len(phase[ch_idx]))
    t_array = sampleNums / fs
    f, Pxx = signal.welch(phase[ch_idx], nperseg=nperseg,
        fs=fs, detrend=detrend)
    Pxx = np.sqrt(Pxx)
    fmask = (fmin < f) & (f < fmax)

    wl = np.median(Pxx[fmask])
    wl_list_temp.append(wl)



noise_param = wl_list_temp

wl_median = np.median(noise_param)
wl_length = len(noise_param)
channel_length = len(noise_param)
noise_floors = np.median(noise_param)
print('wl_median')

def rough_tune(current_uc_att, current_tune_power, band,slot_num):

    cfg = DetConfig()
    cfg.load_config_files(slot=slot_num)
    S = cfg.get_smurf_control()

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
            lms_freq_hz=cfg.dev.bands[band]["lms_freq_hz"],
            meas_lms_freq=cfg.dev.bands[band]["meas_lms_freq"],
            feedback_start_frac=cfg.dev.bands[band]["feedback_start_frac"],
            feedback_end_frac=cfg.dev.bands[band]["feedback_end_frac"],
            lms_gain=cfg.dev.bands[band]["lms_gain"],
        )

        dat_path = S.stream_data_on()
        # collect stream data
        import time
        stream_time = 20
        time.sleep(stream_time)
        # end the time stream
        S.stream_data_off()


        fmin=5
        fmax=50
        nperseg=2**12
        detrend='constant'
        wl_list_temp = []
        timestamp, phase, mask, tes_bias = S.read_stream_data(dat_path,
                return_tes_bias=True)

        bands, channels = np.where(mask != -1)
        phase *= S.pA_per_phi0 / (2.0 * np.pi)  # uA
        fs = 200
        import scipy.signal as signal
        for c, (b, ch) in enumerate(zip(bands, channels)):
            if ch < 0:
                continue
            ch_idx = mask[b, ch]
            sampleNums = np.arange(len(phase[ch_idx]))
            t_array = sampleNums / fs
            
            f, Pxx = signal.welch(phase[ch_idx], nperseg=nperseg,
                fs=fs, detrend=detrend)
            Pxx = np.sqrt(Pxx)
            fmask = (fmin < f) & (f < fmax)

            wl = np.median(Pxx[fmask])
            wl_list_temp.append(wl)

        noise_param = wl_list_temp

        wl_list.append(np.nanmedian(noise_param))
        wl_len_list.append(len(noise_param))
        noise_floors_list.append(np.median(noise_param))
        channel_length = len(noise_param)

    lowest_wl_index = wl_list.index(min(wl_list))
    estimate_att = attens[lowest_wl_index]
    wl_median = wl_list[lowest_wl_index]
    print(
        "lowest WL: {} with {} channels".format(
            wl_median, channel_length
        )
    )

    return estimate_att, current_tune_power, lowest_wl_index, wl_median


def fine_tune(current_uc_att, current_tune_power, band,slot_num):

    cfg = DetConfig()
    cfg.load_config_files(slot=slot_num)
    S = cfg.get_smurf_control()

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
            lms_freq_hz=cfg.dev.bands[band]["lms_freq_hz"],
            meas_lms_freq=cfg.dev.bands[band]["meas_lms_freq"],
            feedback_start_frac=cfg.dev.bands[band]["feedback_start_frac"],
            feedback_end_frac=cfg.dev.bands[band]["feedback_end_frac"],
            lms_gain=cfg.dev.bands[band]["lms_gain"],
        )

        dat_path = S.stream_data_on()
        # collect stream data
        import time
        stream_time = 20
        time.sleep(stream_time)
        # end the time stream
        S.stream_data_off()
        
        fmin=5
        fmax=50
        nperseg=2**12
        detrend='constant'
        wl_list_temp = []
        timestamp, phase, mask, tes_bias = S.read_stream_data(dat_path,
                return_tes_bias=True)

        bands, channels = np.where(mask != -1)
        phase *= S.pA_per_phi0 / (2.0 * np.pi)  # uA
        sample_nums = np.arange(len(phase[0]))
        fs = 200
        t_array = sample_nums / fs
        import scipy.signal as signal
        for c, (b, ch) in enumerate(zip(bands, channels)):
            if ch < 0:
                continue
            ch_idx = mask[b, ch]
            sampleNums = np.arange(len(phase[ch_idx]))
            t_array = sampleNums / fs

            f, Pxx = signal.welch(phase[ch_idx], nperseg=nperseg,
                fs=fs, detrend=detrend)
            Pxx = np.sqrt(Pxx)
            fmask = (fmin < f) & (f < fmax)

            wl = np.median(Pxx[fmask])
            wl_list_temp.append(wl)
        noise_param = wl_list_temp

        wl_list.append(np.nanmedian(noise_param))
        wl_length = len(noise_param)
        channel_length = len(noise_param)
        noise_floors = np.median(noise_param)

    print(wl_list)
    lowest_wl_index = wl_list.index(min(wl_list))
    print(lowest_wl_index)
    estimate_att = attens[lowest_wl_index]
    wl_median = wl_list[lowest_wl_index]
    print(
        "lowest WL: {} with {} channels".format(
            wl_median, channel_length
        )
    )

    return estimate_att, current_tune_power, lowest_wl_index, wl_median


if wl_median > 150:
    print(
        "WL: {} with {} channels out of {}".format(wl_median, wl_length, channel_length)
    )
    print(
        "something might be wrong, power level might be really off, please investigate"
    )


if wl_median < 60:
    print(
        "WL: {} with {} channels out of {}".format(wl_median, wl_length, channel_length)
    )
    print("can be fine tuned")

    current_uc_att = S.get_att_uc(band)
    current_tune_power = S.amplitude_scale[band]

    estimate_att, current_tune_power, lowest_wl_index, wl_median = fine_tune(
        current_uc_att, current_tune_power, band, slot_num
    )

    print("achieved at uc att {} drive {}".format(estimate_att, current_tune_power))


if wl_median > 60 and wl_median < 120:

    print(
        "WL: {} with {} channels out of {}".format(wl_median, wl_length, channel_length)
    )
    print("can be fine tuned")

    current_uc_att = S.get_att_uc(band)
    current_tune_power = S.amplitude_scale[band]

    estimate_att, current_tune_power, lowest_wl_index,wl_median = rough_tune(
        current_uc_att, current_tune_power, band,slot_num
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
        new_tune_power = current_tune_power - 2
        adjusted_uc_att = current_uc_att - 11
        S.set_att_uc(band, adjusted_uc_att)
        S.find_freq(band, tone_power=new_tune_power, make_plot=True)
        S.setup_notches(band, tone_power=new_tune_power, new_master_assignment=True)
        S.run_serial_gradient_descent(band)
        S.run_serial_eta_scan(band)
        current_uc_att = adjusted_uc_att
        current_tune_power = new_tune_power

    estimate_att, current_tune_power, lowest_wl_index,wl_median = fine_tune(
        current_uc_att, current_tune_power, band,slot_num
    )
    print("achieved at uc att {} drive {}".format(estimate_att, current_tune_power))
    step2_index = lowest_wl_index

    if step2_index == 0:
        print("can be fine tuned")
        estimate_att, current_tune_power, lowest_wl_index,wl_median = fine_tune(
            current_uc_att - 4, current_tune_power, band,slot_num
        )

    print("achieved at uc att {} drive {}".format(estimate_att, current_tune_power))


if wl_median > 120 and wl_median < 150:

    print(
        "WL: {} with {} channels out of {}".format(wl_median, wl_length, channel_length)
    )
    print("can be fine tuned")

    current_uc_att = S.get_att_uc(band)
    current_tune_power = S.amplitude_scale[band]

    estimate_att, current_tune_power, lowest_wl_index,wl_median = rough_tune(
        current_uc_att, current_tune_power, band,slot_num
    )

    if wl_median < 60:
        print(
            "WL: {} with {} channels out of {}".format(
                wl_median, wl_length, channel_length
            )
        )
        print("can be fine tuned")

        current_uc_att = S.get_att_uc(band)
        current_tune_power = S.amplitude_scale[band]

        estimate_att, current_tune_power, lowest_wl_index,wl_median = fine_tune(
            current_uc_att, current_tune_power, band,slot_num
        )






        print("achieved at uc att {} drive {}".format(estimate_att, current_tune_power))

    if wl_median > 60:

        print(
            "WL: {} with {} channels out of {}".format(
                wl_median, wl_length, channel_length
            )
        )
        print("can be fine tuned")

        current_uc_att = S.get_att_uc(band)
        current_tune_power = S.amplitude_scale[band]


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
            new_tune_power = current_tune_power - 2
            adjusted_uc_att = current_uc_att - 11
            S.set_att_uc(band, adjusted_uc_att)
            S.find_freq(band, tone_power=new_tune_power, make_plot=True)
            S.setup_notches(band, tone_power=new_tune_power, new_master_assignment=True)
            S.run_serial_gradient_descent(band)
            S.run_serial_eta_scan(band)
            current_uc_att = adjusted_uc_att
            current_tune_power = new_tune_power

        estimate_att, current_tune_power, lowest_wl_index,wl_median = rough_tune(
            current_uc_att, current_tune_power, band,slot_num
        )
        step1_index = lowest_wl_index

        # if estimate_att < 16:
        #     print("adjusting tune power and uc att")
        #     new_tune_power = current_tune_power + 2
        #     adjusted_uc_att = current_uc_att + 12
        #     S.set_att_uc(band, adjusted_uc_att)
        #     S.find_freq(band, tone_power=new_tune_power, make_plot=True)
        #     S.setup_notches(band, tone_power=new_tune_power, new_master_assignment=True)
        #     S.run_serial_gradient_descent(band)
        #     S.run_serial_eta_scan(band)
        #     current_uc_att = adjusted_uc_att
        #     current_tune_power = new_tune_power

        # if estimate_att > 26:
        #     print("adjusting tune power and uc att")
        #     new_tune_power = current_tune_power + 2
        #     adjusted_uc_att = current_uc_att - 11
        #     S.set_att_uc(band, adjusted_uc_att)
        #     S.find_freq(band, tone_power=new_tune_power, make_plot=True)
        #     S.setup_notches(band, tone_power=new_tune_power, new_master_assignment=True)
        #     S.run_serial_gradient_descent(band)
        #     S.run_serial_eta_scan(band)
        #     current_uc_att = adjusted_uc_att
        #     current_tune_power = new_tune_power

        estimate_att, current_tune_power, lowest_wl_index, wl_median= fine_tune(
            current_uc_att, current_tune_power, band,slot_num
        )
        print("achived at uc att {} drive {}".format(estimate_att, current_tune_power))
        step2_index = lowest_wl_index

        if step2_index == 0 and step1_index == 0:
            print("can be fine tuned")
            estimate_att, current_tune_power, lowest_wl_index,wl_median = fine_tune(
                current_uc_att - 4, current_tune_power, band,slot_num
            )

        if step2_index == 4 and step1_index == 4:
            print("can be fine tuned")
            estimate_att, current_tune_power, lowest_wl_index,wl_median = fine_tune(
                current_uc_att + 4, current_tune_power, band,slot_num
            )

        print("achived at uc att {} drive {}".format(estimate_att, current_tune_power))

try:
    print("WL: {} with {} channels out of {}".format(wl_median, wl_length, channel_length))
    print("achived at uc att {} drive {}".format(estimate_att, current_tune_power))
except:
    print('WL is off, please investigate')
print("plotting directory is:")
print(S.plot_dir)


