'''
Code written in Oct 2021 by Yuhan Wang
to be used through OCS
takes SC noise and takes IV
'''

import csv
import time
import argparse

import numpy as np
import matplotlib
matplotlib.use('Agg')


def uxm_bath_iv_noise(S, cfg, bands):
    for band in bands:
        S.run_serial_gradient_descent(band)
        S.run_serial_eta_scan(band)
        S.set_feedback_enable(band, 1)
        S.tracking_setup(
            band, reset_rate_khz=cfg.dev.bands[band]['flux_ramp_rate_khz'],
            fraction_full_scale=cfg.dev.bands[band]['frac_pp'], make_plot=False,
            save_plot=False, show_plot=False, channel=S.which_on(band),
            nsamp=2 ** 18, lms_freq_hz=None, meas_lms_freq=True,
            feedback_start_frac=cfg.dev.bands[band]['feedback_start_frac'],
            feedback_end_frac=cfg.dev.bands[band]['feedback_end_frac'],
            lms_gain=cfg.dev.bands[band]['lms_gain'],
        )

    bias_groups = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    S.set_filter_disable(0)
    S.set_rtm_arb_waveform_enable(0)
    S.set_downsample_factor(20)
    for bias_index, bias_g in enumerate(bias_groups):
        S.set_tes_bias_low_current(bias_g)

    bias_v = 0
    bias_array = np.zeros(S._n_bias_groups)
    for bg in bias_groups:
        bias_array[bg] = bias_v
    S.set_tes_bias_bipolar_array(bias_array)
    time.sleep(120)
    datafile_self = S.stream_data_on()
    time.sleep(120)
    S.stream_data_off()

    fieldnames = ['bath_temp', 'bias_voltage', 'bias_line', 'band', 'data_path', 'type']
    row = {}
    row['data_path'] = datafile_self
    row['bias_voltage'] = bias_v
    row['type'] = 'noise'
    row['bias_line'] = 'all'
    row['band'] = 'all'
    row['bath_temp'] = bath_temp
    with open(out_fn, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow(row)

    for bias_gp in bias_groups:
        row = {}
        row['bath_temp'] = bath_temp
        row['bias_line'] = bias_gp
        row['band'] = 'all'
        row['bias_voltage'] = 'IV 20 to 0'
        row['type'] = 'IV'
        print(f'Taking IV on bias line {bias_gp}, all band')

        row['data_path'] = S.run_iv(
            bias_groups=[bias_gp],
            wait_time=0.001,
            bias_high=20,
            bias_low=0,
            bias_step=0.025,
            overbias_voltage=18,
            cool_wait=30,
            high_current_mode=False,
            make_plot=False,
            save_plot=True,
            #        cool_voltage=18.
        )

        with open(out_fn, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow(row)

        time.sleep(6)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--slot', type=int)
    parser.add_argument('--temp', type=str)
    parser.add_argument('--output_file', type=str)
