# noise_stack_by_band.py

import os
import time
import matplotlib
import numpy as np
from scipy import signal
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pysmurf.client.util.pub import set_action
import logging

logger = logging.getLogger(__name__)


@set_action()
def noise_stack_by_band(
        S,
        stream_time=20.0,
        fmin=5,
        fmax=50,
        detrend='constant',
):
    logger.info(f"taking {stream_time}s timestream")

    # non blocking statement to start time stream and return the dat filename
    dat_path = S.stream_data_on()
    # collect stream data
    time.sleep(stream_time)
    # end the time stream
    S.stream_data_off()

    start_time = dat_path[-14:-4]

    timestamp, phase, mask, tes_bias = S.read_stream_data(
        dat_path, return_tes_bias=True
    )
    logger.info(f"loaded the .dat file at: {dat_path}")

    # hard coded variables
    fs = S.get_sample_frequency()
    bands, channels = np.where(mask != -1)
    phase *= S.pA_per_phi0 / (2.0 * np.pi)  # uA
    sample_nums = np.arange(len(phase[0]))
    t_array = sample_nums / fs

    # reorganize the data by band then channel
    stream_by_band_by_channel = {}
    for band, channel in zip(bands, channels):
        if band not in stream_by_band_by_channel.keys():
            stream_by_band_by_channel[band] = {}
        ch_idx = mask[band, channel]
        stream_by_band_by_channel[band][channel] = phase[ch_idx]

    # plot the band channel data
    fig, axs = plt.subplots(4, 2, figsize=(12, 24), dpi=50)
    for band in sorted(stream_by_band_by_channel.keys()):
        wl_list_temp = []
        stream_single_band = stream_by_band_by_channel[band]
        ax_this_band = axs[band // 2, band % 2]
        for channel in sorted(stream_single_band.keys()):
            stream_single_channel = stream_single_band[channel]
            nsamps = len(stream_single_channel)
            f, Pxx = signal.welch(
                stream_single_channel, fs=fs, detrend=detrend, nperseg=nsamps
            )
            Pxx = np.sqrt(Pxx)
            fmask = (fmin < f) & (f < fmax)
            wl = np.median(Pxx[fmask])
            wl_list_temp.append(wl)
            stream_single_channel_norm = stream_single_channel - np.mean(
                stream_single_channel
            )
            ax_this_band.plot(
                t_array, stream_single_channel_norm, color="C0", alpha=0.002
            )
        wl_median = np.median(wl_list_temp)
        band_yield = len(stream_single_band)
        ax_this_band.set_xlabel("time [s]")
        ax_this_band.set_ylabel("Phase [pA]")
        ax_this_band.grid()
        ax_this_band.set_title(
            f"band {band} yield {band_yield} median noise {wl_median:.2f}"
        )
        ax_this_band.set_ylim([-10000, 10000])

    save_name = os.path.join(S.plot_dir, f"{start_time}_band_noise_stack.png")
    logger.info(f"Saving plot to {save_name}")
    plt.savefig(save_name)
    S.pub.register_file(save_name, "smurfband_noise", plot=True)

    fig, axs = plt.subplots(4, 4, figsize=(24, 24), dpi=50)
    for band in sorted(stream_by_band_by_channel.keys()):
        wl_list_temp = []
        stream_single_band = stream_by_band_by_channel[band]
        ax_this_band = axs[band // 2, band % 2 * 2]
        for channel in sorted(stream_single_band.keys()):
            stream_single_channel = stream_single_band[channel]
            nsamps = len(stream_single_channel)
            f, Pxx = signal.welch(
                stream_single_channel, fs=fs, detrend=detrend, nperseg=nsamps
            )
            Pxx = np.sqrt(Pxx)
            fmask = (fmin < f) & (f < fmax)
            wl = np.median(Pxx[fmask])
            wl_list_temp.append(wl)
            ax_this_band.loglog(f, Pxx, color="C0", alpha=0.02)

        wl_median = np.median(wl_list_temp)

        band_yield = len(stream_single_band)
        ax_this_band.set_xlabel("Frequency [Hz]")
        ax_this_band.set_ylabel("Amp [pA/rtHz]")
        ax_this_band.grid()
        ax_this_band.axvline(1.4, linestyle="--", alpha=0.6, label="1.4 Hz", color="C1")
        ax_this_band.axvline(60, linestyle="--", alpha=0.6, label="60 Hz", color="C2")
        ax_this_band.set_title(f"band {band} yield {band_yield}")
        ax_this_band.set_ylim([1, 5e3])

        ax_this_band_2 = axs[band // 2, band % 2 * 2 + 1]
        ax_this_band_2.set_xlabel("Amp [pA/rtHz]")
        ax_this_band_2.set_ylabel("count")
        ax_this_band_2.hist(wl_list_temp, range=(0, 300), bins=60)
        ax_this_band_2.axvline(wl_median, linestyle="--", color="gray")
        ax_this_band_2.grid()
        ax_this_band_2.set_title(
            f"band {band} yield {band_yield} median noise {wl_median:.2f}"
        )
        ax_this_band_2.set_xlim([0, 300])

    save_name = os.path.join(S.plot_dir, f"{start_time}_band_psd_stack.png")
    logger.info(f"Saving plot to {save_name}")
    plt.savefig(save_name)
    S.pub.register_file(save_name, "smurfband_noise", plot=True)

    logger.info(f"plotting directory is:\n{S.plot_dir}")


if __name__ == "__main__":
    import argparse
    from sodetlib.det_config import DetConfig

    cfg = DetConfig()
    args = cfg.parse_args()
    S = cfg.get_smurf_control(dump_configs=False, make_logfile=True)

    noise_stack_by_band(
        S,
        stream_time=20.0,
        fmin=5,
        fmax=50,
        detrend='constant',
    )
