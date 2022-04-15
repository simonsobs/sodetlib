# noise_stack_by_band.py

import os
import matplotlib
import numpy as np
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pysmurf.client.util.pub import set_action
import sodetlib as sdl
from sodetlib import noise
import logging

logger = logging.getLogger(__name__)

"""
@set_action()
def noise_stack_by_band(S, cfg, acq_time=30.0, wl_f_range=(10, 30), nperseg=1024):
    logger.info(f"taking {acq_time}s timestream")

    sid = sdl.take_g3_data(S, acq_time)
    am = sdl.load_session(cfg.stream_id, sid, base_dir=cfg.sys['g3_dir'])
    noisedict = noise.get_noise_params(
        am, wl_f_range=wl_f_range, fit=False, nperseg=nperseg)
    bands = am.ch_info.band
    wls = noisedict['noise_pars'][:,0]
    fknees = noisedict['noise_pars'][:,2]
    band_medians = noisedict['band_medians']

    #Plot white noise histograms
    fig_wnl, axes_wnl = plt.subplots(4, 2, figsize=(16, 8),
                             gridspec_kw={'hspace': 0})
    fig_wnl.patch.set_facecolor('white')
    max_bins = 0

    for b in range(8):
        ax = axes_wnl[b % 4, b // 4]
        m = bands == b
        x = ax.hist(wls[m], range=(0,300), bins=60)
        text  = f"Median: {band_medians[b]:0.1f} pA/rtHz\n"
        text += f"Chans pictured: {np.sum(x[0]):0.0f}"
        ax.text(0.7, .7, text, transform=ax.transAxes)
        ax.axvline(np.median(wls[m]), color='red', alpha=0.6)
        max_bins = max(np.max(x[0]), max_bins)
        ax.set(ylabel=f'Band {b}')
        ax.grid(linestyle='--')

    axes_wnl[0][0].set(title="AMC 0")
    axes_wnl[0][1].set(title="AMC 1")
    axes_wnl[-1][0].set(xlabel="White Noise (pA/rt(Hz))")
    axes_wnl[-1][1].set(xlabel="White Noise (pA/rt(Hz))")
    for _ax in axes_wnl:
        for ax in _ax:
            ax.set(ylim=(0, max_bins * 1.1))
    plt.suptitle(
        f'Total yield {len(wls)}, Overall median noise {np.nanmedian(wls):0.1f} pA/rtHz')
    ctime = int(am.timestamps[0])
    savename = os.path.join(S.plot_dir, f'{ctime}_white_noise_summary.png')
    plt.savefig(savename)
    plt.close()
    S.pub.register_file(savename, "smurfband_noise", plot=True)
    logger.info(f"plotting directory is:\n{S.plot_dir}")

    outdict = {'sid': sid, 'noisedict': noisedict, 'stream_id': cfg.stream_id,
               'base_dir': cfg.sys['g3_dir']}
    savename = os.path.join(S.output_dir, f'{ctime}_take_noise.npy')
    np.save(savename, outdict, allow_pickle=True)
    S.pub.register_file(savename, "smurfband_noise", format='npy')

    #Plot ASDs
    fig_wnl, axes_wnl = plt.subplots(4, 2, figsize=(16, 8),
                             gridspec_kw={'hspace': 0})
    fig_wnl.patch.set_facecolor('white')

    for b in range(8):
        ax = axes_wnl[b % 4, b // 4]
        m = bands == b
        med_wl = np.nanmedian(wls[m])
        f_arr = np.tile(noisedict['f'], (sum(m),1))
        x = ax.loglog(f_arr.T, noisedict['axx'][m].T, color='C0', alpha=0.1)
        ax.axvline(1.4, linestyle='--', alpha=0.6, color='C1')
        ax.axvline(60, linestyle='--', alpha=0.6, color='C2')
        ax.axhline(med_wl, color='red', alpha=0.6,
                   label=f'Med. WL: {med_wl:.1f} pA/rtHz')
        ax.set(ylabel=f'ASD (pA/rtHz)')
        ax.grid(linestyle='--', which='both')
        ax.legend(loc='upper right')

    axes_wnl[0][0].set(title="AMC 0")
    axes_wnl[0][1].set(title="AMC 1")
    axes_wnl[-1][0].set(xlabel="Frequency (Hz)")
    axes_wnl[-1][1].set(xlabel="Frequency (Hz)")
    for _ax in axes_wnl:
        for ax in _ax:
            ax.set(ylim=[1, 5e3])
    savename = os.path.join(S.plot_dir, f'{ctime}_band_asds.png')
    plt.savefig(savename)
    S.pub.register_file(savename, "smurfband_noise", plot=True)
    logger.info(f"plotting directory is:\n{S.plot_dir}")
"""

if __name__ == "__main__":
    import argparse
    from sodetlib.det_config import DetConfig

    cfg = DetConfig()
    parser = argparse.ArgumentParser()

    parser.add_argument(
    "--acq-time",
    type=float,
    default=30.0,
    help="float, optional, default is 30.0. The amount of time to sleep in seconds while "
    + "streaming SMuRF data for analysis.",
)

    args = cfg.parse_args(parser)
    S = cfg.get_smurf_control(dump_configs=False, make_logfile=True)
    S.load_tune(cfg.dev.exp['tunefile'])

    nsamps = S.get_sample_frequency() * args.acq_time
    nperseg = 2 ** round(np.log2(nsamps / 5))
    noise.take_noise(
        S, cfg, acq_time=args.acq_time, show_plot=False, save_plot=True, nperseg=nperseg,
    )
