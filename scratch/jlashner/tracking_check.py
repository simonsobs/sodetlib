import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import argparse
import numpy as np
from sodetlib.det_config import DetConfig
from sodetlib.util import cprint, make_filename, get_tracking_kwargs
from pysmurf.client.util.pub import set_action


@set_action()
def get_tracking_goodness(S, cfg, band, tracking_kwargs=None,
                          make_channel_plots=False, r_thresh=0.9):
    """
    Runs tracking setup and returns how good at tracking each channel is

    Args
    -----
        S : SmurfControl
            Pysmurf control object
        cfg : DetConfig
            Detconfig object
        band : int
            band number
        tracking_kwargs : dict
            Dictionary of additional custom args to pass to tracking setup
        r_thresh : float
            Threshold used to set color on plots
    Returns
    --------
        rs : np.ndarray
            Array of size (512) containing values between 0 and 1 which tells
            you how good a channel is at tracking. If close to 1, the channel
            is tracking well and if close to 0 the channel is tracking poorly
        f : np.ndarray
            f as returned from tracking setup
        df : np.ndarray
            df as returned by tracking setup
    """
    band_cfg = cfg.dev.bands[band]
    tk = get_tracking_kwargs(S, cfg, band, kwargs=tracking_kwargs)
    tk['nsamp'] = 2**20  # moreee data
    tk['show_plot'] = False  # Override

    f, df, sync = S.tracking_setup(band, **tk)
    si = S.make_sync_flag(sync)
    nphi0 = int(round(band_cfg['lms_freq_hz'] / S.get_flux_ramp_freq()/1000))

    active_chans = np.zeros_like(f[0], dtype=bool)
    active_chans[S.which_on(band)] = True

    # Average cycles to get single period estimate
    seg_size = (si[1] - si[0]) // nphi0
    fstack = np.zeros((seg_size, len(f[0])))
    nstacks = (len(si)-1) * nphi0
    for i in range(len(si) - 1):
        s = si[i]
        for j in range(nphi0):
            a = s + seg_size * j
            fstack += f[a:a + seg_size, :]
    fstack /= nstacks

    # calculates quality of estimate wrt real data
    y_real = f[si[0]:si[-1], :]
    # Averaged cycle repeated nstack times
    y_est = np.vstack([fstack for _ in range(nstacks)])
    sstot = np.sum((y_real - np.mean(y_real, axis=0))**2, axis=0)
    ssres = np.sum((y_real - y_est)**2, axis=0)

    r = 1 - ssres/sstot
    # Probably means it's a bugged debug channels.
    r[np.isnan(r) & active_chans] = 1

    fname = make_filename(S, 'tracking_goodness.png', plot=True)
    fig, ax = plt.subplots()
    ax.hist(r[active_chans], bins=30)
    ax.axvline(r_thresh, linestyle=':', alpha=0.8)
    text_props = {
        'transform': ax.transAxes, 'fontsize': 11, 'verticalalignment': 'top',
        'bbox': {'facecolor': 'white'}
    }
    props = {'facecolor': 'white'}
    num_good = np.sum(r > r_thresh)
    num_active = np.sum(active_chans)
    s = f"{num_good}/{num_active} Channels above r={r_thresh}"
    ax.text(0.05, 0.95, s, **text_props)
    ax.set(xlabel="Tracking Quality", ylabel="Num Channels",
           title=f"Band {b} Tracking Quality")
    plt.savefig(fname)
    S.pub.register_file(fname, 'tracking_goodness', plot=True)

    if make_channel_plots:
        print("Making channel plots....")
        nramps = 2
        xs = np.arange(len(f))
        m = (si[1] - 20 < xs) & (xs < si[1 + nramps] + 20)
        for chan in S.which_on(band):
            fig, ax = plt.subplots()
            c = 'C1' if r[chan] > 0.85 else 'black'
            ax.plot(xs[m], f[m, chan], color=c)
            props = {'facecolor': 'white'}
            ax.text(0.05, 0.95, f"r={r[chan]:.3f}", transform=ax.transAxes,
                    fontsize=15, verticalalignment="top", bbox=props)
            ax.set_title(f"Band {b} Channel {chan}")
            fname = make_filename(S, f"tracking_b{b}_c{chan}.png", plot=True)
            fig.savefig(fname)
            plt.close(fig)

    return r, f, df, sync


if __name__ == '__main__':
    cfg = DetConfig()
    parser = argparse.ArgumentParser()
    parser.add_argument('--bands', '-b', type=int, nargs='+', required=True)
    parser.add_argument('--threshold', '-t', type=float, default=0.9)
    parser.add_argument('--plots', '-p', action='store_true')
    args = cfg.parse_args(parser)
    S = cfg.get_smurf_control(dump_configs=True)

    for b in args.bands:
        rs, f, df, sync = get_tracking_goodness(S, cfg, b,
                                                make_channel_plots=args.plots,
                                                r_thresh=args.threshold)
        nchans = len(S.which_on(b))
        good_chans = np.where(rs > args.threshold)[0]
        cprint(f"{len(good_chans)} / {nchans} have passed on band {b}",
               True)
        cprint(f"Good chans:\n{good_chans}", True)
