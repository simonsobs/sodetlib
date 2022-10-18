from sodetlib.operations import uxm_relock, tracking
from sodetlib import noise
import sodetlib as sdl
import numpy as np
import time
from tqdm.auto import tqdm
import matplotlib.pyplot as plt


def find_min_total_atten(S, band):
    """
    Finds the minimum total atten level (UC + DC) required for the ADCs not to
    be saturated. This assumes tones are already enabled.

    Args
    -----
    S : SmurfControl
        Pysmurf instance
    band : int
        Band number
    """
    uc, dc = 0, 0
    S.set_att_uc(band, 0)
    S.set_att_dc(band, 0)

    for uc in range(0, 31, 2):
        S.set_att_uc(band, uc)
        if not S.check_adc_saturation(band):
            return uc + dc
        time.sleep(0.05)

    for dc in range(0, 31, 2):
        S.set_att_dc(band, dc)
        if not S.check_adc_saturation(band):
            return uc + dc
        time.sleep(0.05)

    return False


@sdl.set_action()
def optimize_attens(S, cfg, bands, meas_time=30, total_atts=None, ucs=None, show_pb=False):
    """
    This function will optimize values for uc and dc attenuators. First, it
    will find the minimum total attenuation required to avoid saturating ADCs.
    Then it loops through values of UC attenuators (keeping the total atten
    constant), runs gradient descent, tracking setup, and the takes noise.
    It will chose the optimal attenuation to minimize the white noise level.
    After this is run, you should re-run grad-descent and tracking functions
    before taking data.

    Args
    -----
    S : SmurfControl
        Pysmurf instance
    cfg : DetConfig
        DetConfig instance
    meas_time : float
        Duration (sec) for noise measurement
    total_atts : list
        Total attenuations for each specified band. If this is not passed,
        it will calculate this automatically
    ucs : list
        List of UC attenuations to loop over
    show_pb : bool
        If True, will display a progress bar.
    """
    bands = np.atleast_1d(bands)
    if total_atts is None:
        total_atts = []
        for b in bands:
            att = find_min_total_atten(S, b)
            S.log(f"Total att for band {b}: {att}")
            total_atts.append(att)
    total_atts = np.array(total_atts)
    S.log(f"Total atts: {total_atts}")

    if ucs is None:
        ucs = np.arange(30, -1, -2)

    nsteps = len(ucs)
    sids = np.zeros(nsteps)
    wls = np.full((len(bands), nsteps), np.nan)
    wls_full = []

    for i, uc in enumerate(tqdm(ucs, disable=not show_pb)):
        active_bidxs = []
        for bidx, b in enumerate(bands):
            dc = max(total_atts[bidx] - uc, 0)
            if dc <= 30:
                S.set_att_uc(b, uc)
                S.set_att_dc(b, dc)
                active_bidxs.append(bidx)
        if not active_bidxs:
            continue

        S.log(f"Running UC={uc}")
        S.flux_ramp_off()
        S.set_flux_ramp_dac(0)

        uxm_relock.run_grad_descent_and_eta_scan(S, cfg, bands, update_tune=False)
        tracking.relock_tracking_setup(S, cfg, bands)
        _, res = noise.take_noise(S, cfg, meas_time, plot_band_summary=False, show_plot=False, save_plot=False)
        sids[i] = res['sid']
        wls[active_bidxs, i] = res['band_medians'][bands[active_bidxs]]
        wls_full.append(res['noise_pars'][0])

    wls_full = np.array(wls_full)
    data = dict(
        bands=bands, total_atts=total_atts, ucs=ucs, sids=sids, wls=wls,
        wls_full=wls_full
    )

    fname = sdl.make_filename(S, 'atten_optimization.npy')
    np.save(fname, data, allow_pickle=True)
    S.log(f"Saving {fname}")
    S.pub.register_file(fname, 'atten_optimzation', format='npy')

    S.log("Optimal attens")
    for bidx, band in enumerate(bands):
        bcfg = cfg.dev.bands[band]
        imin = np.nanargmin(wls[bidx])
        opt_uc = ucs[imin]
        opt_dc = total_atts[bidx] - opt_uc
        S.log(f"Band {band}, (uc, dc) = ({opt_uc}, {opt_dc}); wl = {wls[bidx, imin]:.2f} pA/rt(Hz")
        bcfg['uc_att'] = opt_uc
        bcfg['dc_att'] = opt_dc
        S.set_att_uc(band, opt_uc)
        S.set_att_dc(band, opt_dc)

    fig, _ = plot_atten_optimization(data)
    fname = sdl.make_filename(S, 'atten_optimization.png', plot=True)
    S.log(f"Saving {fname}")
    fig.savefig(fname)
    S.pub.register_file(fname, 'atten_optimzation', format='png')

    return data


def plot_atten_optimization(data):
    """
    Plots the results of the optimize_attens function.
    """
    fig, axes = plt.subplots(2, 4, figsize=(16, 10), gridspec_kw={'hspace': 0.2})
    _axes = axes.ravel()

    for i, b in enumerate(data['bands']):
        wls = data['wls'][i]
        _axes[b].plot(data['ucs'], wls)
        _axes[b].set_title(f"Band {b}, total att = {data['total_atts'][i]}")
        imin = np.nanargmin(data['wls'][i])
        _axes[b].plot([data['ucs'][imin]], [wls[imin]], 'x', color='red', markersize=10)
        txt = f"Min = {wls[imin]:.1f} pA/rt(Hz)"
        _axes[b].text(0.02, 0.03, txt, transform=_axes[b].transAxes, bbox=dict(facecolor='wheat', alpha=0.2))

    for ax in _axes:
        ax.set(xlabel='UC atten', xlim=(0, 30), ylim=(0, 200))

    return fig, axes
