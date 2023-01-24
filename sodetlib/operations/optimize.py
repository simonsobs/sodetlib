from sodetlib.operations import uxm_relock, tracking, uxm_setup
from sodetlib import noise
import sodetlib as sdl
import numpy as np
import time
from tqdm.auto import tqdm
import matplotlib.pyplot as plt


def find_min_total_atten(S, band, atten_offset=2):
    """
    Finds the minimum total atten level (UC + DC) required for the ADCs not to
    be saturated. This assumes tones are already enabled.

    Args
    -----
    S : SmurfControl
        Pysmurf instance
    band : int
        Band number
    atten_offset : int
        Int to add to total attens to try to avoid ADC saturation.
    """
    uc, dc = 0, 0
    S.set_att_uc(band, 0)
    S.set_att_dc(band, 0)

    for uc in range(0, 31, 2):
        S.set_att_uc(band, uc)
        if not S.check_adc_saturation(band):
            return min(uc + dc + atten_offset, 60)
        time.sleep(0.05)

    for dc in range(0, 31, 2):
        S.set_att_dc(band, dc)
        if not S.check_adc_saturation(band):
            return min(uc + dc + atten_offset, 60)
        time.sleep(0.05)

    return False

@sdl.set_action()
def optimize_band_atten(S, cfg, band, meas_time=30, 
    total_att=None, ucs=None, show_pb=False, bgs=None, full_tune=False):
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
    bgs : list
        List of bias lines to overbias. This must at least contain detectors on
        the band that is being optimized. If None is specified, will just
        overbias all detectors at each step.
    full_tune : bool
        If True, will run ``setup_tune``, which performs find freq and setup
        otches, at each step. This is the way to get the most accurate
        optimization, as eta and the resonance frequency change as you adjust
        the uc attenuation. If your uc step is small enough (1 or maybe 2), then
        the gradient descent and eta scan functions may be sufficient and you
        might not need to do a full tune at each step.
    """

    if total_att is None:
        total_att = find_min_total_atten(S, band)
        S.log(f"Total att for band {band}: {total_att}")

    if bgs is None:
        bgs = cfg.dev.exp['active_bgs']
    bgs = np.atleast_1d(bgs)

    if ucs is None:
        ucs = np.arange(0, 31, 3)

    nsteps = len(ucs)
    sids = np.zeros(nsteps)
    wls = np.full(nsteps, np.nan)

    wls_full = [None for _ in range(nsteps)]
    tunefiles = [None for _ in range(nsteps)]

    for i, uc in enumerate(tqdm(ucs, disable=not show_pb)):
        dc = max(total_att - uc, 0)
        if dc <= 30:
            S.set_att_uc(band, uc)
            S.set_att_dc(band, dc)
        else:
            continue

        S.log(f"Running UC={uc}")

        S.flux_ramp_off()
        S.set_flux_ramp_dac(0)
        for bg in bgs:
            S.set_tes_bias_bipolar(bg, 0)

        if full_tune:
            uxm_setup.setup_tune(S, cfg, bands=band, update_cfg=False)
        else:
            uxm_relock.run_grad_descent_and_eta_scan(S, cfg, bands=[band])

        tunefiles[i] = S.tune_file
        tracking.relock_tracking_setup(S, cfg, band)
        S.overbias_tes_all(bgs)
        _, res = noise.take_noise(S, cfg, meas_time, plot_band_summary=False, show_plot=False, save_plot=False)
        sids[i] = res['sid']
        wls[i] = res['band_medians'][band]
        wls_full[i] = res['noise_pars'][:, 0]

    fname = sdl.make_filename(S, f'atten_optimization_b{band}.npy')
    wls_full = np.array(wls_full)
    data = dict(
        band=band, total_att=total_att, ucs=ucs, sids=sids, wls=wls,
        wls_full=wls_full, path=fname, tunefiles=tunefiles, meta=sdl.get_metadata(S, cfg)
    )

    np.save(fname, data, allow_pickle=True)
    S.log(f"Saving {fname}")
    S.pub.register_file(fname, 'atten_optimzation', format='npy')

    fig, _ = plot_atten_optimization(data)
    fname = sdl.make_filename(S, 'atten_optimization.png', plot=True)
    S.log(f"Saving {fname}")
    fig.savefig(fname)
    S.pub.register_file(fname, 'atten_optimzation', format='png')

    return data


def plot_atten_optimization(data, ax=None, ylim=None, text_loc=(0.02, 0.03)):
    """
    Plots the results of the optimize_attens function.

    Args
    -----
    data : dict
        Results from the ``optimize_band_atten`` function
    ax : Matplotlib axis
        Axis to plot on. If None this will create a new figure.
    ylim : tuple
        ylim to set
    text_loc : tuple
        Location of the textbox
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    ax.plot(data['ucs'], data['wls'])

    imin = np.nanargmin(data['wls'])
    opt_uc = data['ucs'][imin]
    opt_dc = max(data['total_att'] - opt_uc, 0)
    ax.plot([opt_uc], [data['wls'][imin]], marker='x', color='red')
    ax.set_title(f"Band {data['band']}, total_att={data['total_att']}")
    txt = '\n'.join([
        f"Min = {data['wls'][imin]:.1f} pA/rt(Hz)",
        f"Opt (UC, DC) = ({opt_uc}, {opt_dc})"
    ])
    ax.text(*text_loc, txt, transform=ax.transAxes, 
            bbox=dict(facecolor='wheat', alpha=0.2))
    if ylim is not None:
        ax.set_ylim(*ylim)

    return fig, ax
