import numpy as np
import time
import sys
import sodetlib as sdl
from sodetlib import noise
from tqdm.auto import trange
import matplotlib.pyplot as plt
from pysmurf.client.util.pub import set_action


def plot_optimize_attens(data, wlmax=1000, vmin=None, vmax=None,
                         save_path=None):
    ucs = data['ucs']
    dcs = data['dcs']
    nucs = len(np.unique(ucs))
    ndcs = len(np.unique(dcs))
    ucs = ucs.reshape(nucs, ndcs)
    dcs = dcs.reshape(nucs, ndcs)
    meds = data['band_medians'].reshape(8, nucs, ndcs)

    if vmin is None:
        vmin = np.min(meds) * 0.9
    if vmax is None:
        vmax = min(np.max(meds) * 1.1, 500)  # We don't rly care if wl > 800

    fig, axes = plt.subplots(2, 4, figsize=(24, 12))
    fig.patch.set_facecolor('white')

    bbox={'facecolor': 'wheat', 'alpha': 0.9}
    for band, ax in enumerate(axes.ravel()):
        if band not in data['run_bands']:
            continue

        if nucs == 1:  # Sweep is dc only
            uc = ucs[0, 0]
            ax.plot(dcs[0, :], meds[band, 0, :])
            ax.set(xlabel="DC atten", ylabel="white noise (pA/rt(Hz))")
            ax.set(title=f"Band {band}")
            ax.set(ylim=(vmin, vmax))

        elif ndcs == 1:  # Sweep is uc only
            dc = dcs[0, 0]
            ax.plot(ucs[:, 0], meds[band, :, 0])
            ax.set(xlabel="UC atten", ylabel="white noise (pA/rt(Hz))")
            ax.set(title=f"Band {band}")
            ax.set(ylim=(vmin, vmax))

        else:  # Do full 2d heat map
            im = ax.pcolor(ucs, dcs, meds[band], vmin=vmin, vmax=vmax,
                           shading='auto')
            ax.set(xlabel="UC atten", ylabel="DC atten", title=f"Band {band}")
            txt = '\n'.join([
                f"Opt UC: {data['opt_ucs'][band]}",
                f"Opt DC: {data['opt_dcs'][band]}",
                f"Opt noise: {data['opt_wls'][band]:0.2f}"
            ])
            ax.text(0.05, 0.05, txt, transform=ax.transAxes, bbox=bbox, fontsize=13)
            if band == 0:
                fig.colorbar(im, label='Median White Noise [pA/rt(Hz)]',
                             ax=axes)

    if save_path is not None:
        fig.savefig(save_path)

    return fig, axes


@set_action()
def optimize_attens(S, cfg, bands, meas_time=10, uc_attens=None,
                    dc_attens=None, tone_power=None, silence_logs=True,
                    tracking_kwargs=None, skip_setup_notches=False,
                    setup_notches_every_step=False,
                    update_cfg=True, set_to_opt=True):
    """
    UC and DC attenuation optimization function, built to work efficiently
    with multiple bands.

    Args
    ----
    S : SmurfControl
        Pysmurf control object
    cfg : DetConfig
        Det Config instance
    meas_time : float
        Measurement time (sec) for white noise analysis
    tone_power : int
        Tone power to use for scan.
    silence_logs : bool
        If true will send pysmurf logs to file instead of stdout to declutter
        logs.
    tracking_kwargs : dict
        Custom tracking kwargs to pass to tracking setup
    skip_setup_notches : bool
        If true, will skip the initial setup notches at the start of the
        optimization. This is not recommended unless you are just testing the
        base functionality
    update_cfg : bool
        If true, will update the cfg object with the new atten values
    set_to_opt : bool
        If true, will set atten to the optimal values afterwards and run setup
        functions.
    """
    bands = np.atleast_1d(bands)

    if uc_attens is None:
        uc_attens = np.arange(30, -2, -2)
    uc_attens = np.sort(np.atleast_1d(uc_attens))

    if dc_attens is None:
        dc_attens = np.arange(30, -2, -2)
    dc_attens = np.sort(np.atleast_1d(dc_attens))

    nsteps = len(uc_attens) * len(dc_attens)
    ucs = np.zeros(nsteps, dtype=int)
    dcs = np.zeros(nsteps, dtype=int)
    # Creates uc/dc grid with dc being the fast index and uc being the slow idx
    i = 0
    for uc in uc_attens:
        for dc in dc_attens:
            ucs[i] = uc
            dcs[i] = dc
            i += 1

    out = {}
    # Run bands to not conflict with ``bands`` key required by normalized
    # datafiles
    out['run_bands'] = bands
    out['band_medians'] = np.full((8, nsteps), np.inf)
    out['sid'] = np.zeros(nsteps, dtype=int)
    out['start_times'] = np.zeros(nsteps, dtype=float)
    out['stop_times'] = np.zeros(nsteps, dtype=float)
    out['ucs'] = ucs
    out['dcs'] = dcs

    logs_silenced = False
    logfile = None
    if S.log.logfile != sys.stdout:
        logfile = S.log.logfile.name
    elif silence_logs:
        logfile = sdl.make_filename(S, 'optimize_atten.log')
        print(f"Writing pysmurf logs to {logfile}")
        S.set_logfile(logfile)
        logs_silenced = True

    S.log("-" * 60)
    S.log("Atten optimization plan")
    S.log(f"bands: {bands}")
    S.log(f"uc_attens: {uc_attens}")
    S.log(f"dc_attens: {dc_attens}")
    S.log(f"logfile: {logfile}")
    S.log("-" * 60)

    tks = {}
    for b in bands:
        tks[b] = sdl.get_tracking_kwargs(S, cfg, b, kwargs=tracking_kwargs)
        tks[b].update({
            'return_data': False, 'make_plot': False, 'save_plot': False
        })

    for i in trange(nsteps):
        uc, dc = ucs[i], dcs[i]
        S.log(f"Step {i} / {nsteps},  uc: {uc}, dc: {dc}")
        new_uc = uc != S.get_att_uc(bands[0])
        run_setup_notches = new_uc or setup_notches_every_step

        for b in bands:
            S.set_att_uc(b, uc)
            S.set_att_dc(b, dc)
            if skip_setup_notches:
                continue

            if run_setup_notches:
                S.setup_notches(b, tone_power=tone_power,
                                new_master_assignment=False)

            S.run_serial_gradient_descent(b)
            S.run_serial_eta_scan(b)
            S.tracking_setup(b, **tks[b])


        out['start_times'][i] = time.time()
        am, outdict = noise.take_noise(
            S, cfg, meas_time, show_plot=False, plot_band_summary=False
        )
        out['stop_times'][i] = time.time()
        out['sid'][i] = outdict['sid']
        if 'bands' not in out:
            out['bands'] = am.ch_info.band
            out['channels'] = am.ch_info.channel

        out['band_medians'][:, i] = outdict['noisedict']['band_medians']

        S.log(f"Band medians for uc,dc = {uc},{dc}: {out['band_medians'][:, i]}")

    opt_ucs = np.zeros(8, dtype=int)
    opt_dcs = np.zeros(8, dtype=int)
    opt_wls = np.full((8, ), np.inf)
    for b in bands:
        idx = np.argmin(out['band_medians'][b])
        opt_ucs[b] = ucs[idx]
        opt_dcs[b] = dcs[idx]
        opt_wls[b] = out['band_medians'][b, idx]
        if update_cfg:
            cfg.dev.update_band(b, {
                'uc_att': ucs[idx],
                'dc_att': dcs[idx],
            }, update_file=True)

    S.log(f"Optimal UCS: {opt_ucs}")
    S.log(f"Optimal DCS: {opt_dcs}")
    S.log(f"Optimal wls: {opt_wls}")
    out['opt_ucs'] = opt_ucs
    out['opt_dcs'] = opt_dcs
    out['opt_wls'] = opt_wls

    if logs_silenced:  # Returns logs to stdout
        S.set_logfile(None)

    path = sdl.validate_and_save('optimize_atten_summary.npy', out, S=S, cfg=cfg)

    fig_path = sdl.make_filename(S, 'optimize_atten_summary.png', plot=True)
    fig, axes = plot_optimize_attens(out, save_path=fig_path)
    S.pub.register_file(fig_path, '', format='png', plot=True)

    if set_to_opt:
        for i, b in enumerate(bands):
            S.log("Setting attens to opt values and re-running setup notches "
                 f"for band {b}...")
            S.set_att_uc(b, int(opt_ucs[i]))
            S.set_att_dc(b, int(opt_dcs[i]))
            if not skip_setup_notches:
                S.setup_notches(b, tone_power=tone_power,
                                new_master_assignment=False)
                S.run_serial_gradient_descent(b)
                S.run_serial_eta_scan(b)
                S.tracking_setup(b, **tks[b])

    return out, path
