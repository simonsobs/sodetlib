import numpy as np
import time
import sys
import sodetlib as sdl
from sodetlib import noise
from tqdm.auto import trange
import matplotlib.pyplot as plt

def plot_optimize_attens(S, summary, wlmax=1000, vmin=None, vmax=None):
    """
    Plots the results from the optimize_attens functions.
    """
    wls = summary['wl_medians']
    grid = summary['atten_grid']
    shape = wls[0].shape

    xs = np.reshape(grid[:, 0], shape)
    ys = np.reshape(grid[:, 1], shape)

    ucs = summary['uc_attens']
    dcs = summary['dc_attens']

    fig, axes = plt.subplots(2, 4, figsize=(18, 6),
                             gridspec_kw={'hspace': 0.4, 'wspace': 0.4})
    fig.patch.set_facecolor('white')

    if len(ucs) == 1:
        fig.suptitle(f"Median White Noise, UC Atten = {ucs[0]}")
    elif len(dcs) == 1:
        fig.suptitle(f"Median White Noise, DC Atten = {dcs[0]}")

    if vmin is None:
        vmin = np.min(wls) * 0.9
    if vmax is None:
        vmax = min(np.max(wls) * 1.1, 500)  # We don't rly care if wl > 800

    for i, band in enumerate(summary['bands']):
        ax = axes[band // 4, band % 4]

        if len(ucs) == 1:  # Sweep is dc only
            uc = ucs[0]
            _wls = wls[i, 0, :]
            ax.plot(dcs, _wls)
            ax.set(xlabel="DC atten", ylabel="white noise (pA/rt(Hz))")
            ax.set(title=f"Band {band}")
            ax.set(ylim=(vmin, vmax))

        elif len(dcs) == 1:  # Sweep is uc only
            dc = dcs[0]
            _wls = wls[i, :, 0]
            ax.plot(ucs, _wls)
            ax.set(xlabel="UC atten", ylabel="white noise (pA/rt(Hz))")
            ax.set(title=f"Band {band}")
            ax.set(ylim=(vmin, vmax))

        else:  # Do full 2d heat map
            im = ax.pcolor(ucs, dcs, wls[i].T, vmin=vmin, vmax=vmax)
            ax.set(xlabel="UC atten", ylabel="DC atten", title=f"Band {band}")
            if i == 0:
                fig.colorbar(im, label='Median White Noise [pA/rt(Hz)]',
                             ax=axes.ravel().tolist())

        fname = su.make_filename(S, f'atten_sweep_b{band}.png', plot=True)
        fig.savefig(fname)


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
    nbands = len(bands)

    if uc_attens is None:
        uc_attens = np.arange(30, -2, -2)
    uc_attens = np.atleast_1d(uc_attens)

    if dc_attens is None:
        dc_attens = np.arange(30, -2, -2)
    dc_attens = np.atleast_1d(dc_attens)

    nsteps = len(uc_attens) * len(dc_attens)
    ucs = np.zeros(nsteps)
    dcs = np.zeros(nsteps)
    # Creates uc/dc grid with dc being the fast index and uc being the slow idx
    i = 0
    for uc in enumerate(uc_attens):
        for dc in enumerate(dc_attens):
            ucs[i] = uc
            dcs[i] = dc
            i += 1

    out = {}
    # Run bands to not conflict with ``bands`` key required by normalized
    # datafiles
    out['run_bands'] = bands  
    out['band_medians'] = np.full((8, nsteps), np.inf)
    out['sids'] = np.zeros(nsteps, dtype=int)
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
        am, outdict = noise.take_noise(S, cfg, meas_time, show_plot=False)
        out['stop_times'][i] = time.time()
        out['sid'][i] = out
        if 'bands' not in out:
            out['bands'] = am.ch_info.band
            out['channels'] = am.ch_info.channel
            out['wls'] = np.full((len(am.signal), nsteps), np.inf)

        out['wls'] = outdict['noisedict']['noise_pars'][:, 0]
        out['band_medians'][i] = out['noisedict']['band_medians']

        S.log(f"Band medians for uc,dc = {uc},{dc}: {out['band_medians'][i]}")

    opt_ucs = np.full((8, ), np.nan)
    opt_dcs = np.full((8, ), np.nan)
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

    if set_to_opt:
        for i, b in enumerate(bands):
            ("Setting attens to opt values and re-running setup notches "
                      f"for band {b}...")
            S.set_att_uc(b, opt_ucs[i])
            S.set_att_dc(b, opt_dcs[i])
            S.setup_notches(b, tone_power=tone_power,
                            new_master_assignment=False)
            S.run_serial_gradient_descent(b)
            S.run_serial_eta_scan(b)
            S.tracking_setup(b, **tks[b])

    return out, path
