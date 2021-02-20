import numpy as np
import os
import time
from scipy import signal
import matplotlib.pyplot as plt
import scipy.optimize as opt
from scipy import interpolate
import pickle as pkl
from sodetlib.util import cprint, TermColors, get_psd, make_filename, get_tracking_kwargs

from pysmurf.client.util.pub import set_action
from sodetlib.det_config import DetConfig


@set_action()
def optimize_uc_atten(S, cfg, bands, meas_time=10, fit_curve=False,
                      attens=None, skip_setup_notches=False, tone_power=None, 
                      silence_logs=True, tracking_kwargs=None):
    """
    Finds the drive power and uc attenuator value that minimizes the median
    noise within a band.

    Parameters
    ----------
    bands : (int, list(int))
        Bands ofr which to optimize uc_atten
    meas_time : float
        Measurement time for noise PSD in seconds. Defaults to 10 sec.
    fit_curve: bool
        If True, will fit
    run_setup_notches : bool
        If True, will run setup notches after each uc_atten step.

    Returns
    -------
    min_med_noise : float
        The median noise at the optimized drive power
    atten : int
        Optimized uc attenuator value
    """
    if isinstance(bands, (int, float)):
        bands = [bands]
    bands = np.array(bands)

    if attens is None:
        attens = np.arange(30, -2, -2)

    wl_medians = np.full((len(bands), len(attens)), np.inf)
    dat_files = []

    start_time = time.time()
    if silence_logs:
        logfile = make_filename(S, 'optimize_uc_atten.log')
        print(f"Pysmurf logs being written to {logfile}")
        S.set_logfile(logfile)

    tks = {}
    for b in bands:
        print(f"Setting uc atten to {attens[0]} and running setup-notches "
              f"for band {b}")
        S.set_att_uc(b, attens[0])
        if not skip_setup_notches:
            S.setup_notches(b, tone_power=tone_power,
                            new_master_assignment=False)
        tks[b] = get_tracking_kwargs(S, cfg, b, kwargs=tracking_kwargs)
        tks[b].update({
            'return_data': False,
            'make_plot': False,
            'save_plot': False,
        })

    for i, atten in enumerate(attens):
        cprint(f'Setting UC atten to: {atten}')

        for b in bands:
            print(f"Band {b}, uc-atten {atten}: Runnig serial gradient "
                  "descent and eta scan")
            S.set_att_uc(b, atten)
            S.run_serial_gradient_descent(b)
            S.run_serial_eta_scan(b)
            S.tracking_setup(b, **tks[b])

        datafile = S.take_stream_data(meas_time, make_freq_mask=False)
        dat_files.append(datafile)
        times, phase, mask = S.read_stream_data(datafile)
        f, Pxx = get_psd(S, times, phase)
        fmask = (f > 5) & (f < 50)
        wls = np.nanmedian(Pxx[:, fmask], axis=1)

        for j, b in enumerate(bands):
            rchans = mask[b, mask[b] != -1]
            wl_medians[j, i] = np.nanmedian(wls[rchans])
        print(f"Median noise for uc_att={atten}: {wl_medians[:, i]}")

    stop_time = time.time()
    S.set_logfile(None)
    summary = {
        'start': start_time,
        'stop': stop_time,
        'attens': attens,
        'bands': bands,
        'dat_files': dat_files,
        'wl_medians': wl_medians
    }
    fname = make_filename(S, 'optimize_uc_atten_summary.npy')
    np.save(fname, summary, allow_pickle=True)
    S.pub.register_file(fname, 'optimize_uc_atten_summary', format='npy')
    return summary


if __name__ == '__main__':
    cfg = DetConfig()
    cfg.load_config_files(slot=3)
    S = cfg.get_smurf_control()

    summary_file = optimize_uc_atten(S, cfg, range(8), meas_time=30,
                                     tone_power=12,
                                     tracking_kwargs={'lms_gain': 2})
    print(f"Summary saved to {summary_file}")

