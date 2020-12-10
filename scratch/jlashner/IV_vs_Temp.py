import matplotlib
matplotlib.use('Agg')

import argparse
import numpy as np
import os
import time
import glob
import pickle as pkl

from sodetlib.det_config import DetConfig
from sodetlib.util import get_tracking_kwargs, cprint, make_filename

from ocs.matched_client import MatchedClient
import ocs

from pysmurf.client.util.pub import set_action


def set_dr_temp(lakeshore, temp, timeout=None):
    """
    Servos the DR to a specific temperature and waits until it is stable.

    Args
    ------
        lakeshore : MatchedClient
            lakeshore matched client.
        temp : float
            Temperature to servo to
        timeout : Optional[float]
            Timeout before will stop checking for stability

    Returns
    ---------
        success : bool
            True if check_temperature_stability succeeds, otherwise False.
    """
    lakeshore.servo_to_temperature.start(temperature=temp)
    lakeshore.servo_to_temperature.wait()
    print(f"Servoing to {temp} K. Checking temp stabillity...")

    start_time = time.time()
    while True:
        res = lakeshore.check_temperature_stability(measurements=10,
                                                    threshold=3.e-3)
        if res.status == ocs.OK:
            return True
        else:
            if timeout is not None:
                if time.time() - start_time > timeout:
                    return False
            time.sleep(2)


@set_action()
def IV_vs_temp(S, cfg, lakeshore, bands, temps):
    S.load_tune(cfg.dev.exp['tunefile'])

    # Initilaize smurf stuff
    for band in bands:
        band_cfg = cfg.dev.bands[band]
        S.setup_notches(band, drive=band_cfg['drive'],
                        new_master_assignment=False)
        S.run_serial_gradient_descent(band)
        S.run_serial_eta_scan(band)
        tracking_kwargs = get_tracking_kwargs(cfg, band)
        S.tracking_setup(band, **tracking_kwargs)

    # Initialize DR stuff
    lakeshore.set_pid(P=160, I=30, D=0)
    heater_range = 10e-3
    lakeshore.set_heater_range(range=heater_range)

    iv_files = {}
    for temp in temps:
        set_dr_temp(lakeshore, temp)
        for band in bands:
            S.serial_gradient_descent(band)
            S.run_serial_eta_scan(band)
            iv_file = S.slow_iv_all(...)
            cprint(f"IV info saved: {iv_file}")
            iv_files[(band, temp)] = iv_file

    filename = make_filename(S, 'iv_files_vs_temp.pkl')

    cprint("Run successful! Saving pkl summary to {filename}")
    with open(filename, 'rb') as file:
        pkl.dump(iv_files, file)


if __name__ == '__main__':
    # Create smurf control
    cfg = DetConfig()
    parser = argparse.ArgumentParser()
    parser.add_argument('--lakeshore', '--ls', type=str,
                        help="Instance id of lakeshore agent")
    parser.add_argument('--bands', '-b', type=int, nargs='+', default=None,
                        help='Bands to run ivs')
    parser.add_argument('--temps', type=float, nargs='+', default=None,
                        help='temps to run IV')


    args = cfg.parse_args(parser)
    S = cfg.get_smurf_control(dump_configs=True)

    ls = MatchedClient(args.lakeshore, args=[])

    if args.bands is None:
        args.bands = np.arange(8)
    if args.temps is None:
        args.temps = np.arange(90, 205, 5) / 1000

    try:
        IV_vs_temp(S, cfg, ls, args.bands, args.temps)
    finally:
        cprint("Restoring temperature to 100 mK")
        set_dr_temp(ls, 0.1)
