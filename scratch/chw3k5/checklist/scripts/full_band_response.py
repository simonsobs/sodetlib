"""
The core code is from Princeton, created by Yunan Wang and Daniel Dutcher Oct/Nov 2021.

The code was refactored by Caleb Wheeler Nov 2021 to support augparse
The Argparse (found at the bottom of the file) and provides a minimal
documentation framework.

Use full_band_response.py -h to see the available options.
"""

import time
import numpy as np
import os
import matplotlib.pylab as plt


def full_band_response(S, bands=None, n_scan_per_band=5, wait_btw_bands_sec=5, verbose=True):
    plt.ion()
    prefix_str = f' From {full_band_response.__name__} '

    timestamp = S.get_timestamp()
    if bands is None:
        bands = S.config.get('init').get('bands')

    resp_dict = {}
    for band in sorted(bands):
        if verbose:
            print(f'{prefix_str}\n\nBand {band}\n\n')
        resp_dict[band] = {}
        resp_dict[band]['fc'] = S.get_band_center_mhz(band)

        f, resp = S.full_band_resp(band=band, make_plot=False, show_plot=False, n_scan=n_scan_per_band,
                                   timestamp=timestamp, save_data=True)
        resp_dict[band]['f'] = f
        resp_dict[band]['resp'] = resp

        time.sleep(wait_btw_bands_sec)

    fig, ax = plt.subplots(2, figsize=(6, 7.5), sharex='True')

    # plt.suptitle(f'slot={S.slot_number} AMC0={S.get_amc_asset_tag(0)} AMC2={S.get_amc_asset_tag(1)}')

    ax[0].set_title(f'Full band response {timestamp}')
    last_angle = None
    for band in bands:
        f_plot = resp_dict[band]['f'] / 1e6
        resp_plot = resp_dict[band]['resp']
        plot_idx = np.where(np.logical_and(f_plot > -250, f_plot < 250))
        ax[0].plot(f_plot[plot_idx] + resp_dict[band]['fc'], np.log10(np.abs(resp_plot[plot_idx])), label=f'b{band}')
        angle = np.unwrap(np.angle(resp_plot))
        if last_angle is not None:
            angle -= (angle[0] - last_angle)
        ax[1].plot(f_plot[plot_idx] + resp_dict[band]['fc'], angle[plot_idx], label=f'b{band}')
        last_angle = angle[plot_idx][-1]

    if verbose:
        print(f'{prefix_str}data taking done')

    ax[0].legend(loc='lower left', fontsize=8)
    ax[0].set_ylabel("log10(abs(Response))")
    ax[0].set_xlabel('Frequency [MHz]')

    ax[1].legend(loc='lower left', fontsize=8)
    ax[1].set_ylabel("phase [rad]")
    ax[1].set_xlabel('Frequency [MHz]')

    save_name = f'{timestamp}_full_band_resp_all.png'
    plt.title(save_name)

    plt.tight_layout()

    if verbose:
        print(f'{prefix_str}Saving plot to {os.path.join(S.plot_dir, save_name)}')
    plt.savefig(os.path.join(S.plot_dir, save_name),
                bbox_inches='tight')
    plt.show()

    save_name = f'{timestamp}_full_band_resp_all.npy'
    if verbose:
        print(f'{prefix_str}Saving data to {os.path.join(S.output_dir, save_name)}')
    full_resp_data = os.path.join(S.output_dir, save_name)
    path = os.path.join(S.output_dir, full_resp_data)
    np.save(path, resp_dict)

    # log plot file
    logf = open('/data/smurf_data/smurf_loop.log', 'a+')
    logf.write(f'{os.path.join(S.plot_dir, save_name)}' + '\n')
    logf.close()

    if verbose:
        print(f'{prefix_str}Done running full_band_response.py.')
    return


if __name__ == '__main__':
    import argparse
    from sodetlib.det_config import DetConfig
    from scratch.chw3k5.ufm_optimize.operators.controler import LoadS
    """
    The code below will only run if the file is run directly, but not if elements from this file are imported.
    For example:
        python3 time_steams.py -args_for_argparse
    will have __name__ == '__main__' as True, and the code below will run locally.
    """

    # Seed a new parse for this file with the parser from the SMuRF controller class
    cfg = DetConfig()

    # set up the parser for this Script
    parser = argparse.ArgumentParser(description='Parser for time_stream.py script.')
    parser.add_argument('bands', type=int, metavar='bands', nargs='+', action='append',
                        help='The SMuRF bands (ints) to optimize. This is expected to be a sequence of N integers.')

    # optional arguments
    parser.add_argument('--n-scan-per-band', dest='n_scan_per_band', type=int, default=1,
                        help="int, optional, default is 1.  See n_scan argument in PySmuRF Docs for "
                             "full_band_resp() -> (int, optional, default 1) – The number of scans to take " +
                             "and average.")
    parser.add_argument('--wait-bwt-bands-sec', dest='wait_btw_bands_sec', type=float, default=5,
                        help="float, optional, default is 5. While looping over bands, wait this amount of time in " +
                             "seconds between bands.")
    parser.add_argument('--verbose', dest='verbose', action='store_true',
                        help="Turns on printed output from the script. The default is --verbose." +
                             "--no-verbose has minimal (or no) print statements.")
    parser.add_argument('--no-verbose', dest='verbose', action='store_false', default=True,
                        help="Turns off printed output from the script. The default is --verbose." +
                             "--no-verbose has minimal (or no) print statements.")

    # parse the args for this script
    args = cfg.parse_args(parser)
    S = cfg.get_smurf_control(dump_configs=True)

    # run the def in this file
    full_band_response(S=S, bands=args.bands, n_scan_per_band=args.n_scan_per_band,
                       wait_btw_bands_sec=args.wait_btw_bands_sec, verbose=True)


