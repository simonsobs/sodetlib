import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


def pbias(Tbath, n, Tc, kappa):
    return kappa * (Tc ** n - Tbath ** n)


def get_g(n, Tc, kappa):
    return n * kappa * Tc ** (n - 1)


def get_vbias(data_dict, band, chan, level=0.5, greedy=False):
    '''Returns the conventional P_sat from a SMuRF IV curve dictionary.
    Parameters
    ----------
    data_dict : dict
        The dictionary containing the IV curve data
    band : int
        The smurf band to extract from the dict
    chan : int
        The channel to extract from the band
    level : float
        The definition of P_sat, i.e. power when R = level*R_n
    greedy : bool, optional
        If True, will return -1000 if the R/Rn curve crosses the level more than once, by default False.
        If False, returns the power at the first instance when R/Rn crosses the level.
    Returns
    -------
    float
        The conventional P_sat
    '''
    chan_data = data_dict[band][chan]
    p = chan_data['p_tes']
    rn = chan_data['R'] / chan_data['R_n']
    cross_idx = np.where(np.logical_and(rn - level >= 0, np.roll(rn - level, 1) < 0))[0]
    try:
        assert len(cross_idx) == 1
    except AssertionError:
        if greedy:
            return -1000
        else:
            cross_idx = cross_idx[:1]
    cross_idx = cross_idx[0]
    rn2p = interp1d(rn[cross_idx - 1:cross_idx + 1], p[cross_idx - 1:cross_idx + 1])
    return rn2p(level)


def get_psat(data_dict, band, chan, unit=1e-12, level=0.9, greedy=False):
    """Returns the conventional P_sat from a SMuRF IV curve dictionary.

    Parameters
    ----------
    data_dict : dict
        The dictionary containing the IV curve data
    band : int
        The smurf band to extract from the dict
    chan : int
        The channel to extract from the band
    unit: float
        The conversion to SI units, by default 1e-12
    level : float
        The definition of P_sat, i.e. power when R = level*R_n
    greedy : bool, optional
        If True, will return -1000 if the R/Rn curve crosses the level more than once, by default False.
        If False, returns the power at the first instance when R/Rn crosses the level.

    Returns
    -------
    float
        The conventional P_sat
    """
    chan_data = data_dict[band][chan]

    p = chan_data['p_tes']
    rn = chan_data['R'] / chan_data['R_n']

    cross_idx = np.where(np.logical_and(rn - level >= 0, np.roll(rn - level, 1) < 0))[0]
    try:
        assert len(cross_idx) == 1
    except AssertionError:
        if greedy:
            return -1000
        else:
            cross_idx = cross_idx[:1]

    try:
        cross_idx = cross_idx[0]
    except IndexError:
        # print(f'band {band}, chan {chan} has IndexError, returning 0')
        # print(cross_idx)
        return 0

    try:
        rn2p = interp1d(rn[cross_idx - 1:cross_idx + 1], p[cross_idx - 1:cross_idx + 1])
    except ValueError:
        # print(f'band {band}, chan {chan} has ValueError, returning 0')
        return 0

    return unit * rn2p(level)


def read_psat(coldload_ivs, map_data=None, make_plot=False):
    psat_dict = {}
    for coldload_iv in coldload_ivs:
        ivfile = coldload_iv['data_path']
        iv = np.load(ivfile, allow_pickle=True).item()

        if coldload_iv['band'] != 'all':
            print('Reading band ', int(coldload_iv['band']), ' in ', ivfile)
            band_list = [int(coldload_iv['band'])]
        else:
            band_list = [b for b in iv.keys() if type(b) == np.int64]

        for band in band_list:
            for chan in iv[band].keys():
                dict_key = (int(band), int(chan))
                # test to initialize this part of the psat dict, if needed
                if dict_key not in psat_dict.keys():
                    psat_dict[dict_key] = {"T": [], 'psat': []}
                try:
                    # if coldload_iv['bias_line'] != 'all':
                    #     map_BL = int(map_data.loc[(map_data["smurf_band"] == band) & (map_data["smurf_chan"] == chan)][
                    #                      "biasline"])
                    #     assert map_BL == coldload_iv["bias_line"]
                    ch_psat = np.float(get_psat(iv, band, chan, level=0.9, greedy=False))
                    psat_dict[dict_key]['T'].append(coldload_iv['bath_temp'])
                    psat_dict[dict_key]['psat'].append(ch_psat)
                except TypeError:
                    print(f'TypeError in read_psat() in the file read_iv.py')
                finally:
                    print('Successful band-channel lap in read_psat() in the file read_iv.py')

    if make_plot:
        for key in psat_dict.keys():
            plt.plot(psat_dict[key]['T'], psat_dict[key]['psat'])
        plt.xlabel('Temp(K)')
        plt.ylabel('Psat(W)')
        plt.show()

    return psat_dict


def filter_good_chans(psat_data, psat_dict, badchan=None, max_psat=20e-12, min_psat=0.5e-12, min_sat_T=14,
                      make_plot=False):
    if badchan is None:
        badchan = []
    filtered_dict = {}
    goodchan = []

    for key in psat_dict.keys():
        band, chan = key
        all_T = psat_dict[key]['T']
        all_psat = psat_data[key]['psat']

        try:
            assert [band, chan] not in badchan
            assert len(all_psat) >= 3
            for i, psat in enumerate(all_psat):
                assert psat < max_psat
                if all_T[i] < min_sat_T:
                    assert psat > min_psat
                if i > 0:
                    assert all_psat[i - 1] > all_psat[i]
            filtered_dict[key] = psat_dict[key]
            goodchan.append([band, chan])
        except:
            pass

    if make_plot == True:
        for key in filtered_dict.keys():
            plt.plot(filtered_dict[key]['T'], filtered_dict[key]['psat'])
        plt.xlabel('Temp(K)')
        plt.ylabel('Psat(W)')

    return filtered_dict, goodchan


def match_chan_map(ass_map, psat_dict):
    df_map = pd.read_csv(ass_map)
    pixel_info = {}

    for key in psat_dict.keys():
        band, chan = key
        try:
            df = df_map.loc[(df_map['smurf_band'] == band) & (df_map['smurf_chan'] == chan)]
            assert len(df) == 1, "IV channel cannot be mapped!"
            pixel_info[key] = {}
            pixel_info[key]['det'] = df[
                ['biasline', 'det_row', 'det_col', 'pol', 'freq', 'rhomb', 'opt', 'det_x', 'det_y']].to_dict('records')
            pixel_info[key]['mux'] = df[['mux_band', 'pad', 'mux_posn']].to_dict('records')
            pixel_info[key]['T'] = psat_dict[key]['T']
            pixel_info[key]['psat'] = psat_dict[key]['psat']
            if psat_dict[key]['BL'] != 'all':
                assert int(psat_dict[key]['BL']) == int(df['biasline']), 'Bias line mismatch!'
        except:
            pass
    return pixel_info


def plot_wafer_psat(pixel_info, T, pol=None, freq=None, cbrange=None, **kwargs):
    if pol is None:
        pol = ['A', 'B', 'D']
    if freq is None:
        freq = [90, 150]
    if cbrange is None:
        cbrange = []

    x = []
    y = []
    psat = []
    for key in pixel_info.keys():
        t_list = np.array(pixel_info[key]['T'])
        if T in t_list:
            if pixel_info[key]['det']['pol'] in pol:
                if pixel_info[key]['det']['freq'] in [str(f) for f in freq]:
                    x.append(float(pixel_info[key]['det']['det_x']))
                    y.append(float(pixel_info[key]['det']['det_y']))
                    psat.append(pixel_info[key]['psat'][np.where(t_list == T)[0][0]])

    title = str("Saturation Power at %d K \n Pol=%s freq=%s" % (T, pol, freq))
    wafer_scatter(x, y, psat, 'Psat', title, cbrange, **kwargs)


def wafer_scatter(x, y, vals, label, title, cbrange=None, cmap=plt.cm.inferno, **kwargs):
    if cbrange is None:
        cbrange = []

    if not cbrange:
        vmin = 0
        vmax = np.percentile(vals, 95)
    else:
        assert len(cbrange) == 2
        vmin = cbrange[0]
        vmax = cbrange[1]

    sc = plt.scatter(x, y, c=vals, vmin=vmin, vmax=vmax, cmap=cmap, **kwargs)

    bar = plt.colorbar(sc)
    bar.set_label(label)

    plt.xlabel('x [mm]')
    plt.ylabel('y [mm]')
    plt.title(title)
    plt.show()


def plot_psat_vs_T(pixel_info, freqlist=None, optlist=None):
    if freqlist is None:
        freqlist = [90, 150]
    if optlist is None:
        optlist = [0, 1]
    plt.figure(figsize=(6, 4), dpi=300)
    color_dict = {
        (90, 1): 'orange', (90, 0): 'red',
        (150, 1): 'purple', (150, 0): 'blue'}
    seen_dict = {}
    nchan = 0
    opt_dict = {0: 'Dark', 1: 'Optical'}
    for key in pixel_info.keys():
        freq = int(pixel_info[key]['det']['freq'])
        opt = int(pixel_info[key]['det']['opt'])
        if (freq in freqlist) & (opt in optlist):
            if len(pixel_info[key]['T']) > 0:
                nchan = nchan + 1
                if (freq, opt) not in seen_dict:
                    label = f'{freq}GHz, {opt_dict[opt]}'
                    try:
                        plt.plot(pixel_info[key]['T'], 1e12 * np.array(pixel_info[key]['psat']), label=label,
                                 color=color_dict[freq, opt])
                    except:
                        print(key)
                    seen_dict[freq, opt] = True

                else:
                    plt.plot(pixel_info[key]['T'], 1e12 * np.array(pixel_info[key]['psat']),
                             color=color_dict[freq, opt])

    plt.xlabel('Cold Load Temp [K]')
    plt.ylabel('$P_{sat}$ [pW]')
    plt.legend(title=f'N = {nchan}')
    plt.grid(which='both')
    plt.title(f'$P_{{sat}}$ Curves')
    plt.show()


