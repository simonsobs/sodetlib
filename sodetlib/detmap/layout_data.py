import matplotlib.pyplot as plt
from sodetlib.detmap.simple_csv import read_csv


a_pols = {'T', 'R'}
b_pols = {'B', 'L'}
d_pol = 'X'


def bond_pad_to_wafer_row_parse(single_row, dark_bias_lines=None):
    # regular expression format of SQUID_PIN 'SQ_(.+)_Ch_(.+)_\+'
    _SQ, _squid_num, _CH, bond_pad, _plus = single_row['squid_pin'].split('_')
    mux_layout_position = single_row['mux chip position']

    pol_str = single_row['dtpadlabel']
    pol_letter = pol_str[0].upper()
    if pol_letter in a_pols:
        pol = 'A'
    elif pol_letter in b_pols:
        pol = 'B'
    elif pol_letter == d_pol:
        pol = 'D'
    else:
        raise KeyError(f'polarization character: {pol_letter} is not one of the expected types.')

    rhomb = single_row['dtpixelsection']

    bias_line = single_row['bias line']

    det_row = single_row['dtpixelrow']
    det_col = single_row['dtpixelcolumn']

    det_x = single_row['x'] / 1.0e3
    det_y = single_row['y'] / 1.0e3

    single_description = single_row['dtsignaldescription']

    if any([dark_bias_lines is not None and bias_line in dark_bias_lines, pol == 'D', single_description == 'NC']):
        is_optical = False
    else:
        is_optical = True

    bandpass_str = single_row['dtsignaldescription']

    if bandpass_str == 'NC':
        if pol == 'D':
            bandpass = 90
        else:
            bandpass = 'NC'
    elif 'ghz' == bandpass_str[-3:]:
        bandpass = int(bandpass_str[:-3])
    else:
        raise ValueError(f'Frequency string: {bandpass_str}, cannot be parsed.')

    layout_dict_this_row = {"mux_layout_position": mux_layout_position, "bond_pad": int(bond_pad),
                            "bias_line": bias_line, "pol": pol, "bandpass": bandpass, "det_row": det_row,
                            "det_col": det_col, "rhomb": rhomb, "is_optical": is_optical,
                            "det_x": det_x, "det_y": det_y}
    return layout_dict_this_row


def get_layout_data(filename, dark_bias_lines=None, plot=False):
    """
    Extracts routing wafer to detector wafer map
    Based on code originally writen by Zach Atkin.

    Upgraded for speed and converted to PEP-8 format by Caleb Wheeler Dec 2021
    ----------
    filename:
        Path to the detector-routing wafer map created by NIST and Princeton
    dark_bias_lines:
        Bias lines that are dark in a particular test

    Returns:
    -------
    wafer_info
        A two level dictionary with a primary key of mux_layout_position and a
        secondary key of bond_pad to access a dictionary of detector
        information. In particular, bandpass column indicates 90ghz, 150ghz,
        D for dark detectors which is 90ghz but has different property as optical
        ones, and NC for no-coupled resonators.
    """
    if dark_bias_lines is not None:
        dark_bias_lines = set(dark_bias_lines)

    _data_by_column, data_by_row = read_csv(path=filename)
    parsed_wafer_info = [bond_pad_to_wafer_row_parse(single_row=single_row, dark_bias_lines=dark_bias_lines)
                         for single_row in data_by_row]
    wafer_info = {}
    for wafer_datum in parsed_wafer_info:
        mux_layout_position = wafer_datum['mux_layout_position']
        bond_pad = wafer_datum['bond_pad']
        if mux_layout_position not in wafer_info.keys():
            wafer_info[mux_layout_position] = {}
        if bond_pad in wafer_info[mux_layout_position].keys():
            if wafer_datum == wafer_info[mux_layout_position][bond_pad]:
                pass
            else:
                raise KeyError(f'mux_layout_position and bond_pad must be a unique pair for each row of layout data, ' +
                               f'(mux_layout_position, bond_pad) = ({mux_layout_position}, {bond_pad}) is associated ' +
                               f'with at least two rows of data in the file: {filename}')
        else:
            wafer_info[mux_layout_position][bond_pad] = wafer_datum
    if plot:
        alpha = 1.0
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 10))
        layer_one_coord = set()
        layer_two_coord = set()
        layer_three_coord = set()
        layer_four_coord = set()
        for mux_layout_position in sorted(wafer_info.keys()):
            for bond_pad in sorted(wafer_info[mux_layout_position].keys()):
                layout_datum = wafer_info[mux_layout_position][bond_pad]
                det_x = layout_datum['det_x']
                det_y = layout_datum['det_y']
                det_coord = (det_x, det_y)
                if det_coord in layer_one_coord:
                    if det_coord in layer_two_coord:
                        if det_coord in layer_three_coord:
                            if det_coord in layer_four_coord:
                                raise ValueError(f'Too many layers')
                            else:
                                marker = 's'
                                layer_four_coord.add(det_coord)
                                ax = ax4
                        else:
                            marker = '^'
                            layer_three_coord.add(det_coord)
                            ax = ax3
                    else:
                        marker = 'x'
                        layer_two_coord.add(det_coord)
                        ax = ax2
                else:
                    marker = 'o'
                    layer_one_coord.add(det_coord)
                    ax = ax1

                bandpass = layout_datum['bandpass']
                if bandpass == 90:
                    color = 'firebrick'
                elif bandpass == 150:
                    color = 'dodgerblue'
                elif bandpass == 'NC':
                    color = 'black'
                else:
                    raise KeyError(f'bandpass value: {bandpass} is not a recognized value.')
                ax.plot(det_x, det_y, c=color, marker=marker, ls='None', alpha=alpha)
        marker = 'o'
        fig.legend([plt.Line2D(range(12), range(12), color='firebrick', ls='None',
                               marker=marker, markerfacecolor='firebrick', alpha=alpha),
                    plt.Line2D(range(12), range(12), color='dodgerblue', ls='None',
                               marker=marker, markerfacecolor='dodgerblue', alpha=alpha),
                    plt.Line2D(range(12), range(12), color='black', ls='None',
                               marker=marker, markerfacecolor='black', alpha=alpha)], ['90 GHz', '150 GHz', 'NC'])
        plt.show()
    return wafer_info
