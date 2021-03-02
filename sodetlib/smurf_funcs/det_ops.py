import numpy as np
import time
import os
from sodetlib.util import make_filename
import yaml

from pysmurf.client.util.pub import set_action

@set_action()
def take_iv(S, bias_groups=None, wait_time=.1, bias=None,
            bias_high=1.5, bias_low=0, bias_step=.005,
            show_plot=False, overbias_wait=2., cool_wait=30,
            make_plot=True, save_plot=True, plotname_append='',
            channels=None, band=None, high_current_mode=True,
            overbias_voltage=8., grid_on=True,
            phase_excursion_min=3., bias_line_resistance=None,
            do_analysis=True,cool_voltage = None):
    """
    Sodetlib IV script. Replaces the pysmurf run_iv function
    to be more appropriate for SO-specific usage, as well as 
    easier to edit as needed. 
    Steps the TES bias down slowly. Starts at bias_high to
    bias_low with step size bias_step. Waits wait_time between
    changing steps. After overbiasing, can choose bias point to
    allow system to cool down.
    Args
    ----
    bias_groups : numpy.ndarray or None, optional, default None
        Which bias groups to take the IV on. If None, defaults to
        the groups in the config file.
    wait_time : float, optional, default 0.1
        The amount of time between changing TES biases in seconds.
    bias : float array or None, optional, default None
        A float array of bias values. Must go high to low.
    bias_high : float, optional, default 1.5
        The maximum TES bias in volts.
    bias_low : float, optional, default 0
        The minimum TES bias in volts.
    bias_step : float, optional, default 0.005
        The step size in volts.
    overbias_wait : float, optional, default 2.0
        The time to stay in the overbiased state in seconds.
    cool_wait : float, optional, default 30.0
        The time to stay in the low current state after
        overbiasing before taking the IV.
    high_current_mode : bool, optional, default True
        The current mode to take the IV in.
    overbias_voltage : float, optional, default 8.0
        The voltage to set the TES bias in the overbias stage.
    grid_on : bool, optional, default True
        Grids on plotting. 
    phase_excursion_min : float, optional, default 3.0
        The minimum phase excursion required for making plots.
    bias_line_resistance : float or None, optional, default None
        The resistance of the bias lines in Ohms. If None, loads value
        in config file
    do_analysis: bool, optional, default True
        Whether to do the pysmurf IV analysis
    cool_voltage: float, optional, default None
        The voltage to bias at after overbiasing before taking the IV
        while the system cools.
    Returns
    -------
    output_path : str
        Full path to IV raw npy file.
    """        
    
    n_bias_groups = 12 # SO UFMs have 12 bias groups
    if bias_groups is None:
        bias_groups = np.arange(12) # SO UFMs have 12 bias groups
    bias_groups = np.array(bias_groups)
    if overbias_voltage != 0.:
        overbias = True
    else:
        overbias = False
    if bias is None:
        # Set actual bias levels
        bias = np.arange(bias_high, bias_low-bias_step, -bias_step)
    # Overbias the TESs to drive them normal
    if overbias:
        if cool_voltage is None:
            S.overbias_tes_all(bias_groups=bias_groups,
                overbias_wait=overbias_wait, tes_bias=np.max(bias),
                cool_wait=cool_wait, high_current_mode=high_current_mode,
                overbias_voltage=overbias_voltage)
        else:
            S.overbias_tes_all(bias_groups=bias_groups,
                overbias_wait=overbias_wait, tes_bias=cool_voltage,
                cool_wait=cool_wait, high_current_mode=high_current_mode,
                overbias_voltage=overbias_voltage)
    S.log('Starting to take IV.', S.LOG_USER)
    S.log('Starting TES bias ramp.', S.LOG_USER)
    bias_group_bool = np.zeros((n_bias_groups,))
    bias_group_bool[bias_groups] = 1 # only set things on the bias groups that are on
    S.set_tes_bias_bipolar_array(bias[0] * bias_group_bool)
    time.sleep(wait_time)
    start_time = S.get_timestamp() # get time the IV starts
    S.log(f'Starting IV at {start_time}')
    datafile = S.stream_data_on()
    S.log(f'writing to {datafile}')
    for b in bias:
        S.log(f'Bias at {b:4.3f}')
        S.set_tes_bias_bipolar_array(b * bias_group_bool)
        time.sleep(wait_time)
    S.stream_data_off()
    stop_time = S.get_timestamp() # get time the IV finishes
    S.log(f'Finishing IV at {stop_time}')
    basename, _ = os.path.splitext(os.path.basename(datafile))
    path = os.path.join(S.output_dir, basename + '_iv_bias_all')
    np.save(path, bias)
    # publisher announcement
    S.pub.register_file(path, 'iv_bias', format='npy')
    
    iv_info = {}
    iv_info['plot_dir'] = S.plot_dir
    iv_info['output_dir'] = S.output_dir
    iv_info['R_sh'] = S.R_sh
    iv_info['bias_line_resistance'] = S.bias_line_resistance
    iv_info['high_low_ratio'] = S.high_low_current_ratio
    iv_info['pA_per_phi0'] = S.pA_per_phi0
    iv_info['high_current_mode'] = high_current_mode
    iv_info['start_time'] = start_time
    iv_info['stop_time'] = stop_time
    iv_info['basename'] = basename
    iv_info['datafile'] = datafile
    iv_info['bias_array'] = bias
    iv_info['bias group'] = bias_groups
    iv_info['bias'] = bias
    
    iv_info_fp = os.path.join(S.output_dir, basename + '_iv_info.npy')
 
    np.save(iv_info_fp, iv_info)
    S.log(f'Writing IV information to {iv_info_fp}.')
    S.pub.register_file(iv_info_fp, 'iv_info', format='npy')
    
    return iv_info_fp