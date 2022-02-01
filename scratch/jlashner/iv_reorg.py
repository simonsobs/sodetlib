import numpy as np
import time
from scipy.interpolate import interp1d
from sotodlib.core import AxisManager, LabelAxis, IndexAxis
import sodetlib as sdl


class IVAnalysis:
    def __init__(self, S=None, cfg=None, run_kwargs=None, sid=None,
                 start_times=None, stop_times=None):
        self._S = S
        self._cfg = cfg
        self.run_kwargs = run_kwargs
        self.sid = sid
        self.start_times = start_times
        self.stop_times = stop_times

        if self._S is not None:
            # Initialization stuff on initial creation with a smurf instance
            self.meta = get_metadata(S, cfg)
            self.biases_cmd = np.sort(self.run_kwargs['biases'])
            self.nbiases = len(self.biases_cmd)
            self.bias_groups = self.run_kwargs['bias_groups']

            am = self._load_am()
            self.nchans= len(am.signal)

            self.bands = am.ch_info.band
            self.channels = am.ch_info.channel
            self.resp = np.full((self.nchans, self.nbiases), np.nan)
            self.R = np.full((self.nchans, self.nbiases), np.nan)
            self.p_tes = np.full((self.nchans, self.nbiases), np.nan)
            self.v_tes = np.full((self.nchans, self.nbiases), np.nan)
            self.i_tes = np.full((self.nchans, self.nbiases), np.nan)
            self.v_bias = np.full((self.nbiases, ), np.nan)
            self.i_bias = np.full((self.nbiases, ), np.nan)
            self.start_times = np.full((self.nbiases, ), np.nan)
            self.stop_times = np.full((self.nbiases, ), np.nan)
            self.R_n = np.full((self.nchans, ), np.nan)
            self.R_L = np.full((self.nchans, ), np.nan)
            self.p_sat = np.full((self.nchans, ), np.nan)
            self.s_i = np.full((self.nchans, ), np.nan)
            self.idxs = np.full((self.nchans, 3), -1, dtype=int)


            self.bgmap, self.polarity = sdl.load_bgmap(
                self.bands, self.channels, self.meta['bgmap_file']
            )

    def save(self, path=None, update_cfg=False):
        saved_fields = [
            'meta', 'bands', 'channels', 'sid', 'start_times', 'stop_times',
            'run_kwargs', 'biases_cmd', 'bias_groups', 'nbiases', 'nchans',
            'bgmap', 'polarity', 'resp', 'v_bias', 'i_bias', 'R', 'R_n', 'R_L',
            'idxs', 'p_tes', 'v_tes', 'i_tes', 'p_sat', 'si',
        ]
        data = {
            field: getattr(self, field, None) for field in saved_fields
        }
        if path is not None:
            np.save(path, data, allow_pickle=True)
        else:
            self.filepath = validate_and_save(
                self._S, self._cfg, 'iv_analysis', data, make_path=True,
                register=True)

            if update_cfg:
                self._cfg.dev.update_experiment({'iv_file': self.filepath},
                                                update_file=True)

    @classmethod
    def load(cls, path):
        iva = cls()
        for key, val in np.load(path, allow_pickle=True).item():
            setattr(iva, key, val)
        return iva

    def _load_am(self, arc=None):
        if self.am is None:
            if arc:
                self.am = arc.load_data(self.start, self.stop)
            else:
                self.am = sdl.load_session(self.meta['stream_id'], self.sid)
        return self.am

def compute_psats(iva, psat_level):
    # calculates P_sat as P_TES at 90% R_n
    # if the TES is at 90% R_n more than once, just take the first crossing

    for i in range(iva.nchans):
        if np.isnan(iva.R_n[i]):
            continue

        level = psat_level
        R = iva.R[i]
        R_n = iva.R_n[i]
        p_tes = iva.p_tes[i]
        cross_idx = np.where(
            np.logical_and(R/R_n - level >= 0,
                           np.roll(R/R_n - level, 1) < 0)
        )[0]

        if not cross_idx:
            iva.p_sat[i] = np.nan
            continue

        cross_idx = cross_idx[0]
        if cross_idx == 0:
            iva.p_sat[i] = np.nan
            continue

        iva.idxs[i, 2] = cross_idx
        iva.p_sat[i] = interp1d(
            R[cross_idx-1:cross_idx+1]/R_n,
            p_tes[cross_idx-1:cross_idx+1]
        )(level)

def compute_si(iva):
    smooth_dist = 5
    w_len = 2 * smooth_dist + 1
    w = (1./float(w_len))*np.ones(w_len)  # window

    for i in range(iva.nchans):
        if np.isnan(iva.R_n[i]):
            continue

        # Running average
        i_tes_smooth = np.convolve(iva.i_tes[i], w, mode='same')
        v_tes_smooth = np.convolve(iva.v_tes[i], w, mode='same')
        r_tes_smooth = v_tes_smooth/i_tes_smooth

        sc_idx = iva.idxs[i, 0]
        R_L = iva.R_L[i]

        # Take derivatives
        di_tes = np.diff(i_tes_smooth)
        dv_tes = np.diff(v_tes_smooth)
        R_L_smooth = np.ones(len(r_tes_smooth)) * R_L
        R_L_smooth[:sc_idx] = dv_tes[:sc_idx]/di_tes[:sc_idx]
        r_tes_smooth_noStray = r_tes_smooth - R_L_smooth
        i0 = i_tes_smooth[:-1]
        r0 = r_tes_smooth_noStray[:-1]
        rL = R_L_smooth[:-1]
        beta = 0.

        # artificially setting rL to 0 for now,
        # to avoid issues in the SC branch
        # don't expect a large change, given the
        # relative size of rL to the other terms
        rL = 0

        # Responsivity estimate, derivation done here by MSF
        # https://www.overleaf.com/project/613978cb38d9d22e8550d45c
        si = -(1./(i0*r0*(2+beta)))*(1-((r0*(1+beta)+rL)/(dv_tes/di_tes)))
        iva.si[i] = si


def run_analysis(iva, save=False, update_cfg=False, psat_level=0.9,
                 phase_excursion_min=3.0):

    am = iva._load_am()
    R_sh = iva.meta['R_sh']

    # Calculate phase response and bias_voltages / currents
    for i in range(iva.nbiases):
        # Start from back because analysis is easier low->high voltages
        t0, t1 = iva.start_times[-i], iva.stop_times[-i]
        t0 = t0 + (t1 - t0) * 0.5
        m = (t0 < am.timestamps) & (am.timestamps < t1)
        iva.resp[:, i] = np.mean(am.signal[:, m], axis=1)

        # Assume all bias groups have the same bias during IV
        bias_bits = np.median(am.signal.biases[iva.bias_groups[0], m])
        iva.v_bias[i] = bias_bits * 2 * iva.meta['rtm_bit_to_volt']

    R_inline = iva.meta['bias_line_resistance']
    if iva.meta['high_current_mode'][iva.bias_groups[0]]:
        R_inline /= iva.meta['high_low_current_ratio']
    iva.i_bias = iva.v_bias / R_inline

    # Convert phase to uA
    iva.resp = iva.resp * iva.meta['pA_per_phi0'] / (2*np.pi * 1e6) \
                          * iva.polarity

    for i in range(iva.nchans):
        # Skip channels we don't expect are coupled to detectors based on
        # bias-group mapping
        if iva.bgmap[i] == -1:
            continue

        d_resp = np.diff(iva.resp[i])
        dd_resp = np.diff(d_resp)
        dd_resp_abs = np.abs(dd_resp)

        # Find index of superconducting branch
        sc_idx = np.argmax(dd_resp_abs) + 1
        iva.idxs[i, 0] = sc_idx

        # Find index of normal branch by finding the min index after
        # sc branch
        nb_idx_default = int(0.8*iva.nbiases)
        nb_idx = nb_idx_default
        for i in np.arange(nb_idx_default, sc_idx, -1):
            # look for minimum of IV curve outside of superconducting region
            # but get the sign right by looking at the sc branch
            if d_resp[i]*np.mean(d_resp[:sc_idx]) < 0.:
                nb_idx = i+1
                break

        iva.idxs[i, 1] = nb_idx
        nb_fit_idx = (iva.nbiases + nb_idx) // 2
        norm_fit = np.polyfit(iva.i_bias[nb_fit_idx:],
                              iva.resp[i, nb_fit_idx:], 1)
        iva.resp[i] -= norm_fit[1]  # Put resp in real current units

        sc_fit = np.polyfit(iva.i_bias[:sc_idx], iva.resp[i, :sc_idx], 1)
        # subtract off unphysical y-offset in superconducting branch; this
        # is probably due to an undetected phase wrap at the kink between
        # the superconducting branch and the transition, so it is
        # *probably* legitimate to remove it by hand. We don't use the
        # offset of the superconducting branch for anything meaningful
        # anyway. This will just make our plots look nicer.
        iva.resp[i, :sc_idx] -= sc_fit[1]
        sc_fit[1] = 0  # now change s.c. fit offset to 0 for plotting

        R = R_sh * (iva.i_bias/(iva.resp[i]) - 1)
        R_n = np.mean(R[nb_fit_idx:])
        R_L = np.mean(R[1:sc_idx])

        if R_n < 0:
            continue

        iva.v_tes[i] = iva.i_bias * R_sh * R / (R + R_sh)
        iva.i_tes[i] = iva.v_tes[i] / R
        iva.p_tes[i] = (iva.v_tes[i]**2) / R
        iva.R[i] = R
        iva.R_n[i] = R_n
        iva.R_L[i] = R_L

    compute_psats(iva, psat_level)
    compute_si(iva)

    if save:
        iva.save(update_cfg=update_cfg)



def take_iv(S, cfg, bias_groups=None, overbias_voltage=18.0, overbias_wait=2.0,
            high_current_mode=True, cool_wait=30, cool_voltage=None,
            biases=None, bias_high=16, bias_low=0, bias_step=0.025,
            wait_time=0.001, run_analysis=True, analysis_kwargs=None):
    if not hasattr(S, 'tune_file'):
        raise AttributeError('No tunefile loaded in current '
                             'pysmurf session. Load active tunefile.')

    if bias_groups is None:
        bias_groups = np.arange(12)
    bias_groups = np.atleast_1d(bias_groups)

    if biases is None:
        biases = np.arange(bias_high, bias_low - bias_step, bias_step)
    # Make sure biases is in decreasing order for run function
    biases = np.sort(biases)[::-1]

    run_kwargs = {
        'bias_groups': bias_groups, 'overbias_voltage': overbias_voltage,
        'overbias_wait': overbias_wait, 'high_current_mode': high_current_mode,
        'cool_wait': cool_wait, 'cool_voltage': cool_voltage, 'biases': biases,
        'bias_high': bias_high, 'bias_low': bias_low, 'bias_step': bias_step,
        'wait_time': wait_time, 'run_analysis': run_analysis,
        'analysis_kwargs': analysis_kwargs
    }

    if high_current_mode:
        biases /= S.high_low_current_ratio
        if cool_voltage is not None:
            cool_voltage /= S.high_low_current_ratio


    if overbias_voltage > 0:
        if cool_voltage is None:
            cool_voltage = np.max(biases)
        S.overbias_tes_all(
            bias_groups=bias_groups, overbias_wait=overbias_wait,
            tes_bias=cool_voltage, cool_wait=cool_wait,
            high_current_mode=high_current_mode,
            overbias_voltage=overbias_voltage
        )

    S.log("Starting TES Bias Ramp", S.LOG_USER)
    bias_group_bool = np.zeros((S._n_bias_groups))
    bias_group_bool[bias_groups] = 1
    sid = sdl.stream_g3_on(S)
    start_times = np.zeros_like(biases)
    stop_times = np.zeros_like(biases)
    for i, bias in enumerate(biases):
        S.log("Setting bias to {bias:4.3f}")
        S.set_tes_bias_bipolar_array(bias * bias_group_bool)
        start_times[i] = time.time()
        time.sleep(wait_time)
        stop_times[i] = time.time()
    sdl.stream_g3_off(S)

    iva = IVAnalysis(S, cfg, run_kwargs, sid, start_times, stop_times)

    if run_analysis:
        run_analysis(iva save=True, **analysis_kwargs)

    return iva




# These funcs will all be in sodetlib.util after normalize_datafiles is merged
def map_band_chans(b1, c1, b2, c2, chans_per_band=512):
    """
    Returns an index mapping of length nchans1 from one set of bands and
    channels to another. Note that unmapped indices are returned as -1, so this
    must be handled before (or after) indexing or else you'll get weird
    results.
    Args
    ----
        b1 (np.ndarray):
            Array of length nchans1 containing the smurf band of each channel
        c1 (np.ndarray):
            Array of length nchans1 containing the smurf channel of each channel
        b2 (np.ndarray):
            Array of length nchans2 containing the smurf band of each channel
        c2 (np.ndarray):
            Array of length nchans2 containing the smurf channel of each channel
        chans_per_band (int):
            Lets just hope this never changes.
    """
    acs1 = b1 * chans_per_band + c1
    acs2 = b2 * chans_per_band + c2

    mapping = np.full_like(acs1, -1, dtype=int)
    for i, ac in enumerate(acs1):
        x = np.where(acs2 == ac)[0]
        if len(x > 0):
            mapping[i] = x[0]

    return mapping


def get_metadata(S, cfg):
    """
    Gets collection of metadata from smurf and detconfig instance that's
    useful for pretty much everything. This should be included in all output
    data files created by sodetlib.
    """
    return {
        'tunefile': S.tune_file,
        'high_low_current_ratio': S.high_low_current_ratio,
        'R_sh': S.R_sh,
        'pA_per_phi0': S.pA_per_phi0,
        'rtm_bit_to_volt': S.rtm_bit_to_volt,
        'bias_line_resistance': S.bias_line_resistance,
        'chans_per_band': S.get_number_channels(),
        'high_current_mode': get_current_mode_array(S),
        'timestamp': time.time(),
        'stream_id': cfg.stream_id,
        'action': S.pub._action,
        'action_timestamp': S.pub._action_ts,
        'bgmap_file': cfg.dev.exp.get('bgmap_file'),
        'iv_file': cfg.dev.exp.get('iv_file')
    }


def validate_and_save(fname, data, S=None, cfg=None, register=True,
                      make_path=True):
    """
    Does some basic data validation for sodetlib data files to make sure
    they are normalized based on the following  rules:
        1. The 'meta' key exists storing metadata from smurf and cfg instances
           such as the stream-id, pysmurf constants like high-low-current-ratio
           etc., action and timestamp and more. See the get_metadata function
           definition for more.
        2. 'bands', and 'channels' arrays are defined, specifying the band/
           channel combinations for data taking / analysis products.
        3. 'sid' field exists to contain one or more session-ids for the
           analysis.
    Args
    ----
        fname (str):
            Filename or full-path of datafile. If ``make_path`` is set to True,
            this will be turned into a file-path using the ``make_filename``
            function. If not, this will be treated as the save-file-path
        data (dict):
            Data to be written to disk. This should contain at least the keys
            ``bands``, ``channels``, ``sid``, along with any other data
            products. If the key ``meta`` does not already exist, metadata
            will be populated using the smurf and det-config instances.
        S (SmurfControl):
            Pysmurf instance. This must be set if registering data file,
            ``meta`` is not yet set, or ``make_path`` is True.
        cfg (DetConfig):
            det-config instance. This must be set if registering data file,
            ``meta`` is not yet set, or ``make_path`` is True.
        register (bool):
            If True, will register the file with the pysmurf publisher
        make_path (bool):
            If true, will create a path by passing fname to the function
            make_filename (putting it in the pysmurf output directory)
    """
    if register or make_path or 'meta' not in data:
        if S is None or cfg is None:
            raise ValueError("Pysmurf and cfg instances must be set")

    _data = data.copy()
    if 'meta' not in data:
        meta = get_metadata(S, cfg)
        _data['meta'] = meta

    for k in ['channels', 'bands', 'sid']:
        if k not in data:
            raise ValueError(f"Key '{k}' is required in data")

    if make_path and S is not None:
        path = make_filename(S, fname)
    else:
        path = fname

    np.save(path, data, allow_pickle=True)
    if S is not None and register:
        S.pub.register_file(path, 'sodetlib_data', format='npy')
    return path






