import numpy as np
import tqdm
import os
import time
import numpy as np
import sodetlib as sdl

from pysmurf.client.base.smurf_control import SmurfControl


class CISweep:
    def __init__(self, *args, **kwargs):
        if len(args) > 0:
            self.initialize(*args, **kwargs)
    
    def initialize(self, S, cfg, run_kwargs, sid, start_times, stop_times,
                   bands, channels, state):
        self._S = S
        self._cfg = cfg
        self.meta = sdl.get_metadata(S, cfg)
        self.run_kwargs = run_kwargs
        self.freqs = run_kwargs['freqs']
        self.bgs = run_kwargs['bgs']
        self.tickle_voltage = run_kwargs['tickle_voltage']
        self.start_times = start_times
        self.stop_times = stop_times
        self.sid = sid
        self.bias_array = S.get_tes_bias_bipolar_array()
        self.bands = bands
        self.channels = channels
        self.transition = False
        self.ob_path = cfg.dev.exp.get('complex_impedance_ob_path')
        self.sc_path = cfg.dev.exp.get('complex_impedance_sc_path')
        self.state = state

        # Result arrays go here


    def save(self, path=None):
        saved_fields = [
            'meta', 'run_kwargs', 'freqs', 'bgs', 'start_times', 'stop_times',
            'sid', 'bias_array', 'bands', 'channels', 'ob_path',
            'sc_path', 'state'
        ]
        data = {k: getattr(self, k) for k in saved_fields}
        if path is not None:
            np.save(path, data, allow_pickle=True)
            self.filepath = path
            return path
        else:
            filepath = sdl.validate_and_save(
                'ci_sweep.npy', data, S=self._S, cfg=self._cfg.path, register=True)
            self.filepath = filepath
            return filepath

    @classmethod
    def load(cls, path):
        """
        Loads a CISweep object from file
        """
        self = cls()
        for k, v in np.load(path, allow_pickle=True).item():
            setattr(self, k, v)
        return self


@sdl.set_action()
def take_complex_impedance(
        S, cfg, bgs, freqs=None, state='transition', nperiods=500,
        max_meas_time=20., tickle_voltage=0.005):
    """
    Takes a complex impedance sweep. This will play sine waves on specified
    bias-groups over the current DC bias voltage. This returns a CISweep object.

    Args
    ----
    S : SmurfControl
        Pysmurf Instance
    cfg : DetConfig
        Det config instance
    bgs : array, int
        List of bias groups to run on
    freqs : array, optional
        List of frequencies to sweep over.
    state : str
        Current detector state. Must be 'ob', 'sc', or 'transition'
    nperiods : float
        Number of periods to measure for at each frequency. If the meas_time 
        ends up larger than ``max_meas_time``, ``max_meas_time`` will be used
        instead. This makes it so we don't spend unreasonably long amounts of
        time at higher freqs.
    max_meas_time : float
        Maximum amount of time to wait at any given frequency
    tickle_voltage : float
        Tickle amplitude in low-current-mode volts.
    """
    if state not in ['ob', 'sc', 'transition']:
        raise ValueError("State must be 'ob', 'sc', or 'transition'")

    bgs = np.atleast_1d(bgs)

    if freqs is None:
        freqs = np.logspace(1, np.log10(4e3), 20)
    freqs = np.atleast_1d(freqs)

    run_kwargs = {
        'bgs': bgs, 'freqs': freqs, 'nperiods': nperiods,
        'ma_meas_time': max_meas_time, 'tickle_voltage': tickle_voltage,
    }
    nfreqs = len(freqs)
    nbgs = len(bgs)
    start_times = np.zeros((nbgs, nfreqs), dtype=float)
    stop_times = np.zeros((nbgs, nfreqs), dtype=float)
    sids = np.zeros(nbgs, dtype=int)

    initial_ds_factor = S.get_downsample_factor()
    initial_filter_disable = S.get_filter_disable()

    bgmap_dict = np.load(cfg.dev.exp['bgmap_file'], allow_pickle=True).item()
    bgmap_bands, bgmap_chans, bgmap = [
        bgmap_dict[k] for k in ['bands', 'channels', 'bgmap']
    ]

    bands = []
    channels = []
    scale_array = np.array([S.get_amplitude_scale_array(b) for b in range(8)])
    
    try:
        S.set_downsample_factor(1)
        S.set_filter_disable(1)
        sdl.set_current_mode(S, bgs, 1)
        tickle_voltage /= S.high_low_current_ratio

        init_biases = S.get_tes_bias_bipolar_array()

        for i, bg in enumerate(bgs):
            # We want to run with channels that are in the specified bg
            # and are enabled.
            m = (scale_array[bgmap_bands, bgmap_chans] > 0) & (bgmap == bg)
            channel_mask = bgmap_bands[m] * S.get_number_channels() + bgmap_chans
            bands.extend(bgmap_bands[m])
            channels.extend(bgmap_chans[m])

            sids[i] = sdl.stream_g3_on(S, channel_mask=channel_mask)
            for j, freq in enumerate(freqs):
                meas_time = min(1./freq * nperiods, max_meas_time)
                S.log(f"Tickle with bg={bg}, freq={freq}")
                S.play_sine_tes(bg, tickle_voltage, freq)
                start_times[i, j] = time.time()
                time.sleep(meas_time)
                stop_times[i, j] = time.time()
                S.set_rtm_arb_waveform_enable(0)
                S.set_tes_bias_bipolar(bg, init_biases[bg])

        bands = np.array(bands)
        channels = np.array(channels)
        sweep = CISweep(S, cfg, run_kwargs, sids, start_times, stop_times,
                        bands, channels, state)
        sweep.save()
    finally:
        sdl.set_current_mode(S, bgs, 0)
        S.set_downsample_factor(initial_ds_factor)
        S.set_filter_disable(initial_filter_disable)
        sdl.stream_g3_off(S)

    return sweep
    
def take_complex_impedance_ob_sc(S, cfg, bgs, overbias_voltage=19.9,
                                 tes_bias=15.0, overbias_wait=5.0,
                                 cool_wait=30., **ci_kwargs):
    """
    Takes overbiased and superconducting complex impedance sweeps. These are
    required to analyze any in-transition sweeps.

    Args
    -----
    S : SmurfControl
        Pysmurf Instance
    cfg : DetConfig
        Det config instance
    bgs : array, int
        List of bias groups to run on
    overbias_voltage : float
        Voltage to use to overbias detectors
    tes_bias : float
        Voltage to set detectors to after overbiasing
    overbias_wait : float
        Time to wait at the overbias_voltage
    cool_wait : float
        Time to wait at the tes_bias after overbiasing
    **ci_kwargs : 
        Any additional kwargs will be passed directly to the
        ``take_complex_impedance`` function.
    """

    # Takes SC sweep
    for bg in bgs:
        S.set_tes_bias_bipolar(bg, 0)
    sc = take_complex_impedance(S, cfg, bgs, state='sc', **ci_kwargs)

    S.overbias_tes_all(bias_groups=bgs, overbias_voltage=overbias_voltage,
                       tes_bias=tes_bias, overbias_wait=overbias_wait,
                       cool_wait=cool_wait)
    ob = take_complex_impedance(S, cfg, bgs, state='ob', **ci_kwargs)

    cfg.dev.update_experiment({
        'complex_impedance_sc_path': sc.filepath,
        'complex_impedance_ob_path': ob.filepath
    }, update_file=True)

    return sc, ob
