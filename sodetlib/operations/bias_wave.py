import time
import os
import traceback
import numpy as np
import sodetlib as sdl
import matplotlib.pyplot as plt
import scipy.optimize
from scipy.signal import welch
from sodetlib.operations import bias_steps, iv

np.seterr(all='ignore')

def play_bias_wave(S, cfg, bias_group, freqs_wave, amp_wave, duration,
                   dc_bias=None):
    """
    Play a sine wave on the bias group.

    Args
    ----
    bias_group : int
        The bias group
    freq_wave : float array
        List of frequencies to play. Unit = Hz.
    amp_wave : float
        Amplitude of sine wave to use. Unit = Volts.
    duration : float
        Duration each sine wave is played  
    dc_bias : float, optional
        Offset voltage of sine wave. Unit = Volts.

    Returns
    -------
    start_times : float list
        unix timestamp for the beginning of each sine wave.
    stop_times : float list
        unix timestamp for the end of each sine wave.
    """
    start_times, stop_times = [], []

    if dc_bias is None:
        dc_bias = S.get_tes_bias_bipolar_array()[bias_group]

    for freq in freqs_wave:
        S.log(f"BL sine wave with bg={bias_group}, freq={freq}")
        S.play_sine_tes(bias_group=bias_group,
                        tone_amp=amp_wave,
                        tone_freq=freq, dc_amp=dc_bias)
        start_times.append(time.time())
        time.sleep(duration)
        stop_times.append(time.time())
        S.set_rtm_arb_waveform_enable(0)
        S.set_tes_bias_bipolar(bias_group, dc_bias)
    return start_times, stop_times

def get_amplitudes(f_c, x, fs = 4000, N = 12000, window = 'hann'):
    """
    Function for calculating the amplitude of a sine wave.

    Args
    ----
    f_c : float
        Target frequency to calculate sine wave amplitude at.
    x : np.array
        Data to analyze. Either can be shape nsamp or ndet x nsamp.
    fs : int
        Sample rate. Unit = samples/second.
    N : int
        Number of samples to calculate FFT on.
    window : str
        Window function to use. See scipy.signal.get_window for a list of 
        valid window functions to use. Default is ``hann``.

    Returns
    -------
    a_peak : float
        Amplitudes of sine waves. Shape is len(x)
    """
    x = np.atleast_2d(x)
    f, p = welch(x, fs = fs, nperseg = N, 
                        scaling = 'spectrum',return_onesided=True,
                        window = window)
    a = np.sqrt(p)
    # Ran into problem with f == f_c when nperseg and len(x) are not right.
    # Ben to add a function to check this or enforce correct choice somehow.
    idx = np.argmin(np.abs(f-f_c))
    a_rms = a[:, idx]
    a_peak = np.sqrt(2)*a_rms
    return a_peak

def get_amplitudes_deprojection(f_c, x, ts):
    """
    Function for calculating the amplitude and phase of a wave by deprojecting sine and cosine components of desired frequency.

    Args
    ----
    f_c : float
        Target frequency to calculate wave amplitude at.
    x : np.array
        Data to analyze. Either can be shape nsamp or ndet x nsamp.
    ts : np.array
        Timestamps of data to analyze. Shape is nsamp.

    Returns
    -------
    a_peak : float
        Amplitudes of sine wave of the desired frequency. Shape is len(x)
    phase : float
        Phase of sine wave of the desired frequency. Shape is len(x)
    """

    vects = np.zeros((2, len(ts)), dtype='float32')
    vects[0, :] = np.sin(2*np.pi*f_c*ts)
    vects[1, :] = np.cos(2*np.pi*f_c*ts)
    I = np.linalg.inv(np.tensordot(vects, vects, (1, 1)))
    coeffs = np.matmul(x, vects.T)
    coeffs = np.dot(I, coeffs.T).T
    coeffs = np.atleast_2d(coeffs)

    a_peak = np.sqrt(coeffs[:,0]**2 + coeffs[:,1]**2)
    phase = np.arctan2(coeffs[:,0], coeffs[:,1])

    return a_peak, phase

class BiasWaveAnalysis:
    """
    UPDATE THE DOCSTRING...Maybe we can ask Ben to do this one.
    """
    def __init__(self, S=None, cfg=None, bgs=None, run_kwargs=None):
        self._S = S
        self._cfg = cfg

        self.bgs = bgs
        self.am = None

        if S is not None:
            self.meta = sdl.get_metadata(S, cfg)
            self.stream_id = cfg.stream_id

            if run_kwargs is None:
                run_kwargs = {}
            self.run_kwargs = run_kwargs
            self.high_current_mode = run_kwargs.get("high_current_mode", True)
          
    def save(self, path=None):
        data = {}
        saved_fields = [
            # Run data and metadata
            'bands', 'channels', 'sid', 'meta', 'run_kwargs', 'start', 'stop',
            'high_current_mode', 'start_times', 'stop_times',
            # Bgmap data
            'bgmap', 'polarity',
            # Step data and fits, including chunked bias data
            'resp_times', 'mean_resp', 'wave_resp', 'wave_biases',
            # Add in tau fit stuff here. The below commented out params are anticipated to be reported for tau analysis. 
            # 'step_fit_tmin', 'step_fit_popts', 'step_fit_pcovs',
            # 'tau_eff',
            # Det param data
            'transition_range', 'Ibias', 'Vbias', 'dIbias', 'dVbias', 'dItes',
            'R0', 'I0', 'Pj', 'Si',
            # From IV's
            'R_n_IV', 'Rfrac',
        ]

        for f in saved_fields:
            if not hasattr(self, f):
                print(f"WARNING: field {f} does not exist... "
                      "defaulting to None")
            data[f] = getattr(self, f, None)

        if path is not None:
            np.save(path, data, allow_pickle=True)
            self.filepath = path
        else:
            self.filepath = sdl.validate_and_save(
                'bias_wave_analysis.npy', data, S=self._S, cfg=self._cfg,
                make_path=True
            )

    @classmethod
    def load(cls, filepath):
        self = cls()
        data = np.load(filepath, allow_pickle=True).item()
        for k, v in data.items():
            setattr(self, k, v)
        self.filepath = filepath
        return self
    
    def run_analysis(self, arc=None, base_dir='/data/so/timestreams',
                     R0_thresh=30e-3, save=False, bg_map_file=None):
        """
        Analyzes data taken with take_bias_waves.

        Parameters
        ----------
        arc (optional, G3tSmurf):
                G3tSmurf archive. If specified, will attempt to load
                axis-manager using archive instead of sid.
        base_dir (optiional, str):
                Base directory where timestreams are stored. Defaults to
                /data/so/timestreams.
        R0_thresh (float):
                Any channel with resistance greater than R0_thresh will be
                unassigned from its bias group under the assumption that it's
                crosstalk
        save (bool):
                If true will save the analysis to a npy file.
        bg_map_file (optional, path):
                If create_bg_map is false and this file is not None, use this file
                to load the bg_map.

        CURRENTLY ONLY INCLUDES CALCULATION OF DC PARAMETERS.
        NO TIME CONSTANT FITS.
        """
        self._load_am(arc=arc, base_dir=base_dir)
        self._split_frequencies()
        if bg_map_file is not None:
            self.bgmap, self.polarity = sdl.load_bgmap(
                self.bands, self.channels, bg_map_file)
        else:
            self.bgmap, self.polarity = sdl.load_bgmap(
                self.bands, self.channels, self.meta['bgmap_file'])

        # GOT TO HERE
        self._get_wave_response()
        self._compute_dc_params(R0_thresh=R0_thresh)

        # Load R_n from IV
        self.R_n_IV = np.full(self.nchans, np.nan)
        # Rfrac determined from R0 and R_n
        self.Rfrac = np.full(self.nchans, np.nan)
        if self.meta['iv_file'] is not None:
            if os.path.exists(self.meta['iv_file']):
                iva = iv.IVAnalysis.load(self.meta['iv_file'])
                chmap = sdl.map_band_chans(
                    self.bands, self.channels, iva.bands, iva.channels
                )
                self.R_n_IV = iva.R_n[chmap]
                self.R_n_IV[chmap == -1] = np.nan
                self.Rfrac = self.R0 / self.R_n_IV

        # Can add in function for fitting time constant from multifrequency data here.
        # self._fit_tau_effs(tmin=fit_tmin)

        if save:
            self.save()

    def _load_am(self, arc=None, base_dir='/data/so/timestreams', fix_timestamps=True):
        """
        Attempts to load the axis manager from the sid or return one that's
        already loaded. Also sets the `abs_chans` array.
        """
        if self.am is None:
            if arc:
                self.am = arc.load_data(self.start, self.stop, stream_id=self.meta['stream_id'])
            else:
                self.am = sdl.load_session(self.meta['stream_id'], self.sid,
                                            base_dir=base_dir)

            # Fix up timestamp jitter from timestamping in software
            if fix_timestamps:
                fsamp, t0 = np.polyfit(self.am.primary['FrameCounter'],
                                        self.am.timestamps, 1)
                self.am.timestamps = t0 + self.am.primary['FrameCounter']*fsamp
            if "det_info" in self.am:
                self.bands = self.am.det_info.smurf.band
                self.channels = self.am.det_info.smurf.channel
            else:
                self.bands = self.am.ch_info.band
                self.channels = self.am.ch_info.channel
            self.abs_chans = self.bands*512 + self.channels
            self.nbgs = len(self.am.biases)
            self.nchans = len(self.am.signal)
        return self.am

    def _split_frequencies(self, am=None):
        """
        Gets indices for each sine wave frequency on each bias group.

        Returns
        -------
        start_idxs : float array
            Array of indices for the start of each frequency sine wave.
            Shape (n_bias_groups, n_frequencies).
        stop_idxs : float array
            Array of indices for the end of each frequency sine wave.
            Shape (n_bias_groups, n_frequencies).
        """
        if am is None:
            am = self.am

        self.start_idxs = np.full(np.shape(self.start_times), -1)
        self.stop_idxs = np.full(np.shape(self.stop_times), -1)

        for bg, bias in enumerate(am.biases[:12]):
            if np.all(np.isnan(self.start_times[bg,:])):
                continue
            for i, [start, stop] in enumerate(zip(self.start_times[bg,:], self.stop_times[bg,:])):
                self.start_idxs[bg, i] = np.argmin(np.abs(am.timestamps - start))
                self.stop_idxs[bg, i] = np.argmin(np.abs(am.timestamps - stop))

        return self.start_idxs, self.stop_idxs
    
    def _get_wave_response(self, am=None):
        """
        Splits up full axis manager into each sine wave.
        Here we enforce the number of points in each sine wave to be equal.

        Returns
        -------
        ts : float array
            Array of timestamps for each sine wave period
            Shape (n_bias_groups, n_frequencies, n_pts_in_sine_wave).
        sigs : float array
            Array of sine response data.
            Shape (n_detectors, n_frequencies, n_pts_in_sine_wave).
        UPDATE THE DOCSTRING
        """
        if am is None:
            am = self.am

        nchans = len(am.signal)
        nbgs = 12
        n_freqs = np.shape(self.start_idxs)[-1]
        npts = np.nanmin(self.stop_idxs-self.start_idxs)

        sigs = np.full((nchans, n_freqs, npts), np.nan)
        biases = np.full((nbgs, n_freqs, npts), np.nan)
        ts = np.full((nbgs, n_freqs, npts), np.nan)

        A_per_rad = self.meta['pA_per_phi0'] / (2*np.pi) * 1e-12
        for bg in np.unique(self.bgmap):
            if bg == -1:
                continue
            rcs = np.where(self.bgmap == bg)[0]
            for i, si in enumerate(self.start_idxs[bg]):
                if np.isnan(ts[bg,i]).all():
                    ts[bg, i, :] = am.timestamps[si:si+npts] - am.timestamps[si]
                sigs[rcs, i, :] = am.signal[rcs, si:si+npts] * A_per_rad
                biases[bg, i, :] = am.biases[bg, si:si+npts]

        self.resp_times = ts
        self.wave_resp = (sigs.T * self.polarity).T
        self.mean_resp = (np.nanmean(sigs, axis=1).T * self.polarity).T
        self.wave_biases = biases

        return ts, sigs, biases
    
    def _compute_dc_params(self, R0_thresh=30e-3):
        """
        Calculates Ibias, dIbias, and dItes from axis manager, and then
        runs the DC param calc to estimate R0, I0, Pj, etc. 
        If multiple frequency are taken, the DC Params are only calculated
        off of the minimum frequency in the array of frequencies.

        Args:
            R0_thresh: (float)
                Any channel with resistance greater than R0_thresh will be
                unassigned from its bias group under the assumption that it's
                crosstalk

        Saves:
            Ibias:
                Array of shape (nbgs) containing the DC bias current for each
                bias group
            Vbias:
                Array of shape (nbgs) containing the DC bias voltage for each
                bias group
            dIbias:
                Array of shape (nbgs) containing the wave current amplitude for
                each bias group
            dVbias:
                Array of shape (nbgs) containing the wave voltage amplitude for
                each bias group
            dItes:
                Array of shape (nchans) containing the wave current amplitude for
                each detector.
        """
        nbgs = 12
        nchans = len(self.am.signal)
        npts = np.nanmin(self.stop_idxs-self.start_idxs)

        Ibias = np.full(nbgs, np.nan)
        dIbias = np.full(nbgs, 0.0, dtype=float)
        dItes = np.full(nchans, np.nan)

        dIbias_phase = np.full(nbgs, 0.0, dtype=float)
        dItes_phase = np.full(nchans, np.nan)

        # Compute Ibias and dIbias
        bias_line_resistance = self.meta['bias_line_resistance']
        high_low_current_ratio = self.meta['high_low_current_ratio']
        rtm_bit_to_volt = self.meta['rtm_bit_to_volt']
        amp_per_bit = 2 * rtm_bit_to_volt / bias_line_resistance
        if self.high_current_mode:
            amp_per_bit *= high_low_current_ratio
        
        for bg in range(nbgs):
            if len(self.start_idxs[bg]) == 0:
                continue
            rcs = np.where(self.bgmap == bg)[0]
            s = slice(int(self.start_idxs[bg][0]),
                      int(self.start_idxs[bg][0] + npts))
            Ibias[bg] = np.nanmean(self.am.biases[bg, s]) * amp_per_bit
            
            dIbias[bg], dIbias_phase[bg] = get_amplitudes_deprojection(self.run_kwargs['freqs_wave'][0],
                                        self.am.biases[bg, s], self.resp_times[bg, 0, :])
            dIbias[bg] = dIbias[bg] * amp_per_bit
          
            dItes[rcs], dItes_phase[rcs] = get_amplitudes_deprojection(self.run_kwargs['freqs_wave'][0],
                                        self.wave_resp[rcs,0,:], self.resp_times[bg, 0, :])

        self.Ibias = Ibias
        self.Vbias = Ibias * bias_line_resistance
        self.dIbias = dIbias
        self.dIbias_phase = dIbias_phase
        self.dVbias = dIbias * bias_line_resistance
        self.dItes = dItes
        self.dItes_phase = dItes_phase

        R0, I0, Pj = self._compute_R0_I0_Pj()

        Si = -1./(I0 * (R0 - self.meta['R_sh']))

        # If resistance is too high, most likely crosstalk so just reset
        # bg mapping and det params
        if R0_thresh is not None:
            m = np.abs(R0) > R0_thresh
            self.bgmap[m] = -1
            for arr in [R0, I0, Pj, Si]:
                arr[m] = np.nan

        self.R0 = R0
        self.I0 = I0
        self.Pj = Pj
        self.Si = Si

        return R0, I0, Pj, Si

    def _compute_R0_I0_Pj(self):
        """
        Computes the DC params R0 I0 and Pj

        Carbon copy from BiasStepAnalysis
        """
        Ib = self.Ibias[self.bgmap]
        dIb = self.dIbias[self.bgmap]
        dItes = self.dItes

        dIb_phase = self.dIbias_phase[self.bgmap]
        dItes_phase = self.dItes_phase
      
        Ib[self.bgmap == -1] = np.nan
        dIb[self.bgmap == -1] = np.nan

        #sign of phase response, is there a better way to do it?
        rel_phase = dItes_phase - dIb_phase 
        rel_phase  = np.isclose(abs(rel_phase), np.pi, rtol = 1e-1)
        rel_phase_sign = np.full(rel_phase.shape, 1.0)
        rel_phase_sign[np.where(rel_phase == True)] = -1.0
      
        dIrat = rel_phase_sign * (dItes / dIb)
    
        R_sh = self.meta["R_sh"]

        I0 = np.zeros_like(dIrat)
        I0_nontransition = Ib * dIrat
        I0_transition = Ib * dIrat / (2 * dIrat - 1)
        I0[dIrat>0] = I0_nontransition[dIrat>0]
        I0[dIrat<0] = I0_transition[dIrat<0]

        Pj = I0 * R_sh * (Ib - I0)
        R0 = Pj / I0**2
        R0[I0 == 0] = 0

        return R0, I0, Pj

@sdl.set_action()
def take_bias_waves(S, cfg, bgs=None, amp_wave=0.05, freqs_wave=[23.0],
                    duration=10, high_current_mode=True, hcm_wait_time=3,
                    run_analysis=True, analysis_kwargs=None, channel_mask=None,
                    g3_tag=None, stream_subtype='bias_waves',
                    enable_compression=False, plot_rfrac=True, show_plots=False):
    """
    Takes bias wave data at the current DC voltage. Assumes bias lines
    are already in low-current mode (if they are in high-current this will
    not run correction). This function runs bias waves and returns a
    BiasWaveAnalysis object, which can be used to easily view and re-analyze
    data.

    Parameters:
        S (SmurfControl):
            Pysmurf control instance
        cfg (DetConfig):
            Detconfig instance
        bgs ( int, list, optional):
            Bias groups to run steps on, defaulting to all 12. Note that the
            bias-group mapping generated by the bias step analysis will be
            restricted to the bgs set here so if you only run with a small
            subset of bias groups, the map might not be correct.
        amp_wave (float):
            Bias wave amplitude voltage in Low-current-mode units.
            This will be divided by the high-low-ratio before running the steps
            in high-current mode.
        freqs_wave (float):
            List of frequencies to take bias wave data at.
        duration (float):
            Duration in seconds of bias wave frequency
        high_current_mode (bool):
            If true, switches to high-current-mode. If False, leaves in LCM
            which runs through the bias-line filter, so make sure you
            extend the step duration to be like >2 sec or something
        hcm_wait_time (float):
            Time to wait after switching to high-current-mode.
        run_analysis (bool):
            If True, will attempt to run the analysis to calculate DC params
            and tau_eff. If this fails, the analysis object will
            still be returned but will not contain all analysis results.
        analysis_kwargs (dict, optional):
            Keyword arguments to be passed to the BiasWaveAnalysis run_analysis
            function.
        channel_mask : np.ndarray, optional
            Mask containing absolute smurf-channels to write to disk
        g3_tag: string, optional
            Tag to attach to g3 stream.
        stream_subtype : optional, string
            Stream subtype for this operation. This will default to 'bias_waves'.
        enable_compression: bool, optional
            If True, will tell the smurf-streamer to compress G3Frames. Defaults
            to False because this dominates frame-processing time for high
            data-rate streams.
        plot_rfrac : bool
            Create rfrac plot, publish it, and save it. Default is True.
        show_plots : bool
            Show plot in addition to saving when running interactively. Default is False.
    """
    if bgs is None:
        bgs = cfg.dev.exp['active_bgs']
    bgs = np.atleast_1d(bgs)

    # Dumb way to get all run kwargs, but we probably want to save these in
    # data object
    freqs_wave = np.sort(np.atleast_1d(freqs_wave)) # Enforces lowest frequency first.
    run_kwargs = {
        'bgs': bgs, 'amp_wave': amp_wave,
        'freqs_wave': freqs_wave, 'duration': duration,
        'high_current_mode': high_current_mode,
        'hcm_wait_time': hcm_wait_time, 'run_analysis': run_analysis,
        'analysis_kwargs': analysis_kwargs, 'channel_mask': channel_mask,
    }

    initial_ds_factor = S.get_downsample_factor()
    initial_filter_disable = S.get_filter_disable()
    initial_dc_biases = S.get_tes_bias_bipolar_array()

    try:
        dc_biases = initial_dc_biases
        init_current_mode = sdl.get_current_mode_array(S)
        if high_current_mode:
            dc_biases = dc_biases / S.high_low_current_ratio
            amp_wave /= S.high_low_current_ratio
            sdl.set_current_mode(S, bgs, 1)
            S.log(f"Waiting {hcm_wait_time} sec after switching to hcm")
            time.sleep(hcm_wait_time)

        bwa = BiasWaveAnalysis(S, cfg, bgs, run_kwargs=run_kwargs)

        bwa.sid = sdl.stream_g3_on(
            S, tag=g3_tag, channel_mask=channel_mask, downsample_factor=1,
            filter_disable=True, subtype=stream_subtype, enable_compression=enable_compression
        )

        bwa.start_times = np.full((12, len(freqs_wave)), np.nan)
        bwa.stop_times = np.full((12, len(freqs_wave)), np.nan)

        bwa.start = time.time()

        for bg in bgs:
            bwa.start_times[bg,:], bwa.stop_times[bg,:] = play_bias_wave(S, cfg, bg,
                                                                        freqs_wave,
                                                                        amp_wave, duration)

        bwa.stop = time.time()

    finally:
        sdl.stream_g3_off(S)

        # Restores current mode to initial values
        sdl.set_current_mode(S, np.where(init_current_mode == 0)[0], 0)
        sdl.set_current_mode(S, np.where(init_current_mode == 1)[0], 1)

        S.set_downsample_factor(initial_ds_factor)
        S.set_filter_disable(initial_filter_disable)

    if run_analysis:
        S.log("Running bias wave analysis")
        try:
            if analysis_kwargs is None:
                analysis_kwargs = {}
            bwa.run_analysis(save=True, **analysis_kwargs)
            if plot_rfrac:
                fig, _ = bias_steps.plot_Rfrac(bwa)
                path = sdl.make_filename(S, 'bw_rfrac_summary.png', plot=True)
                fig.savefig(path)
                S.pub.register_file(path, 'bw_rfrac_summary', plot=True, format='png')
                if not show_plots:
                    plt.close(fig)
        except Exception:
            print(f"Bias wave analysis failed with exception:")
            print(traceback.format_exc())

    return bwa
