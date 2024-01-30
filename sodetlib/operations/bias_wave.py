import time
import os
import traceback
import numpy as np
import sodetlib as sdl
import matplotlib.pyplot as plt
import scipy.optimize
from sodetlib.operations import bias_steps

np.seterr(all='ignore')

def play_bias_wave(S, bias_group, freqs_wave, amp_wave, duration,
                   dc_bias=None):
    """
    UPDATE THE DOCSTRING
    """
    start_times, stop_times = [], []

    if dc_bias is None:
        dc_bias = S.get_tes_bias_bipolar_array()[bias_group]

    for freq in freqs_wave:
        S.log(f"BL sine wave with bg={bias_group}, freq={freq}")
        S.play_sine_tes(bias_group, amp_wave, freq, dc_amp=dc_bias)
        start_times.append(time.time())
        time.sleep(duration)
        stop_times.append(time.time())
        S.set_rtm_arb_waveform_enable(0)
        S.set_tes_bias_bipolar(bias_group, dc_bias)
    return start_times, stop_times

class BiasWaveAnalysis:
    """
    UPDATE THE DOCSTRING
    """
    def __init__(self, S=None, cfg=None, bgs=None, run_kwargs=None):
        self._S = S
        self._cfg = cfg

        self.bgs = bgs
        self.am = None
        # WHAT DO I REPLACE THIS WITH??
        # self.edge_idxs = None
        
        # DOES BEN USE THIS?
        # self.transition_range = None

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
            # Step data and fits
            'resp_times', 'mean_resp', 'step_resp',
            # Step fit data
            'step_fit_tmin', 'step_fit_popts', 'step_fit_pcovs',
            'tau_eff',
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
    
    def run_analysis(self, assignment_thresh=0.3, arc=None, base_dir='/data/so/timestreams', 
                     step_window=0.03, fit_tmin=1.5e-3, transition=None, R0_thresh=30e-3,
                     save=False, bg_map_file=None):
        """
        UPDATE THE DOCSTRING
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

        if create_bg_map and save_bg_map and self._S is not None:
            # Write bgmap after compute_dc_params because bg-assignment
            # will be un-set if resistance estimation is too high.
            ts = str(int(time.time()))
            data = {
                'bands': self.bands,
                'channels': self.channels,
                'sid': self.sid,
                'meta': self.meta,
                'bgmap': self.bgmap,
                'polarity': self.polarity,
            }
            path = os.path.join('/data/smurf_data/bias_group_maps',
                                ts[:5],
                                self.meta['stream_id'],
                                f'{ts}_bg_map.npy')
            os.makedirs(os.path.dirname(path), exist_ok=True)
            sdl.validate_and_save(path, data, S=self._S, cfg=self._cfg,
                                    register=True, make_path=False)
            self._cfg.dev.update_experiment({'bgmap_file': path},
                                            update_file=True)


        self._fit_tau_effs(tmin=fit_tmin)

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
        UPDATE THE DOCSTRING
        """
        if am is None:
            am = self.am

        self.start_idxs = np.full(np.shape(self.start_times), np.nan)
        self.stop_idxs = np.full(np.shape(self.stop_times), np.nan)

        for bg, bias in enumerate(am.biases):
            if np.isnan(self.start_times[bg,:]):
                continue
            for i, [start, stop] in enumerate(zip(self.start_times[bg,:], self.stop_times[bg,:])):
                self.start_idxs[bg, i] = np.argmin(np.abs(am.timestamps - start))
                self.stop_idxs[bg, i] = np.argmin(np.abs(am.timestamps - stop))

        return self.start_idxs, self.stop_idxs
    
    def _get_wave_response(self, step_window=0.03, pts_before_step=20,
                            restrict_to_bg_sweep=False, am=None):
        """
        UPDATE THE DOCSTRING
        """
        if am is None:
            am = self.am

        nchans = len(am.signal)
        nbgs = len(am.biases)
        n_freqs = np.shape(self.start_idxs)[-1]
        npts = np.nanmin(self.stop_idxs-self.start_idxs)

        sigs = np.full((nchans, n_freqs, npts), np.nan)
        ts = np.full((nbgs, npts), np.nan)

        A_per_rad = self.meta['pA_per_phi0'] / (2*np.pi) * 1e-12
        for bg in np.unique(self.bgmap):
            if bg == -1:
                continue
            rcs = np.where(self.bgmap == bg)[0]
            for i, si in enumerate(self.start_idxs[bg]):
                s = slice(si, si + npts)
                if np.isnan(ts[bg]).all():
                    ts[bg, :] = am.timestamps[s] - am.timestamps[si]
                sigs[rcs, i, :] = am.signal[rcs, s] * A_per_rad

        # NEED TO ADD BACK IN POLARITY STUFF HERE DERIVED FROM BGMAP
        self.resp_times = ts
        self.step_resp = sigs.T
        self.mean_resp = np.nanmean(sigs, axis=1)

        return ts, sigs

@sdl.set_action()
def take_bias_waves(S, cfg, bgs=None, amp_wave=0.05, freqs_wave=[23.0],
                    duration=10, high_current_mode=True, hcm_wait_time=3,
                    run_analysis=True, analysis_kwargs=None, dacs='pos',
                    channel_mask=None, g3_tag=None, stream_subtype='bias_waves',
                    enable_compression=False, plot_rfrac=True, show_plots=False):
    """
    UPDATE DOCSTRING 
    """
    if bgs is None:
        bgs = cfg.dev.exp['active_bgs']
    bgs = np.atleast_1d(bgs)

    # Dumb way to get all run kwargs, but we probably want to save these in
    # data object
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