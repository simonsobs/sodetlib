import time
import os
import traceback
import numpy as np
import sodetlib as sdl
import matplotlib.pyplot as plt
import scipy.optimize
from sodetlib.operations import iv
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
from copy import deepcopy

np.seterr(all='ignore')


def _play_tes_bipolar_waveform(S, bias_group, waveform, do_enable=True,
                                 continuous=True, **kwargs):
    """
    Play a bipolar waveform on the bias group.
    Args
    ----
    bias_group : int
                The bias group
    waveform : float array
                The waveform the play on the bias group.
    do_enable : bool, optional, default True
                Whether to enable the DACs (similar to what is required
                for TES bias).
    continuous : bool, optional, default True
                Whether to play the TES waveform continuously.
    """
    bias_order = S.bias_group_to_pair[:,0]

    dac_positives = S.bias_group_to_pair[:,1]
    dac_negatives = S.bias_group_to_pair[:,2]

    dac_idx = np.ravel(np.where(bias_order == bias_group))

    dac_positive = dac_positives[dac_idx][0]
    dac_negative = dac_negatives[dac_idx][0]

    # https://confluence.slac.stanford.edu/display/SMuRF/SMuRF+firmware#SMuRFfirmware-RTMDACarbitrarywaveforms
    # Target the two bipolar DACs assigned to this bias group:
    S.set_dac_axil_addr(0, dac_positive)
    S.set_dac_axil_addr(1, dac_negative)

    # Must enable the DACs (if not enabled already)
    if do_enable:
        S.set_rtm_slow_dac_enable(dac_positive, 2, **kwargs)
        S.set_rtm_slow_dac_enable(dac_negative, 2, **kwargs)

    # Load waveform into each DAC's LUT table.  Opposite sign so
    # they combine coherenty
    S.set_rtm_arb_waveform_lut_table(0, waveform)
    S.set_rtm_arb_waveform_lut_table(1, -waveform)

    # Enable waveform generation (1=on both DACs)
    S.set_rtm_arb_waveform_enable(3)

    # Continous mode to play the waveform continuously
    if continuous:
        S.set_rtm_arb_waveform_continuous(1)
    else:
        S.set_rtm_arb_waveform_continuous(0)

def _make_step_waveform(S, step_dur, step_voltage, dc_voltage):
    """
    Returns a waveform for a bias step. Waveform will contain step-up
    to (dc_voltage + step_voltage) for `step_dur` seconds, and then a
    step-down to (dc_voltage) for `step_dur` seconds.

    Args
    -----
    S : SmurfControl
        Pysmurf instance
    step_dur : float
        Duration of each step (sec)
    step_voltage : float
        Amplitude of step (V)
    dc_voltage : float
        DC voltage of step

    Returns
    ---------
    sig : np.ndarray
        Waveform to use
    timer_size : float
        Waveform timer size
    """
    # Setup waveform
    sig = np.ones(2048)
    sig *= dc_voltage / (2*S._rtm_slow_dac_bit_to_volt)
    sig[1024:] += step_voltage / (2*S._rtm_slow_dac_bit_to_volt)
    # Use twice the step dur because the signal contains two steps
    timer_size = int(step_dur*2/(6.4e-9 * 2048))
    return sig, timer_size

def play_bias_steps_waveform(S, cfg, bias_group, step_duration, step_voltage,
                             num_steps=5, dc_bias=None):
    """
    Plays bias steps  on a single bias line using the Arb waveform generator

    Args
    ----
    S:
        Pysmurf control instance
    cfg:
        DetConfig instance
    bias_group: int
        Bias groups to play bias step on. Defaults to all 12
    step_duration: float
        Duration of each step in sec
    step_voltage : float
        Step size (volts)
    num_steps: int
        Number of bias steps
    dacs : str
        Which group of DACs to play bias-steps on. Can be 'pos',
        'neg', or 'both'
    """
    if dc_bias is None:
        dc_bias = S.get_tes_bias_bipolar(bias_group)
    sig, timer_size = _make_step_waveform(S, step_duration, step_voltage, dc_bias)
    S.set_rtm_arb_waveform_timer_size(timer_size, wait_done=True)
    _play_tes_bipolar_waveform(S, bias_group, sig)
    start_time = time.time()
    time.sleep(step_duration * (num_steps+1))
    stop_time = time.time()
    S.set_rtm_arb_waveform_enable(0)
    S.set_tes_bias_bipolar(bias_group, dc_bias)
    return start_time, stop_time


def play_bias_steps_dc(S, cfg, bias_groups, step_duration, step_voltage,
                       num_steps=5, dacs='pos'):
    """
    Plays bias steps on a group of bias groups stepping with only one DAC

    Args:
        S:
            Pysmurf control instance
        cfg:
            DetConfig instance
        bias_group: (int, list, optional)
            Bias groups to play bias step on. Defaults to all 12
        step_duration: float
            Duration of each step in sec
        step_voltage : float
            Step size (volts)
        num_steps: int
            Number of bias steps
        dacs : str
            Which group of DACs to play bias-steps on. Can be 'pos',
            'neg', or 'both'
    """
    if bias_groups is None:
        bias_groups = cfg.dev.exp['active_bgs']
    bias_groups = np.atleast_1d(bias_groups)

    if dacs not in ['pos', 'neg', 'both']:
        raise ValueError("Arg dac must be in ['pos', 'neg', 'both']")

    dac_volt_array_low = S.get_rtm_slow_dac_volt_array()
    dac_volt_array_high = dac_volt_array_low.copy()

    bias_order, dac_positives, dac_negatives = S.bias_group_to_pair.T

    for bg in bias_groups:
        bg_idx = np.ravel(np.where(bias_order == bg))
        dac_positive = dac_positives[bg_idx][0] - 1
        dac_negative = dac_negatives[bg_idx][0] - 1
        if dacs == 'pos':
            dac_volt_array_high[dac_positive] += step_voltage
        elif dacs == 'neg':
            dac_volt_array_high[dac_negative] -= step_voltage
        elif dacs == 'both':
            dac_volt_array_high[dac_positive] += step_voltage / 2
            dac_volt_array_high[dac_negative] -= step_voltage / 2

    start = time.time()
    for _ in range(num_steps):
        S.set_rtm_slow_dac_volt_array(dac_volt_array_high)
        time.sleep(step_duration)
        S.set_rtm_slow_dac_volt_array(dac_volt_array_low)
        time.sleep(step_duration)
    stop = time.time()

    return start, stop


def exp_fit(t, A, tau, b):
    """
    Fit function for exponential falloff in bias steps
    """
    return A * np.exp(-t / tau) + b


class BiasStepAnalysis:
    """
    Container to manage analysis of bias steps taken with the take_bias_steps
    function. The main function is ``run_analysis`` and will do a series of
    analysis procedures to create a biasgroup map and calculate DC detector
    parameters and tau_eff:

     - Loads an axis manager with all the data
     - Finds locations of step edges for each bias group
     - Creates a bg map using the isolated bg step responses
     - Gets detector responses of each step and aligns them based on the time
       of the step
     - Computes DC params R0, I0, Pj, Si from step responses
     - Fits exponential to the average step response and estimates tau_eff.

    Most analysis inputs and products will be saved to a npy file so they can
    be loaded and re-analyzed easily on another computer like simons1.

    To load data from an saved step file, you can run::

        bsa = BiasStepAnalysis.load(<path>)

    Attributes:
        tunefile (path): Path of the tunefile loaded by the pysmurf instance
        high_low_current_ratio (float):
            Ratio of high to low current
        R_sh (float):
            Shunt resistance loaded into pysmurf at time of creation
        pA_per_phi0 (float):
            pA_per_phi0, as loaded in pysmurf at time of creation
        rtm_bit_to_volt (float):
            Conversion between bias dac bit and volt
        bias_line_resistance (float):
            Bias line resistance loaded in pysmurf at time of creation
        high_current_mode (bool):
            If high-current-mode was used
        stream_id (string):
            stream_id of the streamer this was run on.
        sid (int):
            Session-id of streaming session
        start, stop (float):
            start and stop time of all steps
        edge_idxs (array(ints) of shape (nbgs, nsteps)):
            Array containing indexes (wrt axis manager) of bias group steps
            for each bg
        edge_signs (array(+/-1) of shape (nbgs, nsteps)):
            Array of signs of each step, denoting whether step is rising or
            falling
        bg_corr (array (float) of shape (nchans, nbgs)):
            Bias group correlation array, stating likelihood that a given
            channel belongs on a given bias group determined from the isolated
            steps
        bgmap (array (int) of shape (nchans)):
            Map from readout channel to assigned bias group. -1 means not
            assigned (that the assignment threshold was not met for any of the
            12 bgs)
        abs_chans (array (int) of shape (nchans)):
            Array of the absolute smurf channel number for each channel in the
            axis manager.
        resp_times (array (float) shape (nbgs, npts)):
            Shared timestamps for each of the step responses in <step_resp> and
            <mean_resp> with respect to the bg-step location, with the step
            occuring at t=0.
        mean_resp (array (float) shape (nchans, npts)):
            Step response averaged accross all bias steps for a given channel
            in Amps.
        step_resp (array (float) shape (nchans, nsteps, npts)):
            Each individual step response for a given channel in amps
        Ibias (array (float) shape (nbgs)):
            DC bias current of each bias group (amps)
        Vbias:
            DC bias voltage of each bias group (volts in low-current mode)
        dIbias (array (float) shape (nbgs)):
            Step current for each bias group (amps)
        dVbias (array (float) shape (nbgs)):
            Step voltage for each bias group (volts in low-current mode)
        dItes (array (float) shape (nchans)):
            Array of tes step heigh for each channel (amps)
        R0 (array (float) shape (nchans)):
            Computed TES resistances for each channel (ohms)
        I0 (array (float) shape (nchans)):
            Computed TES currents for each channel (amps)
        Pj (array (float) shape (nchans)):
            Bias power computed for each channel (W)
        Si (array (float) shape (nchans)):
            Responsivity computed for each channel (1/V)
        step_fit_tmin (array (float) shape (nchans)):
            Time after bias step to start fitting exponential (sec)
        step_fit_popts (array (float) of shape (nchans, 3)):
            Optimal fit parameters (A, tau, b) for the exponential fit of each
            channel
        step_fit_pcovs (array (float) shape (nchans, 3, 3)):
            Fit covariances for each channel
        tau_eff (array (float) shape (nchans)):
            Tau_eff for each channel (sec). Same as step_fit_popts[:, 1].
        R_n_IV (array (float) shape (nchans)):
            Array of normal resistances for each channel pulled from IV
            in the device cfg.
        Rfrac (array (float) shape (nchans)):
            Rfrac of each channel, determined from R0 and the channel's normal
            resistance.
    """

    def __init__(self, S=None, cfg=None, run_kwargs=None):
        self._S = S
        self._cfg = cfg

        self.am = None
        self.edge_idxs = None
        self.transition_range = None

        if S is not None:
            self.meta = sdl.get_metadata(S, cfg)
            self.stream_id = cfg.stream_id

            if run_kwargs is None:
                run_kwargs = {}
            self.run_kwargs = run_kwargs
            self.high_current_mode = run_kwargs.get("high_current_mode", True)


    @classmethod
    def from_dict(cls, data) -> 'BiasStepAnalysis':
        self = cls()
        for k, v in data.items():
            setattr(self, k, v)
        return self

    def to_dict(self) -> Dict[str, Any]:
        data = {}
        saved_fields = [
            # Run data and metadata
            'bands', 'channels', 'sid', 'meta', 'run_kwargs', 'start', 'stop',
            'high_current_mode',
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
        return data

    def save(self, path=None) -> None:
        data = self.to_dict()
        if path is not None:
            np.save(path, data, allow_pickle=True)
            self.filepath = path
        else:
            self.filepath = sdl.validate_and_save(
                'bias_step_analysis.npy', data, S=self._S, cfg=self._cfg,
                make_path=True
            )

    @classmethod
    def load(cls, filepath) -> 'BiasStepAnalysis':
        self = cls.from_dict(np.load(filepath, allow_pickle=True).item())
        self.filepath = filepath
        return self

    def run_analysis(
            self, create_bg_map=False, assignment_thresh=0.3, save_bg_map=True,
            arc=None, base_dir='/data/so/timestreams', step_window=0.03, fit_tmin=1.5e-3,
            transition=None, R0_thresh=30e-3, save=False, bg_map_file=None, fit_tau=True,
    ):
        """
        Runs the bias step analysis.

        Parameters:
            create_bg_map (bool):
                If True, will create a bg map from the step data. If False,
                will use the bgmap from the device cfg
            assignment_thresh (float):
                Correlation threshold for which channels should be assigned to
                particular bias groups.
            save_bg_map (bool):
                If True, will save the created bgmap to disk and set it as
                the bgmap path in the device cfg.
            arc (optional, G3tSmurf):
                G3tSmurf archive. If specified, will attempt to load
                axis-manager using archive instead of sid.
            base_dir (optiional, str):
                Base directory where timestreams are stored. Defaults to
                /data/so/timestreams.
            step_window (float):
                Time after the bias step (in seconds) to use for the analysis.
            fit_tmin (float):
                DEPRECATED!! do not use.
            transition: (tuple, bool, optional)
                DEPRECATED!! do not use.
            R0_thresh (float):
                Any channel with resistance greater than R0_thresh will be
                unassigned from its bias group under the assumption that it's
                crosstalk
            save (bool):
                If true will save the analysis to a npy file.
            bg_map_file (optional, path):
                If create_bg_map is false and this file is not None, use this file
                to load the bg_map.
            fit_tau (bool):
                If True, perform the time constant fit. If False, skip this.
        """
        self._load_am(arc=arc, base_dir=base_dir)
        self._find_bias_edges()
        if create_bg_map:
            self._create_bg_map(assignment_thresh=assignment_thresh,
                                save_bg_map=save_bg_map)
        elif bg_map_file is not None:
            self.bgmap, self.polarity = sdl.load_bgmap(
                self.bands, self.channels, bg_map_file)
        else:
            self.bgmap, self.polarity = sdl.load_bgmap(
                self.bands, self.channels, self.meta['bgmap_file'])

        self._get_step_response(step_window=step_window)
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

        if fit_tau:
            self._fit_tau_effs()

        if save:
            self.save()

    def _load_am(self, arc=None, base_dir='/data/so/timestreams', fix_timestamps=True):
        """
        Attempts to load the axis manager from the sid or return one that's
        already loaded. Also sets the `abs_chans` array.

        TODO: adapt this to work on with a general G3tSmurf archive if
        supplied.
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

    def _find_bias_edges(self, am=None):
        """
        Finds sample indices and signs of bias steps in timestream.

        Returns
        --------
            edge_idxs: list
                List containing the edge sample indices for each bias group.
                There are n_biaslines elements, each one a np.ndarray
                contianing a sample idx for each edge found

            edge_signs: list
                List with the same shape as edge_idxs, containing +/-1
                depending on whether the edge is rising or falling
        """
        if am is None:
            am = self.am

        edge_idxs = [[] for _ in am.biases]
        edge_signs = [[] for _ in am.biases]

        for bg, bias in enumerate(am.biases):
            idxs = np.where(np.diff(bias) != 0)[0]
            signs = np.sign(np.diff(bias)[idxs])

            # Delete double steps
            doubled_idxs = np.where(np.diff(signs) == 0)[0]
            idxs = np.delete(idxs, doubled_idxs)
            signs = np.delete(signs, doubled_idxs)

            edge_idxs[bg] = idxs
            edge_signs[bg] = signs

        self.edge_idxs = edge_idxs
        self.edge_signs = edge_signs

        return edge_idxs, edge_signs

    def _create_bg_map(self, step_window=0.03, assignment_thresh=0.3,
                       save_bg_map=True):
        """
        Creates a bias group mapping from the bg step sweep. The step sweep
        goes down and steps each bias group one-by-one up and down twice. A
        bias group correlation factor is computed by integrating the diff of
        the TES signal times the sign of the step for each bg step. The
        correlation factors are normalized such that the sum across bias groups
        for each channel is 1, and an assignment is made if the normalized
        bias-group correlation is greater than some threshold.

        Saves:
            bg_corr (np.ndarray):
                array of shape (nchans, nbgs) that contain the correlation
                factor for each chan/bg combo (normalized st the sum is 1).
            bgmap (np.ndarray):
                Array of shape (nchans) containing the assigned bg of each
                channel, or -1 if no assigment could be determined
        """
        am = self._load_am()
        if self.edge_idxs is None:
            self._find_bias_edges()

        fsamp = np.nanmean(1./np.diff(am.timestamps))
        npts = int(fsamp * step_window)

        nchans = len(am.signal)
        nbgs = len(am.biases)
        bgs = np.arange(nbgs)
        bg_corr = np.zeros((nchans, nbgs))

        for bg in bgs:
            for i, ei in enumerate(self.edge_idxs[bg]):
                s = slice(ei, ei+npts)
                ts = am.timestamps[s]
                if not (self.start < ts[0] < self.stop):
                    continue
                sig = self.edge_signs[bg][i] * am.signal[:, s]
                bg_corr[:, bg] += np.sum(np.diff(sig), axis=1)
        abs_bg_corr = np.abs(bg_corr)
        normalized_bg_corr = (abs_bg_corr.T / np.sum(abs_bg_corr, axis=1)).T
        normalized_bg_corr[np.isnan(normalized_bg_corr)] = 0.
        bgmap = np.nanargmax(normalized_bg_corr, axis=1)
        m = np.max(normalized_bg_corr, axis=1) < assignment_thresh
        bgmap[m] = -1

        # Calculate the sign of each channel
        self.polarity = np.ones(self.nchans, dtype=int)
        for i in range(self.nchans):
            self.polarity[i] = np.sign(bg_corr[i, bgmap[i]])

        self.bg_corr = normalized_bg_corr
        self.bgmap = bgmap

        return self.bgmap

    def _get_step_response(self, step_window=0.03, pts_before_step=20,
                           restrict_to_bg_sweep=False, am=None):
        """
        Finds each channel's response to the bias step by looking at the signal
        in a small window of <npts> around each edge-index.

        Saves:
            resp_times:
                Array of shape (nbgs, npts) containing the shared timestamps
                for channels on a given bias group
            step_resp:
                Array of shape (nchans, nsteps, npts) containing the response
                of each channel in a window around every step
            mean_resp:
                Array of (nchans, npts) containing the averaged step response
                for each channel
        """
        if am is None:
            am = self.am

        fsamp = np.nanmean(1./np.diff(am.timestamps))
        pts_after_step = int(fsamp * step_window)
        nchans = len(am.signal)
        nbgs = len(am.biases)
        npts = pts_before_step + pts_after_step
        n_edges = np.max([len(ei) for ei in self.edge_idxs])

        sigs = np.full((nchans, n_edges, npts), np.nan)
        ts = np.full((nbgs, npts), np.nan)

        A_per_rad = self.meta['pA_per_phi0'] / (2*np.pi) * 1e-12
        for bg in np.unique(self.bgmap):
            if bg == -1:
                continue
            rcs = np.where(self.bgmap == bg)[0]
            for i, ei in enumerate(self.edge_idxs[bg][:-2]):
                if i < 2:
                    continue
                s = slice(ei - pts_before_step, ei + pts_after_step)
                if np.isnan(ts[bg]).all():
                    ts[bg, :] = am.timestamps[s] - am.timestamps[ei]
                sig = self.edge_signs[bg][i] * am.signal[rcs, s] * A_per_rad
                # Subtracts mean of last 10 pts such that step ends at 0
                sigs[rcs, i, :] = (sig.T - np.nanmean(sig[:, -10:], axis=1)).T

        self.resp_times = ts
        self.step_resp = (sigs.T * self.polarity).T
        self.mean_resp = (np.nanmean(sigs, axis=1).T * self.polarity).T

        return ts, sigs

    def _compute_R0_I0_Pj(self):
        """
        Computes the DC params R0 I0 and Pj
        """
        Ib = self.Ibias[self.bgmap]
        dIb = self.dIbias[self.bgmap]
        dItes = self.dItes
        Ib[self.bgmap == -1] = np.nan
        dIb[self.bgmap == -1] = np.nan
        dIrat = dItes / dIb

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

    def _compute_dc_params(self, R0_thresh=30e-3):
        """
        Calculates Ibias, dIbias, and dItes from axis manager, and then
        runs the DC param calc to estimate R0, I0, Pj, etc. Here you must

        Args:
            transition: (tuple)
                Range of voltage bias values (in low-cur units) where the
                "in-transition" resistance calculation should be used. If True,
                or False, will use in-transition or normal calc for all
                channels. Will default to ``cfg.dev.exp['transition_range']`` or
                (1, 8) if that does not exist or if self._cfg is not set.
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
                Array of shape (nbgs) containing the step current for each
                bias group
            dVbias:
                Array of shape (nbgs) containing the step voltage for each
                bias group
        """
        nbgs = len(self.am.biases)

        Ibias = np.full(nbgs, np.nan)
        dIbias = np.full(nbgs, 0.0, dtype=float)

        # Compute Ibias and dIbias
        bias_line_resistance = self.meta['bias_line_resistance']
        high_low_current_ratio = self.meta['high_low_current_ratio']
        rtm_bit_to_volt = self.meta['rtm_bit_to_volt']
        amp_per_bit = 2 * rtm_bit_to_volt / bias_line_resistance
        if self.high_current_mode:
            amp_per_bit *= high_low_current_ratio
        for bg in range(nbgs):
            if len(self.edge_idxs[bg]) == 0:
                continue
            b0 = self.am.biases[bg, self.edge_idxs[bg][0] - 3]
            b1 = self.am.biases[bg, self.edge_idxs[bg][0] + 3]
            Ibias[bg] = b0 * amp_per_bit
            dIbias[bg] = (b1 - b0) * amp_per_bit

        # Compute dItes
        i0 = np.nanmean(self.mean_resp[:, :5], axis=1)
        i1 = np.nanmean(self.mean_resp[:, -10:], axis=1)
        dItes = i1 - i0

        self.Ibias = Ibias
        self.Vbias = Ibias * bias_line_resistance
        self.dIbias = dIbias
        self.dVbias = dIbias * bias_line_resistance
        self.dItes = dItes

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

    def _fit_tau_effs(self, tmin=1.5e-3, weight_exp=0.3):
        """
        Fits mean step responses to exponential

        Args:
            tmin: float
                DEPRECATED!! do not use.

        Saves:
            step_fit_tmin: np.ndarray
                Array of shape (nchans) containing tmin used for each chan.
            step_fit_popts: np.ndarray
                Array of shape (nchans, 3) containing popt for each chan.
            step_fit_pcovs: np.ndarray
                Array of shape (nchans, 3, 3) containing pcov matrices for each
                channel
            tau_eff: np.ndarray
                Array of shape (nchans) contianing tau_eff for each chan.
        """
        nbgs = len(self.am.biases)
        nchans = len(self.am.signal)
        step_fit_popts = np.full((nchans, 3), np.nan)
        step_fit_pcovs = np.full((nchans, 3, 3), np.nan)
        step_fit_tmin = np.full(nchans, np.nan)

        for bg in range(nbgs):
            rcs = np.where(self.bgmap == bg)[0]
            if not len(rcs):
                continue
            ts = self.resp_times[bg]
            if not len(ts[~np.isnan(ts)]):
                continue
            for rc in rcs:
                resp = self.mean_resp[rc]
                tmin = np.max((0, ts[np.abs(resp) > 0.9*np.max(np.abs(resp))][-1]))
                m = (ts > tmin) & (~np.isnan(resp))
                if not m.any():
                  continue
                offset_guess = np.nanmean(resp[np.abs(ts - ts[-1]) < 0.01])
                bounds = [
                    (-np.inf, 0, -np.inf),
                    (np.inf, 0.1, np.inf)
                ]
                p0 = (0.1, 0.001, offset_guess)
                try:
                    popt, pcov = scipy.optimize.curve_fit(
                        exp_fit, ts[m], resp[m],
                        sigma=ts[m]**weight_exp, p0=p0, bounds=bounds
                    )
                    step_fit_popts[rc] = popt
                    step_fit_pcovs[rc] = pcov
                    step_fit_tmin[rc] = tmin
                except RuntimeError:
                    pass

        self.step_fit_tmin = step_fit_tmin
        self.step_fit_popts = step_fit_popts
        self.step_fit_pcovs = step_fit_pcovs
        self.tau_eff = step_fit_popts[:, 1]

####################################################################
# Plotting functions
####################################################################

def plot_steps(bsa, rc, nsteps=5, offset=0, ax=None, **kw):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    bg = bsa.bgmap[rc]
    i0 = bsa.edge_idxs[bg][2]
    if 3 + nsteps < len(bsa.edge_idxs[bg]):
        i1 = bsa.edge_idxs[bg][3 + nsteps]
    else:
        i1 = bsa.edge_idxs[bg][-1]

    sl = slice(i0, i1)
    ts = bsa.am.timestamps[sl]
    ts = ts - ts[0]
    sig = bsa.am.signal[rc, sl]
    sig = sig - np.nanmean(sig) + offset
    ax.plot(ts, sig, **kw)

    return fig, ax


def plot_step_fit(bsa, rc, ax=None, plot_all_steps=True):
    """
    Plots the step response and fit of a given readout channel
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    bg = bsa.bgmap[rc]
    ts = bsa.resp_times[bg]
    try:
        m = ts > bsa.step_fit_tmin[rc]
    except TypeError:
        m = ts > bsa.step_fit_tmin
    if plot_all_steps:
        for sig in bsa.step_resp[rc]:
            ax.plot(ts*1000, sig, alpha=0.1, color='grey')
    ax.plot(ts*1000, bsa.mean_resp[rc], '.', label='avg step')
    ax.plot(ts[m]*1000, exp_fit(ts[m], *bsa.step_fit_popts[rc]), label='fit')

    text = '\n'.join([
        r'$\tau_\mathrm{eff}$=' + f'{bsa.tau_eff[rc]*1000:0.2f} ms',
        r'$R_\mathrm{TES}=$' + f'{bsa.R0[rc]* 1000:0.2f}' + r'm$\Omega$',
    ])

    ax.text(0.7, 0.1, text, transform=ax.transAxes,
            bbox={'facecolor': 'wheat', 'alpha': 0.3}, fontsize=12)

    ax.legend()
    ax.set(xlabel="Time (ms)", ylabel="Current (Amps)")

    return fig, ax

def plot_iv_res_comparison(bsa, lim=None, bgs=None, ax=None):
    if bgs is None:
        bgs = bsa.bgs
    bgs = np.atleast_1d(bgs)

    iva = iv.IVAnalysis.load(bsa.meta['iv_file'])
    chmap = sdl.map_band_chans(bsa.bands, bsa.channels,
                               iva.bands, iva.channels)
    iv_res = np.full(bsa.nchans, np.nan)

    if ax is None:
        fig, ax = plt.subplots()
        fig.patch.set_facecolor('white')
    else:
        fig = ax.figure

    for bg in bgs:
        m = bsa.bgmap == bg
        vb = bsa.Vbias[bg]
        idx = np.nanargmin(np.abs(iva.v_bias - vb))
        iv_res = iva.R[chmap[m], idx]
        ax.scatter(bsa.R0[m]*1000, iv_res*1000, marker='.', alpha=0.2,
                   label=f'Bias Group {bg}')
    if lim is None:
        lim = (-0.1, 10)
    ax.plot(lim, lim, ls='--', color='grey', alpha=0.4)
    ax.set(xlim=lim, ylim=lim)
    ax.set_xlabel("Bias Step Resistance (mOhm)")
    ax.set_ylabel("IV Resistance (mOhm)")

    return fig, ax

def get_plot_text(bsa):
    """
    Gets text to add to plot textboxes
    """
    return '\n'.join([
        f"stream_id: {bsa.meta['stream_id']}",
        f"sid: {bsa.sid}",
        f"path: {os.path.basename(bsa.filepath)}",
    ])

def plot_bg_assignment(bsa, text_loc=(0.05, 0.85), text_kw=None):
    """
    Plots bias group assignment summary
    """
    xs = np.arange(13)
    ys = np.zeros(13)

    xticklabels = []
    for i in range(12):
        ys[i] = np.sum(bsa.bgmap == i)
        xticklabels.append(str(i))
    ys[12] = np.sum(bsa.bgmap == -1)
    xticklabels.append('None')

    fig, ax = plt.subplots(figsize=(16, 10))
    ax.bar(xs, ys)
    ax.set_xticks(np.arange(13))
    ax.set_xticklabels(xticklabels)
    ax.set_xlabel("Bias Group", fontsize=16)
    ax.set_ylabel("Num Channels", fontsize=16)
    if text_kw is None:
        text_kw = {}
    
    ax.text(
        *text_loc, get_plot_text(bsa), transform=ax.transAxes,
        fontsize=18, bbox=dict(facecolor='white', alpha=0.8)
    )
    return fig, ax


def plot_Rtes(bsa):
    """
    Plots Rtes
    """
    fig, ax = plt.subplots()
    ax.hist(bsa.R0 * 1000, range=(-1, 10), bins=40)
    ax.set_xlabel(r"$R_\mathrm{TES}$ (m$\Omega$)", fontsize=12)
    return fig, ax

def plot_Rfrac(bsa, text_loc=(0.6, 0.8)):
    """
    Plots Rfrac split out by bias group
    """
    iva = iv.IVAnalysis.load(bsa.meta['iv_file'])
    chmap = sdl.map_band_chans(bsa.bands, bsa.channels,
                               iva.bands, iva.channels)

    Rfracs = bsa.R0 / iva.R_n[chmap]
    lim = (-0.1, 1.2)
    nbins=30
    bins = np.linspace(*lim, nbins)
    bgs = np.arange(12)
    hists = np.zeros((len(bgs), nbins-1))
    for bg in bgs:
        m = bsa.bgmap == bg
        if not m.any():
            continue
        hists[bg, :] = np.histogram(Rfracs[m], bins=bins)[0]

    offset = 0.2* np.max(hists)
    fig, ax = plt.subplots(figsize=(18, 10))
    fig.patch.set_facecolor('white')
    for bg, h in enumerate(hists):
        ax.plot(bins[1:], h + offset  * bg, '-o', markersize=4)
        ax.fill_between(bins[1:], h + offset * bg, y2=offset * bg, alpha=0.2)
        txt = f"{int(np.sum(h))} Chans"
        ax.text(1.1, offset*(bg + 0.15), txt, fontsize=20)

    ax.set_xlabel(r"Rfrac", fontsize=24)
    ax.set_yticks(offset * bgs)
    ax.set_yticklabels([f'BG {bg}' for bg in bgs], fontsize=20)
    ax.tick_params(axis='x', labelsize=24)
    txt = get_plot_text(bsa)
    ax.text(*text_loc, txt, bbox=dict(facecolor='white', alpha=0.8), fontsize=20,
            transform=ax.transAxes)
    return fig, ax


@dataclass
class BiasStepConfig:
    bgs: Optional[float] = None
    dc_voltage: float = None
    step_voltage: float = 0.05
    step_duration: float = 0.05
    nsteps: int = 20
    high_current_mode: bool = True
    hcm_wait_time: float = 3.
    dacs: str = 'pos'
    use_waveform: bool = True
    plot_rfrac: bool = True
    show_plots: bool = True
    run_analysis: bool = True
    channel_mask: Optional[np.ndarray] = None
    g3_tag: Optional[str] = None
    stream_subtype: str = 'bias_steps'
    enable_compression: bool = False
    analysis_kwargs: Optional[dict] = None

    def __post_init__(self):
        if self.analysis_kwargs is None:
            self.analysis_kwargs = {}

        if self.dacs not in ['pos', 'neg', 'both']:
            raise ValueError(f'dacs={self.dacs} not in ["pos", "neg", "both"]')

@dataclass
class BgMapConfig(BiasStepConfig):
    dc_voltage: float = 0.3
    step_voltage: float = 0.01
    hcm_wait_time: float = 0
    plot_rfrac: bool = False
    stream_subtype: str = 'bgmap'

    def __post_init__(self):
        super().__post_init__()

        # Makes sure these kwargs are set by default unless otherwise
        # specified
        kw = {
            'assignment_thresh': 0.3, 'create_bg_map': True,
            'save_bg_map': True
        }
        kw.update(self.analysis_kwargs)
        self.analysis_kwargs = kw

    def get_bias_step_cfg(self):
        return BiasStepConfig(
            **{f.name: getattr(self, f.name) for f in fields(BiasStepConfig)}
        )


@sdl.set_action()
def take_bgmap(S, cfg, **bgmap_pars):
    """
    Function to easily create a bgmap. This will set all bias group voltages
    to 0 (since this is best for generating the bg map), and run bias-steps
    with default parameters optimal for creating a bgmap.

    See also:
    ---------
    take_bias_steps
    """
    kw = deepcopy(cfg.dev.exp['bgmap_defaults'])
    kw.update(bgmap_pars)
    bgmap_cfg = BgMapConfig(**kw)

    if bgmap_cfg.bgs is None:
        bgmap_cfg.bgs = cfg.dev.exp['active_bgs']
    bgmap_cfg.bgs = np.atleast_1d(bgmap_cfg.bgs)

    bsa = take_bias_steps(S, cfg, **asdict(bgmap_cfg))

    if hasattr(bsa, 'bgmap'):
        fig, ax = plot_bg_assignment(bsa)
        sdl.save_fig(S, fig, 'bg_assignments.png')
        if bgmap_cfg.show_plots:
            plt.show()
        else:
            plt.close()

    for bg in bgmap_cfg.bgs:
        S.set_tes_bias_bipolar(bg, 0.)

    return bsa


@sdl.set_action()
def take_bias_steps(S, cfg, **bscfg_pars):
    """
    Takes bias step data at the specified DC voltage. Assumes bias lines
    are already in low-current mode (if they are in high-current this will
    not run correction). This function runs bias steps and returns a
    BiasStepAnalysis object, which can be used to easily view and re-analyze
    data.

    This function will first run a "bias group sweep", running multiple steps
    on each bias-line one at a time. This data is used to generate a bgmap.
    After, <nsteps> bias steps are played on all channels simultaneously.

    Parameters:
        S (SmurfControl):
            Pysmurf control instance
        cfg (DetConfig):
            Detconfig instance
        bgs ( int, list, optional):
            Bias groups to run steps on.
        dc_voltage : float, optional
            DC bias used at low end of bias step to avoid divide by zeros in
            the analysis. If unspecified, uses the current value.
        step_voltage (float):
            Step voltage in Low-current-mode units. (i.e. this will be divided
            by the high-low-ratio before running the steps in high-current
            mode)
        step_duration (float):
            Duration in seconds of each step
        nsteps (int):
            Number of steps to run
        high_current_mode (bool):
            If true, switches to high-current-mode. If False, leaves in LCM
            which runs through the bias-line filter, so make sure you
            extend the step duration to be like >2 sec or something
        hcm_wait_time (float):
            Time to wait after switching to high-current-mode.
        dacs : {'pos', 'neg', 'both'}
            Which group of DACs to play bias-steps on.
        run_analysis (bool):
            If True, will attempt to run the analysis to calculate DC params
            and tau_eff. If this fails, the analysis object will
            still be returned but will not contain all analysis results.
        analysis_kwargs (dict, optional):
            Keyword arguments to be passed to the BiasStepAnalysis run_analysis
            function.
        channel_mask : np.ndarray, optional
            Mask containing absolute smurf-channels to write to disk
        g3_tag: string, optional
            Tag to attach to g3 stream.
        stream_subtype : optional, string
            Stream subtype for this operation. This will default to 'bias_steps'.
        enable_compression: bool, optional
            If True, will tell the smurf-streamer to compress G3Frames. Defaults
            to False because this dominates frame-processing time for high
            data-rate streams.
        plot_rfrac : bool
            Create rfrac plot, publish it, and save it. Default is True.
        show_plots : bool
            Show plot in addition to saving when running interactively. Default is False.
    """
    kw = deepcopy(cfg.dev.exp['biasstep_defaults'])
    kw.update(bscfg_pars)
    bscfg = BiasStepConfig(**kw)

    if bscfg.bgs is None:
        bscfg.bgs = cfg.dev.exp['active_bgs']
    bscfg.bgs = np.atleast_1d(bscfg.bgs)

    # Adds to account for steps that may be cut in analysis
    bscfg.nsteps += 4

    if bscfg.dc_voltage is not None:
        for bg in bscfg.bgs:
            S.set_tes_bias_bipolar(bg, bscfg.dc_voltage)

    initial_ds_factor = S.get_downsample_factor()
    initial_filter_disable = S.get_filter_disable()
    initial_dc_biases = S.get_tes_bias_bipolar_array()
    init_current_mode = sdl.get_current_mode_array(S)

    try:
        dc_biases = initial_dc_biases
        if bscfg.high_current_mode:
            dc_biases = dc_biases / S.high_low_current_ratio
            bscfg.step_voltage /= S.high_low_current_ratio
            sdl.set_current_mode(S, bscfg.bgs, 1)
            S.log(f"Waiting {bscfg.hcm_wait_time} sec after switching to hcm")
            time.sleep(bscfg.hcm_wait_time)

        bsa = BiasStepAnalysis(S, cfg, run_kwargs=asdict(bscfg))

        bsa.sid = sdl.stream_g3_on(
            S, tag=bscfg.g3_tag, channel_mask=bscfg.channel_mask, downsample_factor=1,
            filter_disable=True, subtype=bscfg.stream_subtype,
            enable_compression=bscfg.enable_compression
        )

        bsa.start = time.time()

        for bg in bscfg.bgs:
            if bscfg.use_waveform:
                play_bias_steps_waveform(
                    S, cfg, bg, bscfg.step_duration, bscfg.step_voltage,
                    num_steps=bscfg.nsteps
                )
            else:
                play_bias_steps_dc(
                    S, cfg, bg, bscfg.step_duration, bscfg.step_voltage,
                    num_steps=bscfg.nsteps, dacs=bscfg.dacs,
                )
        bsa.stop = time.time()

    finally:
        sdl.stream_g3_off(S)

        # Restores current mode to initial values
        sdl.set_current_mode(S, np.where(init_current_mode == 0)[0], 0)
        sdl.set_current_mode(S, np.where(init_current_mode == 1)[0], 1)

        S.set_downsample_factor(initial_ds_factor)
        S.set_filter_disable(initial_filter_disable)

    if bscfg.run_analysis:
        S.log("Running bias step analysis")
        try:
            bsa.run_analysis(save=True, **bscfg.analysis_kwargs)
            if bscfg.plot_rfrac:
                fig, ax = plot_Rfrac(bsa)
                path = sdl.make_filename(S, 'bs_rfrac_summary.png', plot=True)
                fig.savefig(path)
                S.pub.register_file(path, 'bs_rfrac_summary', plot=True, format='png')
                if not bscfg.show_plots:
                    plt.close(fig)
        except Exception:
            print(f"Bias step analysis failed with exception:")
            print(traceback.format_exc())

    return bsa


def plot_taueff_hist(bsa, text_loc=(0.4, 0.8), fontsize=15, figsize=(10, 6),
                     range=(0, 3)):
    """
    Plots a histogram of the tau_eff measurements for all bias lines.

    Args
    -----
    bsa : BiasStepAnalysis
        analyzed BiasStepAnalysis object
    text_loc : tuple[float]
        Location of textbox in the axis coordinate system.
    fontsize : int
        font size of text in textbox
    figsize : tuple
        Figure size
    range : tuple
        histogram range
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(bsa.tau_eff * 1000, range=range, bins=30)
    ax.set_xlabel("Tau eff (ms)", fontsize=18)
    txt = get_plot_text(bsa)
    ax.text(*text_loc, txt, bbox=dict(facecolor='white', alpha=0.8),
            fontsize=fontsize, transform=ax.transAxes)
    return fig, ax
