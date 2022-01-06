import time
import numpy as np
import sodetlib.smurf_funcs.smurf_ops as so
from sodetlib.util import make_filename, set_current_mode, save_data
import matplotlib.pyplot as plt
import scipy.optimize
from pysmurf.client.util.pub import set_action

np.seterr(all='ignore')


def play_bias_steps_dc(S, cfg, bias_groups, step_duration, step_voltage,
                       num_steps=5):
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
        num_steps: int
            Number of bias steps
    """
    if bias_groups is None:
        bias_groups = np.arange(12)
    bias_groups = np.atleast_1d(bias_groups)

    dac_volt_array_low = S.get_rtm_slow_dac_volt_array()
    dac_volt_array_high = dac_volt_array_low.copy()

    bias_order, dac_positives, dac_negatives = S.bias_group_to_pair.T

    for bg in bias_groups:
        bg_idx = np.ravel(np.where(bias_order == bg))
        dac_positive = dac_positives[bg_idx][0] - 1
        dac_volt_array_high[dac_positive] += step_voltage

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
        bg_sweep_start (float):
            start time of isolated bg steps
        bg_sweep_stop (float):
            stop time of isolated bg steps
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
            Bias power computed for each channel
        Si (array (float) shape (nchans)):
            Responsivity computed for each channel
        step_fit_tmin (float):
            Time after bias step to start fitting exponential (sec)
        step_fit_popts (array (float) of shape (nchans, 3)):
            Optimal fit parameters (A, tau, b) for the exponential fit of each
            channel
        step_fit_pcovs (array (float) shape (nchans, 3, 3)):
            Fit covariances for each channel
        tau_eff (array (float) shape (nchans)):
            Tau_eff for each channel (sec). Same as step_fit_popts[:, 1].
    """

    def __init__(self, S=None, cfg=None, bgs=None):
        self._S = S
        if S is not None:
            self.tunefile = S.tune_file
            self.high_low_current_ratio = S.high_low_current_ratio
            self.R_sh = S.R_sh
            self.pA_per_phi0 = S.pA_per_phi0
            self.rtm_bit_to_volt = S._rtm_slow_dac_bit_to_volt
            self.bias_line_resistance = S.bias_line_resistance

        self._cfg = cfg
        if cfg is not None:
            self.stream_id = cfg.stream_id

        self.bgs = bgs
        self.am = None
        self.edge_idxs = None
        self.transition_range = None

    def save(self):
        filepath = make_filename(self._S, 'bias_step_analysis.npy')
        self.filepath = filepath

        data = {
            'bands': self.bands,
            'channels': self.channels,
            'sid': self.sid,
        }

        saved_fields = [
            'bg_sweep_start', 'bg_sweep_stop', 'start', 'stop', 'edge_idxs',
            'edge_signs', 'bg_corr', 'bgmap', 'resp_times', 'mean_resp',
            'step_resp', 'high_current_mode', 'transition_range', 'Ibias',
            'Vbias', 'dIbias', 'dVbias', 'dItes', 'R0', 'I0', 'Pj', 'Si',
            'step_fit_tmin', 'step_fit_popts', 'step_fit_pcovs', 'tau_eff',
        ]
        for f in saved_fields:
            if not hasattr(self, f):
                print(f"WARNING: field {f} does not exist... defaulting to None")
            data[f] = getattr(self, f, None)

        save_data(self._S, self._cfg, filepath, data)

    @classmethod
    def load(cls, filepath):
        self = cls()
        data = np.load(filepath, allow_pickle=True).item()
        for k, v in data.items():
            setattr(self, k, v)
        self.filepath = filepath
        return self

    def run_analysis(self, assignment_thresh=0.3, arc=None, step_window=0.03,
                     fit_tmin=1.5e-3, transition=None, R0_thresh=30e-3,
                     save=False):
        """
        Runs the bias step analysis.


        Parameters:
            assignment_thresh (float):
                Correlation threshold for which channels should be assigned to
                particular bias groups.
            arc (optional, G3tSmurf):
                G3tSmurf archive. If specified, will attempt to load
                axis-manager using archive instead of sid.
            step_window (float):
                Time after the bias step (in seconds) to use for the analysis.
            fit_tmin (float):
                tmin used for the fit
            transition: (tuple, bool, optional)
                Range of voltage bias values (in low-cur units) where the
                "in-transition" resistance calculation should be used. If True,
                or False, will use in-transition or normal calc for all
                channels. Will default to ``cfg.dev.exp['transition_range']`` or
                (1, 8) if that does not exist or if self._cfg is not set.
            R0_thresh (float):
                Any channel with resistance greater than R0_thresh will be
                unassigned from its bias group under the assumption that it's
                crosstalk
            save (bool):
                If true will save the analysis to a npy file.
        """
        self._load_am(arc=arc)
        self._find_bias_edges()
        self._create_bg_map(assignment_thresh=assignment_thresh)
        self._get_step_response(step_window=step_window)
        self._compute_dc_params(transition=transition, R0_thresh=R0_thresh)
        self._fit_tau_effs(tmin=fit_tmin)
        if save:
            self.save()

    def _load_am(self, arc=None):
        """
        Attempts to load the axis manager from the sid or return one that's
        already loaded. Also sets the `abs_chans` array.

        TODO: adapt this to work on with a general G3tSmurf archive if
        supplied.
        """
        if self.am is None:
            if arc:
                self.am = arc.load_data(self.start, self.stop)
            else:
                self.am = so.load_session(self._cfg, self.sid)
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
            edge_idxs[bg] = np.where(np.diff(bias) != 0)[0]
            edge_signs[bg] = np.sign(np.diff(bias)[edge_idxs[bg]])

        self.edge_idxs = edge_idxs
        self.edge_signs = edge_signs

        return edge_idxs, edge_signs

    def _create_bg_map(self, step_window=0.03, assignment_thresh=0.3):
        """
        Creates a bias group mapping from the bg step sweep. The step sweep
        goes down and steps each bias group one-by-one up and down twice. A
        bias group correlation factor is computed by integrating the diff of
        the TES signal times the sign of the step for each bg step. The
        correlation factors are normalized such that the sum across bias groups
        for each channel is 1, and an assignment is made if the normalized
        bias-group correlation is greater than some threshold.

        Saves:
            bg_corr: np.ndarray
                array of shape (nchans, nbgs) that contain the correlation
                factor for each chan/bg combo (normalized st the sum is 1).
            bgmap: np.ndarray:
                Array of shape (nchans) containing the assigned bg of each
                channel, or -1 if no assigment could be determined
        """
        if self.bg_sweep_start is None:
            raise ValueError("sweep start and stop times are not set")

        am = self._load_am()

        if self.edge_idxs is None:
            self._find_bias_edges()

        fsamp = np.mean(1./np.diff(am.timestamps))
        npts = int(fsamp * step_window)

        nchans = len(am.signal)
        nbgs = len(am.biases)
        bgs = np.arange(nbgs)
        bg_corr = np.zeros((nchans, nbgs))

        for bg in bgs:
            for i, ei in enumerate(self.edge_idxs[bg]):
                s = slice(ei, ei+npts)
                ts = am.timestamps[s]
                if not (self.bg_sweep_start < ts[0] < self.bg_sweep_stop):
                    continue
                sig = self.edge_signs[bg][i] * am.signal[:, s]
                bg_corr[:, bg] += np.sum(np.diff(sig), axis=1)
        bg_corr = np.abs(bg_corr)
        normalized_bg_corr = (bg_corr.T / np.sum(bg_corr, axis=1)).T
        bgmap = np.argmax(normalized_bg_corr, axis=1)
        m = np.max(normalized_bg_corr, axis=1) < assignment_thresh
        bgmap[m] = -1
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

        fsamp = np.mean(1./np.diff(am.timestamps))
        pts_after_step = int(fsamp * step_window)
        nchans = len(am.signal)
        nbgs = len(am.biases)
        npts = pts_before_step + pts_after_step
        n_edges = np.max([len(ei) for ei in self.edge_idxs])

        sigs = np.full((nchans, n_edges, npts), np.nan)
        ts = np.full((nbgs, npts), np.nan)

        for bg in np.unique(self.bgmap):
            if bg == -1:
                continue
            rcs = np.where(self.bgmap == bg)[0]
            for i, ei in enumerate(self.edge_idxs[bg]):
                s = slice(ei - pts_before_step, ei + pts_after_step)
                if np.isnan(ts[bg]).all():
                    ts[bg, :] = am.timestamps[s] - am.timestamps[ei]
                sig = self.edge_signs[bg][i] * am.signal[rcs, s] * self.pA_per_phi0 / (2*np.pi) * 1e-12
                # Subtracts mean of last 10 pts such that step ends at 0
                sigs[rcs, i, :] = (sig.T - np.mean(sig[:, -10:], axis=1)).T

        self.resp_times = ts
        self.step_resp = sigs
        self.mean_resp = np.nanmean(sigs, axis=1)

        return ts, sigs

    def _compute_R0_I0_Pj(self, transition=False):
        """
        Computes the DC params R0 I0 and Pj
        """
        Ib = self.Ibias[self.bgmap]
        dIb = self.dIbias[self.bgmap]
        dItes = self.dItes
        Ib[self.bgmap == -1] = np.nan
        dIb[self.bgmap == -1] = np.nan

        if not transition:
            # Assume R is constant with dI and  Ites is in the same dir as Ib
            dIrat = np.abs(dItes/dIb)  #
            R0 = self.R_sh * (1./dIrat - 1)
            I0 = Ib * (1 + R0 / self.R_sh)**(-1)
            Pj = I0**2 * R0
        else:
            # Assume dItes is in opposite direction of dIb
            dIrat = -np.abs(dItes / dIb)
            Pj = (Ib**2 * self.R_sh * ((dIrat)**2 - (dIrat))/(1 - 2*(dIrat))**2) #W
            temp = Ib**2 - 4 * Pj / self.R_sh
            R0 = self.R_sh * (Ib + np.sqrt(temp)) / (Ib - np.sqrt(temp))
            I0 = 0.5 * (Ib - np.sqrt(temp))

        return R0, I0, Pj

    def _compute_dc_params(self, transition=None, R0_thresh=30e-3):
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
        amp_per_bit = 2 * self.rtm_bit_to_volt / self.bias_line_resistance
        if self.high_current_mode:
            amp_per_bit *= self.high_low_current_ratio
        for bg in range(nbgs):
            if len(self.edge_idxs[bg]) == 0:
                continue
            b0 = self.am.biases[bg, self.edge_idxs[bg][0]]
            b1 = self.am.biases[bg, self.edge_idxs[bg][0] + 3]
            Ibias[bg] = b0 * amp_per_bit
            dIbias[bg] = (b1 - b0) * amp_per_bit

        # Compute dItes
        i0 = np.mean(self.mean_resp[:, :5], axis=1)
        i1 = np.mean(self.mean_resp[:, -10:], axis=1)
        dItes = i1 - i0

        self.Ibias = Ibias
        self.Vbias = Ibias * self.bias_line_resistance
        self.dIbias = dIbias
        self.dVbias = dIbias * self.bias_line_resistance
        self.dItes = dItes

        default_transition = (1, 8)
        if transition is None:
            if self._cfg is None:
                transition = default_transition
            else:
                transition = self._cfg.dev.exp.get('transition_range',
                                                   default_transition)

        tmask = np.zeros(self.nchans, dtype=bool)
        if transition is True or transition is False:
            tmask[:] = transition
        else:
            # Calculate transition mask based on bias voltage and specified 
            # range
            tr0, tr1 = transition
            vb = self.Vbias[self.bgmap]
            tmask = (tr0 < vb) & (vb < tr1)

        R0, I0, Pj = self._compute_R0_I0_Pj(transition=False)
        R0_trans, I0_trans, Pj_trans = self._compute_R0_I0_Pj(transition=True)

        R0[tmask] = R0_trans[tmask]
        I0[tmask] = I0_trans[tmask]
        Pj[tmask] = Pj_trans[tmask]

        Si = -1./(I0 * (R0 - self.R_sh))

        # If resistance is too high, most likely crosstalk so just reset
        # bg mapping and det params
        if R0_thresh is not None:
            m = R0 > R0_thresh
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
                Amount of time after the step at which to start the fit

        Saves:
            step_fit_tmin: float
                tmin used for the fit
            step_fit_popts: np.ndarray
                Array of shape (nchans, 3) containing popt for each chan.
            step_fit_pcovs: np.ndarray
                Array of shape (nchans, 3, 3) containing pcov matrices for each
                channel
            tau_eff: np.ndarray
                Array of shape (nchans) contianing tau_eff for each chan.
        """
        nchans = len(self.am.signal)
        step_fit_popts = np.full((nchans, 3), np.nan)
        step_fit_pcovs = np.full((nchans, 3, 3), np.nan)

        for bg in range(12):
            rcs = np.where(self.bgmap == bg)[0]
            if not len(rcs):
                continue
            ts = self.resp_times[bg]
            for rc in rcs:
                resp = self.mean_resp[rc]
                m = (ts > tmin) & (~np.isnan(resp))
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
                except RuntimeError:
                    pass

        self.step_fit_tmin = tmin
        self.step_fit_popts = step_fit_popts
        self.step_fit_pcovs = step_fit_pcovs
        self.tau_eff = step_fit_popts[:, 1]

    def plot_step_fit(self, rc, ax=None, plot_all_steps=True):
        """
        Plots the step response and fit of a given readout channel
        """
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        bg = self.bgmap[rc]
        ts = self.resp_times[bg]
        m = ts > self.step_fit_tmin
        if plot_all_steps:
            for sig in self.step_resp[rc]:
                ax.plot(ts*1000, sig, alpha=0.1, color='grey')
        ax.plot(ts*1000, self.mean_resp[rc], '.', label='avg step')
        ax.plot(ts[m]*1000, exp_fit(ts[m], *self.step_fit_popts[rc]), label='fit')

        text = r'$\tau_\mathrm{eff}$=' + f'{self.tau_eff[rc]*1000:0.2f} ms'
        ax.text(0.7, 0.1, text, transform=ax.transAxes,
                bbox={'facecolor': 'wheat', 'alpha': 0.3}, fontsize=12)

        ax.legend()
        ax.set(xlabel="Time (ms)", ylabel="Current (Amps)")

        return fig, ax


@set_action()
def take_bias_steps(S, cfg, bgs=None, step_voltage=0.05, step_duration=0.05,
                    nsteps=20, nsweep_steps=5, high_current_mode=True,
                    hcm_wait_time=3, run_analysis=True, analysis_kwargs=None):
    """
    Takes bias step data at the current DC voltage. Assumes bias lines
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
            Bias groups to run steps on, defaulting to all 12. Note that the
            bias-group mapping generated by the bias step analysis will be
            restricted to the bgs set here so if you only run with a small
            subset of bias groups, the map might not be correct.
        step_voltage (float):
            Step voltage in Low-current-mode units. (i.e. this will be divided
            by the high-low-ratio before running the steps in high-current
            mode)
        step_duration (float):
            Duration in seconds of each step
        nsteps (int):
            Number of steps to run
        nsweep_steps (int):
            Number of steps to run per bg in the bg mapping sweep
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
            Keyword arguments to be passed to the BiasStepAnalysis run_analysis
            function.
    """
    if bgs is None:
        bgs = np.arange(12)
    bgs = np.atleast_1d(bgs)

    bsa = BiasStepAnalysis(S, cfg, bgs)
    bsa.high_current_mode = high_current_mode

    initial_ds_factor = S.get_downsample_factor()
    initial_filter_disable = S.get_filter_disable()
    initial_dc_biases = S.get_tes_bias_bipolar_array()

    S.set_downsample_factor(1)
    S.set_filter_disable(1)

    dc_biases = initial_dc_biases
    if high_current_mode:
        dc_biases = dc_biases / S.high_low_current_ratio
        step_voltage /= S.high_low_current_ratio

        set_current_mode(S, bgs, 1)
        S.log(f"Waiting {hcm_wait_time} sec after switching to hcm")
        time.sleep(hcm_wait_time)

    bsa.sid = so.stream_g3_on(S, tag='bias_steps')
    try:
        bsa.start = time.time()

        bsa.bg_sweep_start = time.time()
        for bg in bgs:
            play_bias_steps_dc(S, cfg, bg, step_duration, step_voltage, nsweep_steps)
        bsa.bg_sweep_stop = time.time()

        play_bias_steps_dc(S, cfg, bgs, step_duration, step_voltage, nsteps)
        bsa.stop = time.time()
    finally:
        so.stream_g3_off(S)
        if high_current_mode:
            set_current_mode(S, bgs, 0)

        S.set_downsample_factor(initial_ds_factor)
        S.set_filter_disable(initial_filter_disable)

    if run_analysis:
        S.log("Running bias step analysis")
        try:
            if analysis_kwargs is None:
                analysis_kwargs = {}
            bsa.run_analysis(save=True, **analysis_kwargs)
        except Exception as e:
            print(f"Bias step analysis failed with exception {e}")

    return bsa

