class SingleBand:
    """
    SingleBand is a class to operate on a single band (bias line) for a given SMuRF controller.
    """
    def __init__(self, S, cfg, band, auto_startup=False, verbose=True):
        """
        :param S: SMuRF controller object.
        :param cfg: SMuRF configuration object
        :param band: int - band identifier (bias line)
        :param auto_startup: bool - default is False. When True, the method .startup() runs,
                                    setting up the band in a standard script.
        :param verbose: bool - default is True: When True, toggles print() statements for the
                               various actions taken by the methods in this class. False,
                               runs silently.
        """
        self.S = S
        self.cfg = cfg
        self.band = band
        self.verbose = verbose
        if auto_startup:
            self.startup()

    def startup(self):
        """
        A standard (default) setup for a single band (bias line)
        """
        if self.verbose:
            print(f'setting up band {self.band}')
        self.set_att_dc()
        self.set_tone_power()
        self.est_phase_delay()
        self.set_synthesis_scale()
        self.find_freq()
        self.run_serial_gradient_descent()
        self.run_tracking_setup()

    def set_att_dc(self):
        self.S.set_att_dc(self.band, self.cfg.dev.bands[self.band]["dc_att"])
        if self.verbose:
            print(f'band {self.band} dc_att {self.S.get_att_dc(self.band)}')

    def set_tone_power(self):
        self.S.amplitude_scale[self.band] = self.cfg.dev.bands[self.band]["drive"]
        if self.verbose:
            print(f"band {self.band} tone power {self.S.amplitude_scale[self.band]}")

    def est_phase_delay(self):
        if self.verbose:
            print('estimating phase delay...')
        self.S.estimate_phase_delay(self.band)

    def set_synthesis_scale(self):
        if self.verbose:
            print("setting synthesis scale...")
        # hard coding it for the current fw
        self.S.set_synthesis_scale(self.band, 1)

    def find_freq(self):
        if self.verbose:
            print("running S.find_freq...")

        self.S.find_freq(self.band, tone_power=self.cfg.dev.bands[self.band]["drive"], make_plot=True)

    def setup_notches(self):
        if self.verbose:
            print("running setup notches")
        self.S.setup_notches(self.band, tone_power=self.cfg.dev.bands[self.band]["drive"], new_master_assignment=True)

    def run_serial_gradient_descent(self):
        if self.verbose:
            print("running serial gradient descent and eta scan")
            self.S.run_serial_gradient_descent(self.band)
            self.S.run_serial_eta_scan(self.band)

    def run_tracking_setup(self):
        if self.verbose:
            print("running tracking setup...")
        self.S.set_feedback_enable(self.band, 1)
        self.S.tracking_setup(
            self.band,
            reset_rate_khz=self.cfg.dev.bands[self.band]["flux_ramp_rate_khz"],
            fraction_full_scale=self.cfg.dev.bands[self.band]["frac_pp"],
            make_plot=False,
            save_plot=False,
            show_plot=False,
            channel=self.S.which_on(self.band),
            nsamp=2 ** 18,
            lms_freq_hz=None,
            meas_lms_freq=True,
            feedback_start_frac=self.cfg.dev.bands[self.band]["feedback_start_frac"],
            feedback_end_frac=self.cfg.dev.bands[self.band]["feedback_end_frac"],
            lms_gain=self.cfg.dev.bands[self.band]["lms_gain"])
        if self.verbose:
            print("  tracking setup complete")

    def check_lock(self):
        if self.verbose:
            print('checking tacking lock...')
        self.S.check_lock(self.band,
                          reset_rate_khz=self.cfg.dev.bands[self.band]["flux_ramp_rate_khz"],
                          fraction_full_scale=self.cfg.dev.bands[self.band]["frac_pp"],
                          lms_freq_hz=None,
                          feedback_start_frac=self.cfg.dev.bands[self.band]["feedback_start_frac"],
                          feedback_end_frac=self.cfg.dev.bands[self.band]["feedback_end_frac"],
                          lms_gain=self.cfg.dev.bands[self.band]["lms_gain"])
        if self.verbose:
            print('  tracking lock check complete.')
