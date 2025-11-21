import argparse
import os
import yaml
from collections import OrderedDict
import sys
import time
import shutil


class YamlReps:
    class FlowSeq(list):
        """Represents a list as a flow sequencey by default"""
        pass

    class Odict(OrderedDict):
        """
        Represents an ordered dict in the same way as a dict, but with
        ordering upheld.
        """
        pass

    @classmethod
    def setup_reps(cls):
        def flowseq_rep(dumper, data):
            return dumper.represent_sequence(u'tag:yaml.org,2002:seq', data,
                                             flow_style=True)
        yaml.add_representer(cls.FlowSeq, flowseq_rep)

        def odict_rep(dumper, data):
            return dumper.represent_mapping('tag:yaml.org,2002:map',data.items())
        yaml.add_representer(cls.Odict, odict_rep)


YamlReps.setup_reps()

# Default values for device cfg entries
exp_defaults = {
    # General stuff
    'downsample_factor': 20, 'coupling_mode': 'dc', 'synthesis_scale': 1,
    'active_bands': [0, 1, 2, 3, 4, 5, 6, 7],
    'active_bgs': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    'downsample_mode': 'internal',
    'enable_compression': True,

    # Downsample filter parameters
    # cutoff freq in Hz reverse engineered to produce pysmurf default filter
    "downsample_filter": {"order": 4, "cutoff_freq": 63.0},

    # Dict of device-specific default params to use for take_iv
    'iv_defaults': {},

    # Dict of device-specific default params to use for take_bgmap
    'bgmap_defaults': {},

    # Dict of device-specific default params to use for take_bias_steps
    'biasstep_defaults': {},

    # Amp stuff
    "amps_to_bias": ['hemt', 'hemt1', 'hemt2', '50k', '50k1', '50k2'],
    "amp_enable_wait_time": 10.0, "amp_step_wait_time": 0.2,

    "amp_50k_init_gate_volt": -0.5, "amp_50k_drain_current": 15.0,
    "amp_50k_gate_volt": None, "amp_50k_drain_current_tolerance": 0.2,
    "amp_hemt_init_gate_volt": -1.0, "amp_hemt_drain_current": 8.0,
    "amp_hemt_gate_volt": None, "amp_hemt_drain_current_tolerance": 0.2,

    "amp_50k1_init_gate_volt": -0.5, "amp_50k1_drain_current": 15.0,
    "amp_50k1_gate_volt": None, "amp_50k1_drain_current_tolerance": 0.2,
    "amp_50k1_drain_volt": 4,
    "amp_50k2_init_gate_volt": -0.5, "amp_50k2_drain_current": 15.0,
    "amp_50k2_gate_volt": None, "amp_50k2_drain_current_tolerance": 0.2,
    "amp_50k2_drain_volt": 4,

    "amp_hemt1_init_gate_volt": -1.0, "amp_hemt1_drain_current": 8.0,
    "amp_hemt1_gate_volt": None, "amp_hemt1_drain_current_tolerance": 0.2,
    "amp_hemt1_drain_volt": 0.6,
    "amp_hemt2_init_gate_volt": -1.0, "amp_hemt2_drain_current": 8.0,
    "amp_hemt2_gate_volt": None, "amp_hemt2_drain_current_tolerance": 0.2,
    "amp_hemt2_drain_volt": 0.6,

    # Find freq
    'res_amp_cut': 0.01, 'res_grad_cut': 0.01,

    # Tracking stuff
    "flux_ramp_rate_khz": 4, "init_frac_pp": 0.4, "nphi0": 5,
    "f_ptp_range": [10, 200], "df_ptp_range": [0, 100], "r2_min": 0.9,
    "min_good_tracking_frac": 0.8,
    'feedback_start_frac': 0.02, 'feedback_end_frac': 0.98,

    # Misc files
    "tunefile": None, "bgmap_file": None, "iv_file": None,
    "res_fit_file": None,

    # Overbiasing
    'overbias_wait': 2.0,
    'overbias_sleep_time_sec': 300,

    # Biasing
    'rfrac':[0.3, 0.6],
}
band_defaults = {
    # General
    "band_delay_us": None, "uc_att": None, "dc_att": None,
    "attens_optimized": False, "tone_power": 12,

    # Band-specific tracking stuff
    "lms_gain": 0, "feedback_gain": 2048, "frac_pp": None, "lms_freq_hz": None,

    # Gradient Descent Parameters
    "gradientDescentMaxIters": 100,
    "gradientDescentAverages": 2, 
    "gradientDescentGain": 0.001,
    "gradientDescentConvergeHz": 500,
    "gradientDescentStepHz": 5000,
    "gradientDescentMomentum": 1,
    "gradientDescentBeta": 0.1,

    # Fixed tones
    "fixed_tones": {"enabled": False, "freq_offsets": [], "channels": [], "tone_power": 0},
}
bg_defaults = {
    'overbias_voltage': 19.9,
    'cool_voltage': 10.0,
    'testbed_100mK_bias_voltage': 10.0
}


class DeviceConfig:
    """
    Configuration object containing all "device" specific information. That is,
    parameters that will probably change based on which test-bed you are
    using and what hardware is loaded into that test-bed. Device configuration
    info is split into three groups: ``experiment``, ``bias_groups`` and
    ``bands``.

    The ``experiment`` group contains data that is relevant to the whole
    experiment, e.g. the tunefile that should be loaded or the amplifier
    currents and voltages that should be set.

    The ``bias_groups`` group contains data for each individual bias group.
    For instance this is where the bias_high/low/step values are stored for each
    bias group.

    The ``bands`` group contains data for each individual band. For instance,
    the dc_att and drive values.

    Attributes:
        exp (dict):
            Dict with ``experiment`` config options
        bias_groups (list):
            List of 12 bias group configuration dictionaries.
        bands (list):
            List of 8 band configuration dictionaries.
    """
    def __init__(self):
        self.load_defaults()
        self.source_file = None

    def load_defaults(self):
        self.bands = [band_defaults.copy() for _ in range(8)]
        self.bias_groups = [bg_defaults.copy() for _ in range(12)]
        self.exp = exp_defaults.copy()

    @classmethod
    def from_dict(cls, data):
        """
        Creates a DeviceConfig object from a dictionary.
        """
        self = cls()
        for k, v in data['bands'].items():
            if k.lower().startswith('amc'):  # Key formatted like AMC[i]
                amc_index = int(k[4])
                self._load_amc(amc_index, v)
            if k.lower().startswith('band'):  # Key formatted like Band[i]
                band_index = int(k[5])
                self.bands[band_index].update(v)


        # Loads bias groups from config file
        for i, bg in enumerate(self.bias_groups):
            for k, vlist in data['bias_groups'].items():
                self.bias_groups[i][k] = vlist[i]

        # Loads experiment config data
        self.exp.update(data['experiment'])
        return self

    @classmethod
    def from_yaml(cls, filename):
        """
        Creates a DeviceConfig object from a dev-cfg yaml file.

        Args:
            filename (path): path to device-config file.
        """
        filename = os.path.abspath(os.path.expandvars(filename))
        if not os.path.exists(filename):
            # Just return default device cfg
            print(f"File {filename} doesn't exist! Creating it wth default params")
            self = cls()
            self.source_file = filename
            self.update_file()
            return self
        else:
            with open(filename) as f:
                data = yaml.safe_load(f)
            self = cls.from_dict(data)
            self.source_file = filename
            return self

    def _load_amc(self, amc_index, data):
        """Loads amc data all at once"""
        for k, v in data.items():
            if len(v) != 4:
                raise ValueError("All data in AMC entry must be formatted as a "
                                 "list of length 4.")
            for i in range(4):
                band_index = amc_index*4 + i
                self.bands[band_index][k] = v[i]

    def dump(self, path, clobber=False):
        """
        Dumps all device configuration info to a file.

        Args:
            path (path):
                Location to dump config info to. This file must not already
                exist.
            clobber (bool):
                If true will overwrite existing file. Defaults to false.
        """
        if os.path.exists(path) and not clobber:
            raise FileExistsError(f"Can't dump device config! Path {path}"
                                   "already exists!!")

        def _format_yml(val):
            """Converts np dtypes to python types for yaml files"""
            if hasattr(val, 'dtype'):
                return val.item()
            elif isinstance(val, tuple):
                return list(val)
            else:
                return val

        data = YamlReps.Odict()
        data['experiment'] = {
            k: _format_yml(v) for k, v in self.exp.items()
        }

        data['bias_groups'] = {
            k: YamlReps.FlowSeq([_format_yml(bg[k]) for bg in self.bias_groups])
            for k in self.bias_groups[0].keys()
        }
        data['bands'] = YamlReps.Odict([
            (f'AMC[{i}]', {
                k: YamlReps.FlowSeq([
                    _format_yml(b[k]) for b in self.bands[4*i:4*i+4]
                ])
                for k in self.bands[0].keys()
            }) for i in [0, 1]])

        with open(path, 'w') as f:
            yaml.dump(data, f)

    def update_band(self, band_index, data, update_file=False):
        """
        Updates band configuration object.

        Args:
            band_index (int 0-7):
                Index of band toupdate
            data (dict):
                Dictionary of parameters to update. All parameters must exist in
                the loaded dev-cfg file.
        """
        band = self.bands[band_index]
        for k, v in data.items():
            if k not in band.keys():
                for b in self.bands:
                    b[k] = None
            band[k] = v
        if update_file and self.source_file is not None:
            self.update_file()

    def update_bias_group(self, bg_index, data, update_file=False):
        """
        Updates bias group configuration object.

        Args:
            bg_index (int 0-11):
                Index of bias group to update.
            data (dict):
                Dictionary of parameters to update. All parameters must exist in
                the loaded dev-cfg file.
        """
        bg = self.bias_groups[bg_index]
        for k, v in data.items():
            if k not in bg.keys():
                for _bg in self.bias_groups:
                    _bg[k] = None
            bg[k] = v
        if update_file and self.source_file is not None:
            self.update_file()

    def update_experiment(self, data, update_file=False):
        """
        Updates ``experiment`` configuration object.

        Args:
            data (dict):
                Dictionary of parameters to update. All parameters must exist in
                the loaded dev-cfg file.
        """
        for k, v in data.items():
            self.exp[k] = v
        if update_file and self.source_file is not None:
            self.update_file()

    def update_file(self):
        """
        Updates the device cfg file.1
        """
        self.dump(self.source_file, clobber=True)


    def apply_to_pysmurf_instance(self, S, load_tune=True):
        """
        Applies basic device config params (amplifier biases, attens,
        tone_powers) to a pysmurf instance based on the device cfg values. Note
        that this does not set any of the tracking-related params since this
        shouldn't replace tracking_setup.
        """
        S.set_amplifier_bias(
            bias_hemt=self.exp['amp_hemt_Vg'],
            bias_50k=self.exp['amp_50k_Vg']
        )

        if load_tune:
            tunefile = self.exp['tunefile']
            if os.path.exists(tunefile):
                S.load_tune(tunefile)
            else:
                print(f"Cannot load tunefile {tunefile} because it doesn't exist")

        for b in S._bands:
            band_cfg = self.bands[b]
            if 'uc_att' in band_cfg:
                S.set_att_uc(b, band_cfg['uc_att'])
            if 'dc_att' in band_cfg:
                S.set_att_dc(b, band_cfg['dc_att'])
            if 'drive' in band_cfg:
                # Sets the tone power of all enabled channels to `drive`
                amp_scales = S.get_amplitude_scale_array(b)
                amp_scales[amp_scales != 0] = band_cfg['drive']
                S.set_amplitude_scale_array(b, amp_scales)


def make_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()

    parser.add_argument_group("DetConfig Options")
    parser.add_argument('--sys-file',
                        help="Path to sys-config file. "
                             "Defaults to ``$SMURF_CONFIG_DIR/sys_config.yml``")
    parser.add_argument('--dev-file',
                        help="Path to device-config file. "
                             "Defaults to the path specified in the sys_config.")
    parser.add_argument('--pysmurf-file',
                        help="Path to Pysmurf config file. "
                        "Defaults to the path specified in the sys_config.")
    parser.add_argument('--uxm-file',
                        help="Path to the uxm file. "
                        "Defaults to the path specified in the sys_config")
    parser.add_argument('--slot', '-N', type=int, help="Smurf slot")
    parser.add_argument('--dump-configs', '-D', action='store_true',
                        help="If true, all config info will be written to "
                             "the pysmurf output directory")
    return parser

class DetConfig:
    """
    General Configuration class for SODETLIB.

    Attributes:
        sys_file (path): Path to sys-config file used.
        dev_file (path): Path to device-config file used
        pysmurf_file (path): Path to pysmurf-config file used.
        uxm_file (path): Path to uxm file
        sys (dict):
            System configuration dictionary, generated from the sys_config.yml
            file or ``--sys-file`` command line argument.
        dev (dict):
            Device configuration dictionary, generated from the device_config
            file specified in the sys_config or specified by the ``--dev-file``
            command line argument
    """
    def __init__(self, slot=None, sys_file=None, dev_file=None,
                 pysmurf_file=None, uxm_file=None):
        self.sys_file = sys_file
        self.dev_file = dev_file
        self.pysmurf_file = pysmurf_file
        self.uxm_file = uxm_file
        self.slot = slot
        self.sys = None
        self.dev: DeviceConfig = None
        self.dump = None  # Whether config files should be dumped

        self.S = None

        self._parser: argparse.ArgumentParser = None
        self._argparse_args = None


    def parse_args(self, parser=None, args=None):
        """
        Parses command line arguments along with det_config arguments and
        loads the correct configuration. See ``load_config`` for how config
        files are determined.

        Args:
            parser (argparse.ArgumentParser, optional):
                custom argparse parser to parse args with. If not specified,
                will create its own.
            args (list, optional):
                List of command line arguments to parse.
                Defaults to the command line args.
        """
        self._argparse_args = args
        parser = make_parser(parser)
        self._parser = parser

        args = parser.parse_args(args=args)

        if args.sys_file is None:
            args.sys_file = self.sys_file
        if args.pysmurf_file is None:
            args.pysmurf_file = self.pysmurf_file
        if args.dev_file is None:
            args.dev_file = self.dev_file
        if args.uxm_file is None:
            args.uxm_file = self.uxm_file

        self.load_config_files(
            slot=args.slot, sys_file=args.sys_file, dev_file=args.dev_file,
            pysmurf_file=args.pysmurf_file, uxm_file=args.uxm_file
        )

        return args

    def load_config_files(self, slot=None, sys_file=None, dev_file=None,
                          pysmurf_file=None, uxm_file=None):
        """
        Loads configuration files. If arguments are not specified, sensible
        defaults will be chosen.

        Args:
            slot (int, optional):
                pysmurf slot number. If None and there is only a single slot in
                the sys_file, it will use that slot. Otherwise, it will throw
                and error.
            sys_file (path, optional):
                Path to sys config file. If None, defaults to
                $SMURF_CONFIG_DIR/sys_config.yml
            dev_file (path, optional):
                Path to the Device file. If None, defaults to the device file
                specified in the sys-config file (device_configs[slot-2]).
            pysmurf_file (path, optional):
                Path to pysmurf config file. If None, defaults to the file
                specified in the sys-config (pysmurf_configs[slot-2]).
            uxm_file (path, optional):
                Path to uxm file
        """
        # This should be the same for every smurf-srv
        if 'SMURF_CONFIG_DIR' in os.environ:
            cfg_dir = os.environ['SMURF_CONFIG_DIR']
        elif 'OCS_CONFIG_DIR' in os.environ:
            cfg_dir = os.environ['OCS_CONFIG_DIR']
        else:
            raise ValueError(
                "SMURF_CONFIG_DIR or OCS_CONFIG_DIR must be set in the environment"
            )

        if sys_file is None:
            self.sys_file = os.path.join(cfg_dir, 'sys_config.yml')
        else:
            self.sys_file = sys_file

        # Load system settings
        with open(self.sys_file) as f:
            self.sys = yaml.safe_load(f)

        if slot is not None:
            self.slot = slot
        elif len(self.sys['slot_order']) == 1:
            self.slot = self.sys['slot_order'][0]
        else:
            raise ValueError(
                "Slot could not be automatically determined from sys_config "
                "file! Must specify slot directly from command line with "
                "--slot argument. "
                f"Available slots are {self.sys['slot_order']}."
            )

        slot_cfg = self.sys['slots'][f'SLOT[{self.slot}]']
        self.stream_id = slot_cfg['stream_id']

        if dev_file is None:
            self.dev_file = os.path.abspath(os.path.expandvars(slot_cfg['device_config']))
        else:
            self.dev_file = os.path.abspath(os.path.expandvars(dev_file))
        self.dev = DeviceConfig.from_yaml(self.dev_file)

        # Gets the pysmurf config file
        if pysmurf_file is None:
            self.pysmurf_file = os.path.expandvars(slot_cfg['pysmurf_config'])
        else:
            self.pysmurf_file = pysmurf_file

        if uxm_file is None:
            if 'uxm_file' in slot_cfg:
                self.uxm_file = os.path.expandvars(slot_cfg['uxm_file'])
        else:
            self.uxm_file = uxm_file
        if self.uxm_file is not None:
            with open(self.uxm_file) as f:
                self.uxm = yaml.safe_load(f)
        else:
            self.uxm = None

    def dump_configs(self, output_dir=None, clobber=False, dump_rogue_tree=False):
        """
        Dumps any config information to an output directory

        Args:
            output_dir (path): Directory location to put config files.
            dump_rogue_truee : bool
                If True, will dump the pysmurf rogue tree to the configs dir.

        Returns:
            run_out (path): path to output run file
            sys_out (path): path to output sys file
            dev_out (path): path to output device file.
            uxm_out (path): path to uxm file
        """
        if output_dir is None:
            if not self.S:
                raise ValueError("output_dir cannot be None in offline mode")
            output_dir = os.path.join(
                    self.S.base_dir, self.S.date, self.stream_id, 
                    self.S.name, 'config', self.S.get_timestamp(),
                  )
        print(f"Dumping sodetlib configs to {output_dir}")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Manually set publisher action because set_action decorator won't
        # work without S being the first function argument.
        if self.S:
            start_action = self.S.pub._action
            start_action_ts = self.S.pub._action_ts

        try:
            outfiles = {}
            if self.S:
                self.S.pub._action = "config"
                self.S.pub._action_ts = self.S.get_timestamp()

            # Dump run info
            run_info = YamlReps.Odict([
                ('sys_file', self.sys_file),
                ('dev_file', self.dev_file),
                ('pysmurf_file', self.pysmurf_file),
                ('slot', self.slot),
                ('cwd', os.getcwd()),
                ('argv',  ' '.join(sys.argv)),
                ('time', int(time.time())),
            ])
            if self._argparse_args is not None:
                run_info['cfg_args'] = self._argparse_args
            run_out = os.path.abspath(os.path.join(output_dir, 'run_info.yml'))
            outfiles['run'] = run_out
            with open(run_out, 'w') as f:
                yaml.dump(run_info, f)
            if self.S:
                self.S.pub.register_file(run_out, 'config', format='yaml')

            # Dump sys file
            sys_out = os.path.abspath(os.path.join(output_dir,
                                                   'sys_config.yml'))
            outfiles['sys'] = sys_out
            with open(sys_out, 'w') as f:
                yaml.dump(self.sys, f)
            if self.S:
                self.S.pub.register_file(sys_out, 'config', format='yaml')

            # Dump device file
            dev_out = os.path.abspath(os.path.join(output_dir, 'dev_cfg.yml'))
            outfiles['dev'] = dev_out
            self.dev.dump(dev_out, clobber=clobber)
            if self.S:
                self.S.pub.register_file(dev_out, 'config', format='yaml')

            # Copy pysmurf file
            pysmurf_out = os.path.join(output_dir, 'pysmurf_cfg.yml')
            pysmurf_out = os.path.abspath(pysmurf_out)
            outfiles['pysmurf'] = pysmurf_out
            shutil.copy(self.pysmurf_file, pysmurf_out)
            if self.S:
                self.S.pub.register_file(pysmurf_out, 'config', format='yaml')

            # Dump uxm file
            if self.uxm is not None:
                uxm_out = os.path.abspath(os.path.join(output_dir, 'uxm.yml'))
                outfiles['uxm'] = uxm_out
                with open(uxm_out, 'w') as f:
                    yaml.dump(self.uxm, f)
                if self.S:
                    self.S.pub.register_file(uxm_out, 'config', format='yaml')

            if dump_rogue_tree:
                print("Dumping state")
                state_file = os.path.abspath(os.path.join(output_dir,
                                                          'rogue_state.yaml'))
                outfiles.append(state_file)
                self.S.save_state(state_file)
        finally:
            # Restore publisher action so it doesn't mess up ongoing actions
            if self.S:
                self.S.pub._action = start_action
                self.S.pub._action_ts = start_action_ts

        return outfiles

    def get_smurf_control(self, offline=False, epics_root=None,
                          smurfpub_id=None, make_logfile=False, setup=False,
                          dump_configs=None, config_dir=None,
                          apply_dev_configs=False, load_device_tune=True,
                          **pysmurf_kwargs):
        """
        Creates pysmurf instance based off of configuration parameters.
        If not specified as keyword arguments ``epics_root`` and ``smurf_pub``
        will be created based on the slot and crate id's.

        Args:
            offline (bool):
                Whether to start pysmurf in offline mode. Defaults to False
            epics_root (str, optional):
                Pysmurf epics root. If none, it will be set to
                ``smurf_server_s<slot>``.
            smurfpub_id (str, optional):
                Pysmurf publisher ID. If None, will default to
                crate<crate_id>_slot<slot>.
            make_logfile (bool):
                Whether pysmurf should write logs to a file (True) or
                stdout(False). Defaults to stdout.
            setup (bool):
                Whether pysmurf should run a full setup. Defaults to False.
            dump_configs (bool):
                Whether all configuration settings should be dumped to pysmurf
                data directory. Defaults to True.
            **pysmurf_kwargs:
                Any additional arguments to be passed to pysmurf initialization.

        """
        import pysmurf.client

        slot_cfg = self.sys['slots'][f'SLOT[{self.slot}]']
        if epics_root is None:
            epics_root = f'smurf_server_s{self.slot}'
        if smurfpub_id is None:
            smurfpub_id = self.stream_id
        if dump_configs is None:
            dump_configs = self.dump

        # Pysmurf publisher will check this to determine publisher id.
        os.environ['SMURFPUB_ID'] = smurfpub_id

        if offline:
            S = pysmurf.client.SmurfControl(offline=True)
        else:
            S = pysmurf.client.SmurfControl(
                epics_root=epics_root, cfg_file=self.pysmurf_file, setup=setup,
                make_logfile=make_logfile, data_path_id=smurfpub_id,
                **pysmurf_kwargs)
        self.S = S
        # Lets just stash this in pysmurf...
        S._sodetlib_cfg = self

        # Dump config outputs
        if dump_configs:
            if config_dir is None:
                if not offline:
                    config_dir = os.path.join(
                        S.base_dir, S.date, smurfpub_id, S.name, 'config',
                        S.get_timestamp(),
                    )

                else:
                    config_dir = os.path.join('config', S.get_timestamp())
                    print("Warning! Being run in offline mode with no config "
                          f"directory specified! Writing to {config_dir}")
            self.dump_configs(config_dir)

        if apply_dev_configs:
            print("Applying device cfg parameters...")
            self.dev.apply_to_pysmurf_instance(S, load_tune=load_device_tune)

        return S
