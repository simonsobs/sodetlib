import argparse
import os
import yaml
from collections import OrderedDict
import sys
import time


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
        self.bands = [{} for _ in range(8)]
        self.bias_groups = [{} for _ in range(12)]
        self.exp = {}

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
                self.bands[band_index] = v

        # Loads bias groups from config file
        for i, bg in enumerate(self.bias_groups):
            for k, vlist in data['bias_groups'].items():
                self.bias_groups[i][k] = vlist[i]

        # Loads experiment config data
        self.exp = data['experiment']
        return self

    @classmethod
    def from_yaml(cls, filename):
        """
        Creates a DeviceConfig object from a dev-cfg yaml file.

        Args:
            filename (path): path to device-config file.
        """
        filename = os.path.abspath(os.path.expandvars(filename))
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
            raise FileExistsError(f"Can't dump device config! Path {path} already "
                                  "exists!!")
        data = YamlReps.Odict()
        data['experiment'] = self.exp
        data['bias_groups'] = {
            k: YamlReps.FlowSeq([bg[k] for bg in self.bias_groups])
            for k in self.bias_groups[0].keys()
        }
        data['bands'] = YamlReps.Odict([
            (f'AMC[{i}]', {
                k: YamlReps.FlowSeq([b[k] for b in self.bands[4*i:4*i+4]])
                for k in self.bands[0].keys()
            }) for i in [0, 1]])
        with open(path, 'w') as f:
            yaml.dump(data, f)

    def update_band(self, band_index, data):
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
                raise ValueError(
                    f"{k} is not a valid key for band {band_index}. "
                    f"Check dev-cfg file for available keys.")
            band[k] = v

    def update_bias_group(self, bg_index, data):
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
                raise ValueError(f"{k} is not a valid bias_group key. "
                                 f"Check dev-cfg file for available keys.")
            bg[k] = v

    def update_experiment(self, data):
        """
        Updates ``experiment`` configuration object.

        Args:
            data (dict):
                Dictionary of parameters to update. All parameters must exist in
                the loaded dev-cfg file.
        """
        for k, v in data.items():
            if k not in self.exp.keys():
                raise ValueError(f"{k} is not a valid experiment key. "
                                 f"Check dev-cfg file for available keys")
            self.exp[k] = v


class DetConfig:
    """
    General Configuration class for SODETLIB.

    Attributes:
        sys_file (path): Path to sys-config file used.
        dev_file (path): Path to device-config file used
        pysmurf_file (path): Path to pysmurf-config file used.
        sys (dict):
            System configuration dictionary, generated from the sys_config.yml
            file or ``--sys-file`` command line argument.
        dev (dict):
            Device configuration dictionary, generated from the device_config
            file specified in the sys_config or specified by the ``--dev-file``
            command line argument
    """
    def __init__(self, sys_file=None, dev_file=None, pysmurf_file=None):
        self.sys_file = sys_file
        self.dev_file = dev_file
        self.pysmurf_file = pysmurf_file
        self.sys = None
        self.dev: DeviceConfig = None
        self.dump = None  # Whether config files should be dumped
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
                If specified, argparse will read arguments from this list
                instead of the command line.
        """
        self._argparse_args = args

        if parser is None:
            parser = argparse.ArgumentParser()

        parser.add_argument_group("DetConfig Options")
        parser.add_argument('--sys-file', help="Path to sys-config file",
                            default=self.sys_file)
        parser.add_argument('--dev-file', help="Path to device-config file",
                            default=self.dev_file)
        parser.add_argument('--pysmurf-file', help="Path to Pysmurf config file",
                            default=self.pysmurf_file)
        parser.add_argument('--slot', '-N', type=int, help="Smurf slot")
        parser.add_argument('--dump-configs', '-D', action='store_true',
                            help="If true, all config info will be written to "
                                 "the pysmurf output directory")
        self._parser = parser
        args = parser.parse_args(args=args)
        self.load_config_files(args.slot, args.sys_file, args.dev_file,
                               args.pysmurf_file)

        return args

    def load_config_files(self, slot=None, sys_file=None, dev_file=None, pysmurf_file=None):
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
                $OCS_CONFIG_DIR/sys_config.yml
            dev_file (path, optional):
                Path to the Device file. If None, defaults to the device file
                specified in the sys-config file (device_configs[slot-2]).
            pysmurf_file (path, optional):
                Path to pysmurf config file. If None, defaults to the file
                specified in the sys-config (pysmurf_configs[slot-2]).
        """
        if sys_file is None:
            self.sys_file = os.path.expandvars('$OCS_CONFIG_DIR/sys_config.yml')
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
        if dev_file is None:
            self.dev_file = os.path.expandvars(slot_cfg['device_config'])
        else:
            self.dev_file = dev_file
        self.dev = DeviceConfig.from_yaml(self.dev_file)

        # Gets the pysmurf config file
        if pysmurf_file is None:
            self.pysmurf_file = os.path.expandvars(slot_cfg['pysmurf_config'])
        else:
            self.pysmurf_file = pysmurf_file

    def dump_configs(self, output_dir, clobber=False):
        """
        Dumps any config information to an output directory

        Args:
            output_dir (path): Directory location to put config files.

        Returns:
            run_out (path): path to output run file
            sys_out (path): path to output sys file
            dev_out (path): path to output device file.
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

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
        with open(run_out, 'w') as f:
            yaml.dump(run_info, f)
        sys_out = os.path.abspath(os.path.join(output_dir, 'sys_config.yml'))
        with open(sys_out, 'w') as f:
            yaml.dump(self.sys, f)
        dev_out = os.path.abspath(os.path.join(output_dir, 'dev_cfg.yml'))
        self.dev.dump(dev_out, clobber=clobber)
        return run_out, sys_out, dev_out

    def get_smurf_control(self, offline=False, epics_root=None, smurfpub_id=None,
                              make_logfile=False, setup=False, dump_configs=None,
                              config_dir=None, **pysmurf_kwargs):
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

        if epics_root is None:
            epics_root = f'smurf_server_s{self.slot}'
        if smurfpub_id is None:
            smurfpub_id = f'crate{self.sys["crate_id"]}_slot{self.slot}'
        if dump_configs is None:
            dump_configs = self.dump

        # Pysmurf publisher will check this to determine publisher id.
        os.environ['SMURFPUB_ID'] = smurfpub_id

        if offline:
            S = pysmurf.client.SmurfControl(offline=True)
        else:
            S = pysmurf.client.SmurfControl(
                epics_root=epics_root, cfg_file=self.pysmurf_file, setup=setup,
                make_logfile=make_logfile, **pysmurf_kwargs)

        # Dump config outputs
        if dump_configs:
            if config_dir is None:
                if not offline:
                    config_dir = os.path.join(S.data_dir, f'config',
                                              S.get_timestamp())
                else:
                    config_dir = os.path.join('config', S.get_timestamp())
                    print("Warning! Being run in offline mode with no config "
                          f"directory specified! Writing to {config_dir}")
            self.dump_configs(config_dir)

        return S

