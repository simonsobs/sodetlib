import argparse
import os
import yaml
from collections import OrderedDict
import sys
import time


yaml.add_representer(
    OrderedDict,
    lambda self, data: self.represent_mapping('tag:yaml.org,2002:map',
                                              data.items())
)


class DeviceConfig:
    def __init__(self):
        # self.bands = [BandCfg(f'Band {i}') for i in range(8)]
        self.bands = [{} for _ in range(8)]
        self.bias_groups = [{} for _ in range(12)]
        self.exp = {}

    @classmethod
    def from_yaml(cls, filename):
        """Creates a DeviceConfig object from a yaml file"""
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
        data = OrderedDict()
        data['experiment'] = self.exp
        data['bias_groups'] = {
            k: [bg[k] for bg in self.bias_groups]
            for k in self.bias_groups[0].keys()
        }
        data['bands'] = OrderedDict()
        for i, band in enumerate(self.bands):
            data['bands'][f'Band[{i}]'] = self.bands[i]
        with open(path, 'w') as f:
            yaml.dump(data, f)

    @classmethod
    def from_dict(cls, data):
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
    def __init__(self):
        self.sys_file = None
        self.dev_file = None
        self.pysmurf_file = None
        self.sys = None
        self.dev = None
        self.dump = None  # Whether config files should be dumped

    def parse_args(self, parser=None, args=None):
        if parser is None:
            parser = argparse.ArgumentParser()

        parser.add_argument_group("DetConfig Options")
        parser.add_argument('--sys-file', help="Path to sys-config file")
        parser.add_argument('--dev-file', help="Path to device-config file")
        parser.add_argument('--pysmurf-file', help="Path to Pysmurf config file")
        parser.add_argument('--slot', '-N', type=int, help="Smurf slot")
        parser.add_argument('--dump-configs', '-D', action='store_true',
                            help="If true, all config info will be written to "
                                 "the pysmurf output directory")
        args = parser.parse_args()
        self.load_config_files(args.slot, args.sys_file, args.dev_file,
                               args.pysmurf_file)


    def load_config_files(self, slot=None, sys_file=None, dev_file=None, pysmurf_file=None):
        """
        Loads configuration files. If arguments are not specified, sensible
        defatuls will be chosen.

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
        elif len(self.sys['smurf_slots']) == 1:
            self.slot = self.sys['smurf_slots'][0]
        else:
            raise argparse.ArgumentError("Multiple smurf slots exist! "
                                         "Must specify --slot from command line")

        if dev_file is None:
            self.dev_file = os.path.expandvars(
                self.sys['device_configs'][self.slot - 2])
        else:
            self.dev_file = dev_file
        self.dev = DeviceConfig.from_yaml(self.dev_file)

        # Gets the pysmurf config file
        if pysmurf_file is None:
            self.pysmurf_file = os.path.expandvars(
                self.sys['pysmurf_configs'][self.slot - 2])
        else:
            self.pysmurf_file = pysmurf_file

    def dump_configs(self, output_dir):
        """
        Dumps any config information to an output directory

        Args:
            output_dir (path): Directory location to put config files.
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        run_info = OrderedDict([
            ('sys_file', self.sys_file),
            ('dev_file', self.dev_file),
            ('pysmurf_file', self.pysmurf_file),
            ('slot', self.slot),
            ('cwd', os.getcwd()),
            ('args',  ' '.join(sys.argv)),
            ('time', int(time.time())),
        ])
        with open(os.path.join(output_dir, 'run_info.yml'), 'w') as f:
            yaml.dump(run_info, f)
        with open(os.path.join(output_dir, 'sys_config.yml'), 'w') as f:
            yaml.dump(self.sys, f)
        self.dev.dump(os.path.join(output_dir, 'dev_cfg.yml'))

    def make_pysmurf_instance(self, offline=False, epics_root=None, smurfpub_id=None,
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
                    print("Warning! Being run in offline mode with no config"
                          f"directory specified! Writing to {config_dir}")
            self.dump_configs(config_dir)

        return S

if __name__ == "__main__":
    cfg = DetConfig()

    cfg.dump_configs('dumpdir')

