import matplotlib
matplotlib.use('Agg')
import pysmurf.client
import argparse
import subprocess
import os
import yaml

if __name__ == "__main__":
    sys_config_file = os.path.join(os.environ['OCS_CONFIG_DIR'], 'sys_config.yml')
    with open(sys_config_file, 'r') as stream:
        sys_config = yaml.safe_load(stream)

    parser = argparse.ArgumentParser()

    parser.add_argument('--slot', '-N')
    parser.add_argument('--config-file', '-c', default=sys_config['pysmurf_config_file'])
    parser.add_argument('--shelf-manager', '-S', default=sys_config['shelf_manager'])

    args = parser.parse_args()

    epics_root = f'smurf_server_s{args.slot}'

    S = pysmurf.client.SmurfControl(
        epics_root=epics_root, cfg_file=args.config_file,
        setup=True, make_logfile=False
    )

    S.set_stream_enable(0)
