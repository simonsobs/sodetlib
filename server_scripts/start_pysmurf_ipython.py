from IPython import embed
import traitlets.config
import argparse
import os
import yaml

if __name__ == "__main__":
    sys_config_file = os.path.join(os.environ['OCS_CONFIG_DIR'], 'sys_config.yml')
    with open(sys_config_file, 'r') as stream:
        sys_config = yaml.safe_load(stream)

    parser = argparse.ArgumentParser()

    parser.add_argument('--slot', '-N', type=int)
    parser.add_argument('--config-file', '-c', default=sys_config['pysmurf_config_file'])
    parser.add_argument('--setup', action='store_true')
    parser.add_argument('--agg', action='store_true')

    args = parser.parse_args()

    if args.agg:
        import matplotlib
        matplotlib.use('Agg')
    import pysmurf.client

    epics_root = f'smurf_server_s{args.slot}'

    # Sets pysmurf publisher id
    crate = sys_config['crate_id']
    os.environ['SMURFPUB_ID'] = f'crate{crate}_slot{args.slot}'

    print(f"Creating pysmurf object for slot {args.slot}")
    S = pysmurf.client.SmurfControl(
        epics_root=epics_root,
        cfg_file=args.config_file,
        setup=args.setup
    )

    _ipython_config = traitlets.config.get_config()
    _ipython_config.InteractiveShellEmbed.colors = "LightBG"
    embed(config=_ipython_config)


