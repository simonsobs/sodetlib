import pysmurf.client
import argparse

if __name__=='__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--setup', action='store_true')
    parser.add_argument('--config-file', required=True)
    parser.add_argument('--epics-root', default='smurf_server_s2')

    args = parser.parse_args()
    dwell_time = args.dwell_time

    S = pysmurf.client.SmurfControl(
            epics_root = args.epics_root,
            cfg_file = args.config_file,
            setup = args.setup,make_logfile=False
    )

    S.stream_data_off()
