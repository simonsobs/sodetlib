import matplotlib
matplotlib.use('Agg')
import argparse
from sodetlib.det_config import DetConfig

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('slot', help='Slot number')
    parser.add_argument('out_file', help='output_file')
    args = parser.parse_args()

    cfg = DetConfig()
    cfg.parse_args(args=['-N', args.slot])
    S = cfg.get_smurf_control(setup=False, dump_configs=False)
    print(f"Acquired smurf control for slot {args.slot}")

    S.set_read_all()
    S.save_state(args.out_file)
    
