import argparse
import os
import sys

LOCAL_PATH = './'
if LOCAL_PATH not in sys.path:
    sys.path.append(LOCAL_PATH)

from associate import associate, read_file_list
from tqdm import tqdm


def associate_two_files(fname1, fname2, outname, filehead):
    fname1_list = read_file_list(fname1)
    fname2_list = read_file_list(fname2)

    fname1_fname2_matches = associate(fname1_list, fname2_list, float(args.offset), float(args.max_difference))

    with open(outname, 'w') as f:
        f.write(filehead)
        for a, b in fname1_fname2_matches:
            f.write("%f %s %s\n" % (a, " ".join(fname1_list[a]), " ".join(fname2_list[b])))


if __name__ == '__main__':
    # parse command line
    parser = argparse.ArgumentParser(
        description='''This script associate the rgb.txt, depth.txt, groundtruth.txt for all tumrgbd data sequences''')
    parser.add_argument('--root_path', help='first text file (format: timestamp data)', default="/Users/komatsu/data/tum_slam")
    parser.add_argument('--offset', help='time offset added to the timestamps of the second file (default: 0.0)',
                        default=0.0)
    parser.add_argument('--max_difference',
                        help='maximally allowed time difference for matching entries (default: 0.02)', default=0.02)
    args = parser.parse_args()

    for sequence_name in tqdm(os.listdir(args.root_path)):
        if sequence_name[0] == ".":
            continue
        rgbd = os.path.join(args.root_path, sequence_name, 'rgbd.txt')
        if not os.path.exists(rgbd):
            rgb = os.path.join(args.root_path, sequence_name, 'rgb.txt')
            depth = os.path.join(args.root_path, sequence_name, 'depth.txt')
            associate_two_files(rgb, depth, rgbd, '# rgbd\n'
                                                  '# timestamp rgb_filename depth_filename\n')

        rgbd_gt = os.path.join(args.root_path, sequence_name, 'rgbd_gt.txt')
        if not os.path.exists(rgbd_gt):
            gt = os.path.join(args.root_path, sequence_name, 'groundtruth.txt')
            associate_two_files(rgbd, gt, rgbd_gt, '# rgbd with groundtruth\n'
                                                   '# timestamp rgb_filename depth_filename tx ty tz qx qy qz qw\n')
