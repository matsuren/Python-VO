import os

import numpy as np
import cv2
import argparse
import yaml
import logging
import time

from utils.tools import plot_keypoints

from DataLoader import create_dataloader
from Detectors import create_detector
from Matchers import create_matcher
from VO.VisualOdometry import VisualOdometry, AbosluteScaleComputer

result_folder = "results_tmp"

def keypoints_plot(img, vo):
    if img.shape[2] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return plot_keypoints(img, vo.kptdescs["cur"]["keypoints"], vo.kptdescs["cur"]["scores"]/vo.kptdescs["cur"]["scores"].max())


class TrajPlotter(object):
    def __init__(self):
        self.errors = []
        self.traj = np.zeros((600, 600, 3), dtype=np.uint8)
        pass

    def update(self, est_xyz, gt_xyz, fps, num_inliers):
        x, z = est_xyz[0], est_xyz[2]
        gt_x, gt_z = gt_xyz[0], gt_xyz[2]

        est = np.array([x, z]).reshape(2)
        gt = np.array([gt_x, gt_z]).reshape(2)

        error = np.linalg.norm(est - gt)

        self.errors.append(error)

        avg_error = np.mean(np.array(self.errors))

        # === drawer ==================================
        # each point
        draw_x, draw_y = int(x) + 290, int(z) + 90
        true_x, true_y = int(gt_x) + 290, int(gt_z) + 90

        # draw trajectory
        cv2.circle(self.traj, (true_x, true_y), 1, (0, 0, 255), 2)
        cv2.circle(self.traj, (draw_x, draw_y), 1, (0, 255, 0), 1)
        cv2.rectangle(self.traj, (10, 20), (600, 80), (0, 0, 0), -1)

        # draw text
        num_in = num_inliers[-1]
        num_in_avg = sum(num_inliers) / len(num_inliers)
        text = "[AvgError] %2.4fm [FPS] %3.2f [inliers/avg] %d/%3.1f" % (avg_error, fps, num_in, num_in_avg)
        cv2.putText(self.traj, text, (20, 40),
                    cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, 8)

        return self.traj


def run(args):
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.Loader)

    # create dataloader
    loader = create_dataloader(config["dataset"])
    # create detector
    detector = create_detector(config["detector"])
    # create matcher
    matcher = create_matcher(config["matcher"])

    absscale = AbosluteScaleComputer()
    traj_plotter = TrajPlotter()

    # log
    if not os.path.exists(result_folder):
        os.mkdir(result_folder)
    fname = args.config.split('/')[-1].split('.')[0]
    log_fopen = open(f"{result_folder}/" + fname + ".txt", mode='a')

    vo = VisualOdometry(detector, matcher, loader.cam)

    start_time = time.time()
    num_inliers = []
    for i, img in enumerate(loader):
        gt_pose = loader.get_cur_pose()
        R, t, num_in = vo.update(img, absscale.update(gt_pose))
        num_inliers.append(num_in)

        # === log writer ==============================
        print(i, t[0, 0], t[1, 0], t[2, 0], gt_pose[0, 3], gt_pose[1, 3], gt_pose[2, 3], file=log_fopen)


        # FPS
        elapsed = time.time() - start_time
        fps = (i+1)/elapsed
        # === drawer ==================================
        img1 = keypoints_plot(img, vo)
        img2 = traj_plotter.update(t, gt_pose[:, 3], fps=fps, num_inliers=num_inliers)

        cv2.imshow("keypoints", img1)
        cv2.imshow("trajectory", img2)
        if cv2.waitKey(10) == 27:
            break
    cv2.imwrite(f"{result_folder}/" + fname + '.png', img2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='python_vo')
    parser.add_argument('--config', type=str, default='params/kitti_orb_brutematch.yaml',
                        help='config file')
    parser.add_argument('--logging', type=str, default='INFO',
                        help='logging level: NOTSET, DEBUG, INFO, WARNING, ERROR, CRITICAL')

    args = parser.parse_args()

    logging.basicConfig(level=logging._nameToLevel[args.logging])

    run(args)
