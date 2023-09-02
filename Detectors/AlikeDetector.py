from pathlib import Path
import logging
import sys
import os
sys.path.append(os.path.join(os.path.split(__file__)[0], "ALIKE"))
sys.path.append(os.path.join(os.path.split(__file__)[0], "../"))
from utils.tools import *
from alike import ALike, configs

class AlikeDetector(object):
    default_config = {
        "model": "alike-t",
        "scores_th": 0.2,
        "n_limit": 5000,
        "cuda": True
    }

    def __init__(self, config={}):
        self.config = self.default_config
        self.config = {**self.config, **config}
        logging.info("Alike detector config: ")
        logging.info(self.config)

        self.device = 'cuda' if torch.cuda.is_available() and self.config["cuda"] else 'cpu'

        logging.info("creating Alike detector...")
        self.model = ALike(**configs[self.config["model"]],
                device=self.device,
                top_k=-1,
                scores_th=self.config["scores_th"],
                n_limit=self.config["n_limit"])


    def __call__(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        logging.debug("detecting keypoints with ALIKE...")
        pred = self.model(image, sub_pixel=True)

        ret_dict = {
            "image_size": np.array([image.shape[0], image.shape[1]]),
            "torch": pred,
            "keypoints": pred["keypoints"],
            "scores": pred["scores"],
            "descriptors": pred["descriptors"]
        }

        return ret_dict


if __name__ == "__main__":
    img = cv2.imread("../test_imgs/sequences/00/image_0/000000.png")

    detector = AlikeDetector()
    kptdescs = detector(img)

    img = plot_keypoints(img, kptdescs["keypoints"], kptdescs["scores"])
    cv2.imshow("AlikeDetector", img)
    cv2.waitKey()
