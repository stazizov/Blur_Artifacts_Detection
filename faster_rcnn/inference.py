# Some basic setup:
# Setup detectron2 logger
import detectron2
import sys
import os
import argparse

sys.path.insert(0, os.path.abspath('./detectron2'))


from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, cv2
import matplotlib.pyplot as plt

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path2image", type=str)
    parser.add_argument("--path2save", type=str)
    args = parser.parse_args()

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml"))
    cfg.MODEL.WEIGHTS = "model_final.pth"  # path to the model we just trained
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.001  # set a custom testing threshold
    cfg.SOLVER.BASE_LR = 0.00025  
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  
    
    predictor = DefaultPredictor(cfg)

    im = cv2.imread(args.path2image)
    outputs = predictor(im)
    v = Visualizer(
            im[:, :, ::-1], 
            )
    for box in outputs["instances"].pred_boxes.to('cpu'):
        v.draw_box(box)
        v.draw_text(str(box[:2].numpy()), tuple(box[:2].numpy()))
    v = v.get_output()
    img =  v.get_image()[:, :, ::-1]
    plt.imshow(img)
    cv2.imwrite(args.path2save)