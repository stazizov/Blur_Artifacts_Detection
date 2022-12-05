import cv2
import numpy as np
import albumentations as A
from math import ceil
import os

# image_filename = "WSS1/train/images/crop_1_2.png"
# annotation_filename = "WSS1/train/labels/crop_1_2.xml"
# image = cv2.imread(image_filename)
# window_size = 640


# preprocessing = A.Compose([A.augmentations.crops.transforms.RandomSizedCrop(
#     [window_size, window_size], 
#     window_size, window_size
#     )
# ])

# image = preprocessing(image=image)["image"]
# cv2.imshow('image', image)
# cv2.waitKey(0)

for images_dir in ["train", "test"]:
    images_dir = f"WSS1_640_640/{images_dir}/images"
    for filename in os.listdir(images_dir):
        if filename.startswith("."):
            continue
        filepath = os.path.join(images_dir, filename)
        image = cv2.imread(filepath)
        image = cv2.resize(image, (640, 640))
        cv2.imwrite(filepath, image)