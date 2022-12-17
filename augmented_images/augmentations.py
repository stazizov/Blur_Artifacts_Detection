import numpy as np
import cv2
import albumentations as A
from typing import Tuple
import pprintjson
import matplotlib.pyplot as plt
import json 

class Augmentations:
    def __init__(self):
        self.blurs = [
            A.GaussianBlur(blur_limit=(1,3)),
            A.GaussianBlur(blur_limit=(5,7)),
            A.GaussianBlur(blur_limit=(9, 11)),
            A.GaussianBlur(blur_limit=(15, 17)),
            A.MedianBlur(blur_limit=(1,3)),
            A.MedianBlur(blur_limit=(5,7)),
            A.MedianBlur(blur_limit=(9, 11)),
            A.MedianBlur(blur_limit=(15, 17)),
            A.JpegCompression(),
        ]

    def get_augmentations(self, blur):
        
        return A.ReplayCompose([
            blur,
            A.CLAHE(),
            A.RandomBrightness(),
            A.RandomBrightnessContrast(),
            A.RandomGamma(),
        ])
    
    def __call__(self, image_path:str) -> None:
        image = cv2.imread(image_path)
        for index, blur in enumerate(self.blurs):
            dst_image_path = f"exp_{index}_{image_path}"
            # we use only .png format in our experiments 
            dst_annotations_path = dst_image_path.replace('.png', '.json')
            augmentations = self.get_augmentations(blur)
            data = augmentations(image=image)

            pprintjson.pprintjson(data["replay"])
            cv2.imwrite(dst_image_path, data["image"])
            with open(dst_annotations_path, 'w') as outfile:
                json.dump(data["replay"], outfile)
            # A.save(data["replay"], dst_annotations_path, data_format='yaml')


if __name__ == "__main__":
    num_exp = 1
    src_path = "positive.png"
    augmentator = Augmentations()
    augmentator(src_path)