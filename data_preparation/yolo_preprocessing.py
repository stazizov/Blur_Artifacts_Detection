import tensorflow as tf
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm
import cv2
import os
from random import randint, seed
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class config:
    bbox_max_height = 60
    bbox_max_width = 60


class MaskGenerator():

    def __init__(self, height, width, channels=3, rand_seed=None, filepath=None):
        """Convenience functions for generating masks to be used for inpainting training
        
        Arguments:
            height {int} -- Mask height
            width {width} -- Mask width
        
        Keyword Arguments:
            channels {int} -- Channels to output (default: {3})
            rand_seed {[type]} -- Random seed (default: {None})
            filepath {[type]} -- Load masks from filepath. If None, generate masks with OpenCV (default: {None})
        """

        self.height = height
        self.width = width
        self.channels = channels
        self.filepath = filepath

        # If filepath supplied, load the list of masks within the directory
        self.mask_files = []
        if self.filepath:
            filenames = [f for f in os.listdir(self.filepath)]
            self.mask_files = [f for f in filenames if any(filetype in f.lower() for filetype in ['.jpeg', '.png', '.jpg'])]
            print(">> Found {} masks in {}".format(len(self.mask_files), self.filepath))        

        # Seed for reproducibility
        if rand_seed:
            seed(rand_seed)

    def _generate_mask(self):
        """Generates a random irregular mask with lines, circles and elipses"""

        img = np.zeros((self.height, self.width, self.channels), np.uint8)

        # Set size scale
        size = int((self.width + self.height) * 0.03)
        if self.width < 64 or self.height < 64:
            raise Exception("Width and Height of mask must be at least 64!")
        
        # Draw random lines
        for _ in range(randint(1, 20)):
            x1, x2 = randint(1, self.width), randint(1, self.width)
            y1, y2 = randint(1, self.height), randint(1, self.height)
            thickness = randint(3, size)
            cv2.line(img,(x1,y1),(x2,y2),(1,1,1),thickness)
            
        # Draw random circles
        for _ in range(randint(1, 20)):
            x1, y1 = randint(1, self.width), randint(1, self.height)
            radius = randint(3, size)
            cv2.circle(img,(x1,y1),radius,(1,1,1), -1)
            
        # Draw random ellipses
        for _ in range(randint(1, 20)):
            x1, y1 = randint(1, self.width), randint(1, self.height)
            s1, s2 = randint(1, self.width), randint(1, self.height)
            a1, a2, a3 = randint(3, 180), randint(3, 180), randint(3, 180)
            thickness = randint(3, size)
            cv2.ellipse(img, (x1,y1), (s1,s2), a1, a2, a3,(1,1,1), thickness)
        
        return img

    def _load_mask(self, rotation=True, dilation=True, cropping=True):
        """Loads a mask from disk, and optionally augments it"""

        # Read image
        mask = cv2.imread(os.path.join(self.filepath, np.random.choice(self.mask_files, 1, replace=False)[0]))
        
        # Random rotation
        if rotation:
            rand = np.random.randint(-180, 180)
            M = cv2.getRotationMatrix2D((mask.shape[1]/2, mask.shape[0]/2), rand, 1.5)
            mask = cv2.warpAffine(mask, M, (mask.shape[1], mask.shape[0]))
            
        # Random dilation
        if dilation:
            rand = np.random.randint(5, 47)
            kernel = np.ones((rand, rand), np.uint8) 
            mask = cv2.erode(mask, kernel, iterations=1)
            
        # Random cropping
        if cropping:
            x = np.random.randint(0, mask.shape[1] - self.width)
            y = np.random.randint(0, mask.shape[0] - self.height)
            mask = mask[y:y+self.height, x:x+self.width]

        return (mask > 1).astype(np.uint8)

    def __call__(self, image_width, image_height):
        """Retrieve a random mask"""
        return self._generate_mask()

class BBoxesGenerator:
    def __init__(self, bbox_height, bbox_width, random_ratio_thresh=0.5):
        """Generates random bboxes 

        Args:
            bbox_height (int): max height of generated bbox
            bbox_width (int): max width of generated bbox
            random_ratio_thresh (float): minimum threshold for generating random number for box augmentation
        """
        self.bbox_height = bbox_height
        self.bbox_width = bbox_width
        self.random_ratio = random_ratio_thresh

    def __call__(self, image_height, image_width, num_bboxes) -> List[List[int]]:
        """return batch of coordinates 

        Args:
            image_height (int): height of an image 
            image_width (int): width of an image
            num_bboxes (int): number of bboxes generated per call

        Returns:
            list[list[int]]: batch of coordinates [[y_min, x_min, y_max, x_max]]
        """
        random_x_ratio = np.random.uniform(self.random_ratio, 1, 1)
        random_y_ratio = np.random.uniform(self.random_ratio, 1, 1)
    
        bbox_width = int(self.bbox_width * random_x_ratio)
        bbox_height = int(self.bbox_height * random_y_ratio)

        random_max_coords = [
                (randint(bbox_height, image_height),
                randint(bbox_width, image_width))
                for _ in range(num_bboxes)
            ]

        return [
            (
                coord[0] - bbox_height, 
                coord[1] - bbox_width, 
                coord[0],
                coord[1]
            ) for coord in random_max_coords
        ]

class BlurAugmentation:
    def __init__(
        self, 
        bbox_height, 
        bbox_width, 
        random_ratio_thresh,
        max_num_bboxes = 20
        ) -> None:
        """Applying randomly generated blur masks to clippings of randomly generated image bboxes

        Args:
            bbox_height (int): max height of bbox 
            bbox_width (int): max width of bbox
            random_ratio_thresh (float): minimum threshold for generating random number for box augmentation
            max_num_bboxes (int, optional): max number of generated bboxes per image. Defaults to 20.
        """
        # for generating bboxes
        self.bbox_generator = BBoxesGenerator(
            bbox_height, 
            bbox_width, 
            random_ratio_thresh
            )

        # to generate random binary masks 
        self.mask_generator = MaskGenerator(
            bbox_height, 
            bbox_width,
        )

        # max number of generated bboxes per image
        self.max_num_bboxes = max_num_bboxes

    def __call__(self, image) -> Tuple[np.ndarray, List[List[int]]]:
        # generate random number of bboxes in range of 0-self.max_num_bboxes
        bboxes = self.bbox_generator(
            image.shape[0],
            image.shape[1], 
            np.random.randint(self.max_num_bboxes)
        )
        for bbox in bboxes:
            bbox_crop = image[bbox[0]:bbox[2], bbox[1]:bbox[3]]
            mask = cv2.resize(
                self.mask_generator(
                    *bbox_crop.shape[:2]
                ), 
                bbox_crop.shape[:2][::-1]
                ) * 255
            blurred_img = cv2.GaussianBlur(bbox_crop, (21, 21), 0)
            image[bbox[0]:bbox[2], bbox[1]:bbox[3]] = np.where(mask==np.array([255, 255, 255]), bbox_crop, blurred_img)
            # cv2.rectangle(image, (bbox[1], bbox[0]), (bbox[3], bbox[2]), (0, 255, 0), 2)
        return (image, bboxes)

if __name__ == "__main__":
    src_dir = "WSS1/train/images"
    dst_dir = "dataset"
    annotations_path = os.path.join(dst_dir, "annotations.csv")
    num_copies = 10
    image_size = 960
    aug = BlurAugmentation(
                80, 80, 0.5
            )

    filenames = []
    x_mins = []
    y_mins = []
    x_maxs = []
    y_maxs = []

    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)
    for filename in os.listdir(src_dir):
        if filename.startswith("."):
            continue
        src_path = os.path.join(src_dir, filename)
        image = cv2.imread(src_path)
        image =  cv2.resize(image, (image_size, image_size))
        for index in range(num_copies):
            dst_path = os.path.join(dst_dir, f"{filename[:-4]}_{index}.png")
            image, bboxes = aug(image)
            cv2.imwrite(dst_path, image)
            for (y_min, x_min, y_max, x_max) in bboxes:
                filenames.append(src_path)
                x_mins.append(x_min)
                x_maxs.append(x_max)
                y_mins.append(y_min)
                y_maxs.append(y_max)
    
    # save annotations to csv
    pd.DataFrame(dict(
        filename = filenames, 
        x_min = x_mins, 
        y_min = y_mins,
        x_max = x_maxs,
        y_max = y_maxs,
        name = [0]*len(filenames)
    )).to(annotations_path)