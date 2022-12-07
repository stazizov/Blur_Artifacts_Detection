import pathlib
import numpy as np
import cv2
import os

def make_crops(path, xn, yn):
    image = cv2.imread(path)
    path = pathlib.Path(path)
    folder_path = str(path.with_suffix(''))+f"_{xn}_{yn}"
    os.mkdir(folder_path)

    crop_width = image.shape[1] // xn
    crop_height = image.shape[0] // yn
    for yni in range(yn-1):
        for xni in range(xn-1):
            crop = image[yni*crop_height:(yni+1)*crop_height, xni*crop_width:(xni+1)*crop_width]
            cv2.imwrite(os.path.join(folder_path, f"crop_{xni}_{yni}.png"), crop)

if __name__ == "__main__":
    make_crops("wss1/train/01.png", 9, 9)