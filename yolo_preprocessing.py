import cv2
import numpy as np
import albumentations as A
from math import ceil

image_filename = "WSS1/train/images/crop_1_2.png"
annotation_filename = "WSS1/train/labels/crop_1_2.xml"
image = cv2.imread(image_filename)
window_size = 640


preprocessing = A.Compose([A.augmentations.crops.transforms.RandomSizedCrop(
    [window_size, window_size], 
    window_size, window_size
    )
])

image = preprocessing(image=image)["image"]
cv2.imshow('image', image)
cv2.waitKey(0)

# print(image.shape)
# for y_index in range(ceil(image.shape[0] / window_size)):
#     y_top = y_index * window_size
#     y_bottom = (y_index + 1) * window_size
#     delta_y = image.shape[0] - y_bottom
#     if delta_y < 0:
#         y_top -= abs(delta_y)
#         y_bottom -= abs(delta_y)
        
#     for x_index in range(ceil(image.shape[1] / window_size)):
#         x_left = x_index * window_size
#         x_right = (x_index + 1) * window_size
#         delta_x = image.shape[1] - x_right
#         if delta_x < 0:
#             x_left -= abs(delta_x)
#             x_right -= abs(delta_x)
#         print(y_top, y_bottom, x_left, x_right)
