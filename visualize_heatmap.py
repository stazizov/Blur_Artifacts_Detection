import cv2
import numpy as np

def visualize_heatmap(image, heatmap, threshold):
    threshed_heatmap = heatmap.copy() / 255
    threshed_heatmap = cv2.resize(
        threshed_heatmap, 
        image.shape[::-1][1:]
        )
    return (threshed_heatmap * image).astype(np.uint8)