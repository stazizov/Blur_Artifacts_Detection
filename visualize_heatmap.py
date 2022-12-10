import cv2
import numpy as np
import matplotlib.pyplot as plt


image_path = "01.png"
heatmap_path = "heatmap_01.png"

image = cv2.imread(image_path)
heatmap = cv2.imread(heatmap_path)
# heatmap = cv2.resize(heatmap, image.shape[::-1][1:])

def visualize_heatmap(image, heatmap, threshold):
    # assert image.shape[:-1] == heatmap.shape[:-1]
    threshed_heatmap = heatmap.copy() / 255
    # threshed_heatmap[heatmap < 255*threshold] = 0.2
    # threshed_heatmap[heatmap >= 255*threshold] = 1.0
    threshed_heatmap = cv2.resize(
        threshed_heatmap, 
        image.shape[::-1][1:]
        )
    return (threshed_heatmap * image).astype(np.uint8)

if __name__ == "__main__":
    # processed_heatmap = visualize_heatmap(image, heatmap, 0.7)
    # plt.imshow(np.hstack([image, heatmap, processed_heatmap]))
    # plt.show()

    for threshold in range(7, 10):
        threshold /= 10
        cv2.imwrite(
            f"heatmaps_01/heatmap_thresh_{threshold}.png",
            visualize_heatmap(image, heatmap, threshold)
            )
        break