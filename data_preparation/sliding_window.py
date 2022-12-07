import cv2
from metrics import HistologicalTransforms, LaplacianBlurDetector
import matplotlib.pyplot as plt
import numpy as np
import tqdm
import os

laplacian = LaplacianBlurDetector()
transforms = HistologicalTransforms()

def sliding_window(image, stepSize, windowSize):
  for y in range(0, image.shape[0], stepSize):
    for x in range(0, image.shape[1], stepSize):
      yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

if __name__ == "__main__":
    windowSize = (100, 100)
    stepSize = windowSize[0]
    filename = "S1-v2/test/s1_test_01.png"
    image = cv2.imread(filename) 
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    heatmap = np.ones_like(image, dtype=np.float32)
    windows = sliding_window(image, stepSize, windowSize)
    # os.mkdir(os.path.join("data", os.path.split(filename)[-1]))
    for (x, y, window) in tqdm.tqdm(windows):
        cpbd_metric = transforms(window)
        cv2.imwrite(
          os.path.join("data", os.path.split(filename)[-1], 
          f"{y}_{x}_{os.path.split(filename)[-1]}"),
          window*255
          )
        heatmap_window = np.ones_like(window, dtype=np.float32) * cpbd_metric
        heatmap[y:y+windowSize[0], x:x+windowSize[1]] = heatmap_window
    cv2.imwrite("heatmaps/s1_test_01_heatmap.png", heatmap*255)
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.subplot(1, 2, 2)
    plt.imshow(heatmap)
    plt.show()