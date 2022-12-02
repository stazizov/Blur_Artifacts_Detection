import os

from metrics import compute
from matplotlib.pyplot import imread
import matplotlib.pyplot as plt
import numpy as np
import cv2

if __name__ == "__main__":
    for filename in os.listdir("test_images"):
        if filename.startswith("."):
            continue
        image = cv2.imread(os.path.join("test_images", filename))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        noise = np.random.rand(*image.shape)
        print(compute(image))
        image = image * noise
        print(compute(image))
        plt.imshow(image)
        plt.show()
        break

