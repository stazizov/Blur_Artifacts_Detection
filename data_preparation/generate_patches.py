from sliding_window import sliding_window
import cv2
import os 

def sliding_window(image, stepSize, windowSize):
  for y in range(0, image.shape[0], stepSize):
    for x in range(0, image.shape[1], stepSize):
      yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])
      
if __name__ == '__main__':

    image = cv2.imread("01.png")
    window_size = 128
    for x, y, patch in sliding_window(image, window_size, (window_size, window_size)):
        cv2.imwrite(os.path.join(f"patches_{window_size}", f"patch_{x//window_size}_{y // window_size}.png"), patch)