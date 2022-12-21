import cv2
import os 

def sliding_window(image, stepSize, windowSize):
  for y in range(0, image.shape[0], stepSize):
    for x in range(0, image.shape[1], stepSize):
      yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])
      
if __name__ == '__main__':

    image = cv2.imread("05.png")
    window_size = 448

    if not os.path.exists(f"patches_{window_size}"):
      os.mkdir(f"patches_{window_size}")
    for x, y, patch in sliding_window(image, window_size, (window_size, window_size)):
        dst_path = os.path.join(
          f"patches_{window_size}", 
          f"patch_{x//window_size}_{y // window_size}.png"
          )
        cv2.imwrite(dst_path, patch)