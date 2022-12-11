import pandas as pd
import cv2
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic
from metrics import metrics as cpbd
from catboost import CatBoostClassifier
import os
import numpy as np
from tqdm import tqdm
import scipy
from sklearn.metrics import f1_score as f1
from sklearn.metrics import roc_auc_score
print(scipy.__version__)
from scipy.stats import kurtosis 
import pywt
import argparse

def blur_detect(img, threshold=0.05):
    
    # Convert image to grayscale
    # Y = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    Y=img.copy()
    M, N = Y.shape
    
    # Crop input image to be 3 divisible by 2
    Y = Y[0:int(M/16)*16, 0:int(N/16)*16]
    
    # Step 1, compute Haar wavelet of input image
    LL1,(LH1,HL1,HH1)= pywt.dwt2(Y, 'haar')
    # Another application of 2D haar to LL1
    LL2,(LH2,HL2,HH2)= pywt.dwt2(LL1, 'haar') 
    # Another application of 2D haar to LL2
    LL3,(LH3,HL3,HH3)= pywt.dwt2(LL2, 'haar')
    
    # Construct the edge map in each scale Step 2
    E1 = np.sqrt(np.power(LH1, 2)+np.power(HL1, 2)+np.power(HH1, 2))
    E2 = np.sqrt(np.power(LH2, 2)+np.power(HL2, 2)+np.power(HH2, 2))
    E3 = np.sqrt(np.power(LH3, 2)+np.power(HL3, 2)+np.power(HH3, 2))
    
    M1, N1 = E1.shape

    # Sliding window size level 1
    sizeM1 = 8
    sizeN1 = 8
    
    # Sliding windows size level 2
    sizeM2 = int(sizeM1/2)
    sizeN2 = int(sizeN1/2)
    
    # Sliding windows size level 3
    sizeM3 = int(sizeM2/2)
    sizeN3 = int(sizeN2/2)
    
    # Number of edge maps, related to sliding windows size
    N_iter = int((M1/sizeM1)*(N1/sizeN1))
    
    Emax1 = np.zeros((N_iter))
    Emax2 = np.zeros((N_iter))
    Emax3 = np.zeros((N_iter))
    
    
    count = 0
    
    # Sliding windows index of level 1
    x1 = 0
    y1 = 0
    # Sliding windows index of level 2
    x2 = 0
    y2 = 0
    # Sliding windows index of level 3
    x3 = 0
    y3 = 0
    
    # Sliding windows limit on horizontal dimension
    Y_limit = N1-sizeN1
    
    while count < N_iter:
        # Get the maximum value of slicing windows over edge maps 
        # in each level
        Emax1[count] = np.max(E1[x1:x1+sizeM1,y1:y1+sizeN1])
        Emax2[count] = np.max(E2[x2:x2+sizeM2,y2:y2+sizeN2])
        Emax3[count] = np.max(E3[x3:x3+sizeM3,y3:y3+sizeN3])
        
        # if sliding windows ends horizontal direction
        # move along vertical direction and resets horizontal
        # direction
        if y1 == Y_limit:
            x1 = x1 + sizeM1
            y1 = 0
            
            x2 = x2 + sizeM2
            y2 = 0
            
            x3 = x3 + sizeM3
            y3 = 0
            
            count += 1
        
        # windows moves along horizontal dimension
        else:
                
            y1 = y1 + sizeN1
            y2 = y2 + sizeN2
            y3 = y3 + sizeN3
            count += 1
    
    # Step 3
    EdgePoint1 = Emax1 > threshold;
    EdgePoint2 = Emax2 > threshold;
    EdgePoint3 = Emax3 > threshold;
    
    # Rule 1 Edge Pojnts
    EdgePoint = EdgePoint1 + EdgePoint2 + EdgePoint3
    
    n_edges = EdgePoint.shape[0]
    
    # Rule 2 Dirak-Structure or Astep-Structure
    DAstructure = (Emax1[EdgePoint] > Emax2[EdgePoint]) * (Emax2[EdgePoint] > Emax3[EdgePoint]);
    
    # Rule 3 Roof-Structure or Gstep-Structure
    
    RGstructure = np.zeros((n_edges))

    for i in range(n_edges):
    
        if EdgePoint[i] == 1:
        
            if Emax1[i] < Emax2[i] and Emax2[i] < Emax3[i]:
            
                RGstructure[i] = 1
                
    # Rule 4 Roof-Structure
    
    RSstructure = np.zeros((n_edges))

    for i in range(n_edges):
    
        if EdgePoint[i] == 1:
        
            if Emax2[i] > Emax1[i] and Emax2[i] > Emax3[i]:
            
                RSstructure[i] = 1

    # Rule 5 Edge more likely to be in a blurred image 

    BlurC = np.zeros((n_edges));

    for i in range(n_edges):
    
        if RGstructure[i] == 1 or RSstructure[i] == 1:
        
            if Emax1[i] < threshold:
            
                BlurC[i] = 1                        
        
    # Step 6
    Per = np.sum(DAstructure)/np.sum(EdgePoint)
    
    # Step 7
    if (np.sum(RGstructure) + np.sum(RSstructure)) == 0:
        
        BlurExtent = 100
    else:
        BlurExtent = np.sum(BlurC) / (np.sum(RGstructure) + np.sum(RSstructure))
    
    return Per, BlurExtent

def generate_points(n, ax, ay):
  y  = [i+ay for i in range(-n, n+1)]
  x  = [i+ax for i in range(-n, n+1)]
  points = []
  for i in range(len(y)):
    for j in range(len(x)):
      points.append([x[i], y[j]])
  return points

def get_local_features(x, metric, **kwagrs):
  metric_features=[]  
  for i in range(2, x.shape[0]):


    local_features = metric(x[0:i, 0:i])
    if np.isnan(local_features).any():
      pass
    else:
      metric_features.append(local_features)


    local_features = metric(x[0:x.shape[0]-i, 0:i])
    if np.isnan(local_features).any():
      pass
    else:
      metric_features.append(local_features)


    local_features = metric(x[0:i, 0:x.shape[0]-i])
    if np.isnan(local_features).any():
      pass
    else:
      metric_features.append(local_features)

    local_features = metric(x[x.shape[0]-i-1, i])
    if np.isnan(local_features).any():
      pass
    else:
      metric_features.append(local_features)
 
  return np.asarray(metric_features, dtype=object)

def fourier(image):
  try:
    F1 = np.fft.fft2(image)
  except:
    return np.nan

  # Now shift the quadrants around so that low spatial frequencies are in
  # the center of the 2D fourier transformed image.
  F2 = np.fft.fftshift( F1 )

  # Calculate a 2D power spectrum
  psd2D = np.abs( F2 )**2
  psd2D.flatten()
  
  return psd2D

def get_median_features(x, metric=kurtosis, num_bins=10):
  
  features = get_local_features(x, metric)
  features_binned = []

  for i in range(len(features)):
    data = features[i].flatten()

    mn = min(data)
    mx = max(data)
    bin_means = binned_statistic(data, data, bins=num_bins, range=(mn, mx))[0]
    
    # print(bin_means)
    features_binned.append(np.mean(bin_means[~np.isnan(bin_means)]))
  return features_binned

def get_features(path="patches/patches_128/", metric=kurtosis, size=128):
  cbpd_features = []
  wavelet_features = []

  labels = pd.read_csv(os.path.join(path, 'labels.csv'))
  metric_features = []
  y = []
  for i in tqdm(labels.iloc):
    im = cv2.imread(os.path.join(path,f"patch_{i['Unnamed: 0']}.png"))  
    # print(im)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im = cv2.resize(im, (size, size))
    # print(wavelet(im).shape)
  

    im_features = get_median_features(im, metric)
    # print(im_features)
    if np.isnan(np.asarray(im_features)).any():
      continue
    c1, c2 = blur_detect(im)
    wavelet_features.append(c1)
    cbpd_features.append(cpbd.compute(im))
    metric_features.append(im_features)

    y.append(i['0'])
  max_len = max([len(x) for x in metric_features])
  output = [np.pad(x, (0, max_len - len(x)), 'constant')  for x in metric_features]
  return output, y, cbpd_features, wavelet_features

def get_median_features(x, metric=kurtosis, num_bins=10):
  
  features = get_local_features(x, metric)
  features_binned = []

  for i in range(len(features)):
    data = features[i].flatten()

    mn = min(data)
    mx = max(data)
    bin_means = binned_statistic(data, data, bins=num_bins, range=(mn, mx))[0]
    
    # print(bin_means)
    features_binned.append(np.mean(bin_means[~np.isnan(bin_means)]))
  return features_binned

def get_features(path="patches/patches_128/", metric=kurtosis, size=128):
  cbpd_features = []
  wavelet_features = []

  labels = pd.read_csv(os.path.join(path, 'labels.csv'))
  metric_features = []
  y = []
  for i in tqdm(labels.iloc):
    im = cv2.imread(os.path.join(path,f"patch_{i['Unnamed: 0']}.png"))  
    # print(im)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im = cv2.resize(im, (size, size))
    # print(wavelet(im).shape)
  

    im_features = get_median_features(im, metric)
    # print(im_features)
    if np.isnan(np.asarray(im_features)).any():
      continue
    c1, c2 = blur_detect(im)
    wavelet_features.append(c1)
    cbpd_features.append(cpbd.compute(im))
    metric_features.append(im_features)

    y.append(i['0'])
  max_len = max([len(x) for x in metric_features])
  output = [np.pad(x, (0, max_len - len(x)), 'constant')  for x in metric_features]
  return output, y, cbpd_features, wavelet_features

def sliding_window(image, stepSize, windowSize, metric, classifier):
  cbpd_features = []
  wavelet_features = []
  names = []
  metric_features = []
  for y in tqdm(range(0, image.shape[0], stepSize)):
    for x in range(0, image.shape[1], stepSize):
      crop = image[y:y + windowSize[1], x:x + windowSize[0]]
      crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
      crop = cv2.resize(crop, windowSize)
      im_features = get_median_features(crop, metric)
      c1, c2 = blur_detect(crop)
      cpbd_global_metric= cpbd.compute(crop)
      wavelet_features.append(c1)
      cbpd_features.append(cpbd_global_metric)
      metric_features.append(im_features)
      names.append(f"patch_{x//stepSize}_{y // stepSize}.png")
  max_len = max([len(x) for x in metric_features])    
  output = [np.pad(x, (0, max_len - len(x)), 'constant')  for x in metric_features]
  df = pd.DataFrame(output, columns = [f"dim_{i}" for i in range(len(output[0]))])
  
  df['cpbd'] = cbpd_features
  df['wavelet'] = wavelet_features
  preds = np.max(classifier.predict_proba(df), axis=1)
  dd = {names[i]:preds[i] for i in range(len(names))}
  inference_result = image.copy()
  for y in range(0, image.shape[0], stepSize):
    for x in range(0, image.shape[1], stepSize):
      crop = image[y:y + windowSize[1], x:x + windowSize[0]]
      heat_level = dd[ f"patch_{x//stepSize}_{y // stepSize}.png"]*255
      inference_result[y:y + windowSize[1], x:x + windowSize[0]] = heat_level
  
  return inference_result

def get_catboost():
  model = CatBoostClassifier()

  model = model.load_model('64_full.uu')
  return model

def visualize_heatmap(image, heatmap):
    threshed_heatmap = heatmap.copy() / 255
    threshed_heatmap = cv2.resize(
        threshed_heatmap, 
        image.shape[::-1][1:]
        )
    return (threshed_heatmap * image).astype(np.uint8)

def convert_path(path) -> str:
  filename = "mixed_" + os.path.split(path)[-1]
  return os.path.join(*os.path.split(path)[:-1]+(filename,))

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--path2image", type=str)
  parser.add_argument("--path2save", type=str)
  parser.add_argument("--window_size", default=64, type=str)
  args = parser.parse_args()

  model = get_catboost()

  image = cv2.imread(args.path2image)
  heatmap = sliding_window(
    image, 
    args.window_size, 
    (args.window_size, args.window_size), 
    metric = fourier, 
    classifier=model
    )
  visualized_image = visualize_heatmap(image, heatmap)
  cv2.imwrite(convert_path(args.path2save), visualized_image)
  print(f"combined mask save at: {convert_path(args.path2save)}")
  plt.title("blur heatmap (closer to 1 -> highter probability)")
  plt.axis('off')
  plt.imshow(heatmap[:, :, 0]/255, cmap='magma')
  plt.colorbar()
  plt.savefig(args.path2save)
  print(f"heatmap saved at: {args.path2save}")