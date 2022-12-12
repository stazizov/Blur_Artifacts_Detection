import cv2
import torch
import numpy as np
import argparse
import matplotlib.pyplot as plt
from models import seresnet18, mobilenetv3



def inference(image: np.ndarray, window_size: int, model: torch.nn.Module) -> np.ndarray:
    heatmap = np.zeros(
            (
            image.shape[0]//window_size+1, 
            image.shape[1]//window_size+1
            )
        )
    image = image / 255
    for y in range(0, image.shape[0], window_size):
        for x in range(0, image.shape[1], window_size):
            crop = image[y:y + window_size, x:x + window_size]
            crop = cv2.resize(crop, (window_size, window_size))
            crop = torch.from_numpy(crop).permute(2, 0, 1).unsqueeze(0).float()
            outputs = torch.sigmoid(model(crop)).detach().numpy()
            heatmap[y//window_size, x//window_size] = outputs
    
    if sum(heatmap[-1]) == 0:
        heatmap = heatmap[:-1]
    heatmap = cv2.resize(heatmap, image.shape[:-1][::-1])
    heatmap = np.expand_dims(heatmap, -1)
    return heatmap
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path2image", type=str)
    parser.add_argument("--path2save", type=str)
    parser.add_argument("--model", type=str)
    args = parser.parse_args()
    
    model = seresnet18() if 'res' in args.model else mobilenetv3()
    image = cv2.imread(args.path2image)
    heatmap = inference(image, 224, model)
    combined = (image * heatmap).astype(np.uint8)
    plt.imshow(combined)
    plt.show()