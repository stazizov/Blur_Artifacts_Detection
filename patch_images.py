import cv2
import os
import xml.etree.ElementTree as ET 
import numpy as np
import json
from math import ceil
from sliding_window import sliding_window
import pandas as pd


def format_coco(bbox):
    x, y, width, height = bbox
    # returns xmin ymin xmax ymax
    return x, ceil(y - height), ceil(x + width), y


def format_crop_path(path: str):
    idx = path.find('_png')

    return path[:idx] + '.png'


def read_bboxes_json(json_file: str = "WSS1_crops.v1i.coco/train/_annotations.coco.json"):
    with open(json_file) as f:
        info = json.load(f)
    images, annotations = info["images"], info["annotations"]
    res = {}

    for image in images:
        img_id = image["id"]
        img_path = format_crop_path(image["file_name"])
        res[img_path] = []

        for anno in annotations:
            if anno["image_id"] == img_id:
                bbox = format_coco(anno["bbox"])
                res[img_path].append(bbox)
    return res

def read_bboxes_xml(xml_file: str):
    xmins, ymins, xmaxes, ymaxes = [], [], [], []

    tree = ET.parse(xml_file)

    for elem in tree.iter():
        if 'object' in elem.tag:
            for attr in list(elem):
                if 'bndbox' in attr.tag:
                    for dim in list(attr):

                        if 'xmin' in dim.tag:
                            xmins.append(int(round(float(dim.text))))

                        if 'ymin' in dim.tag:
                            ymins.append(int(round(float(dim.text))))

                        if 'xmax' in dim.tag:
                            xmaxes.append(int(round(float(dim.text))))

                        if 'ymax' in dim.tag:
                            ymaxes.append(int(round(float(dim.text))))

    return [*zip(xmins, ymins, xmaxes, ymaxes)]

def bboxes2mask(crop_path: str, bboxes: list):
    '''
    returns mask for a given crop of the image
    '''
    crop_path = os.path.join("WSS1/train/images", crop_path)
    img = cv2.imread(crop_path)
    mask_crop = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8) # initialize mask

    for (xmin, ymin, xmax, ymax) in bboxes:
        mask_crop[ymin: ymax, xmin: xmax] = 1
    
    return img, mask_crop

def crops2patches(window_size=32, threshold=0.01):
    try:
        os.mkdir(f"patches_{window_size}")
    except OSError as error:
        print(error)

    iterator = 0
    labels = {}
    threshold = int(window_size * window_size * threshold)
    data = read_bboxes_json()

    for crop_path, bboxes in data.items():
        img, mask = bboxes2mask(crop_path, bboxes)
    
        for x, y, patch in sliding_window(img, window_size, (window_size, window_size)):
            label = 0
            percentile = mask[y: y + window_size, x: x + window_size].sum()

            if percentile >= threshold:
                label = 1
            
            cv2.imwrite(os.path.join(f"patches_{window_size}", f"patch_{iterator}.png"), patch)

            labels[iterator] = label
            iterator += 1

    df = pd.DataFrame.from_dict(labels, orient="index")
    df.to_csv(os.path.join(f"patches_{window_size}", "labels.csv"))
    



if __name__ == '__main__':
    crops2patches()


