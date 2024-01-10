import cv2 
import mmcv 
import os 
import glob
from tqdm import tqdm 
import torch 
import numpy as np


psum    = np.array([0.0, 0.0, 0.0])
psum_sq = np.array([0.0, 0.0, 0.0])
total = 0

data_root = "data/coco/train2017"
for img_file in tqdm(os.listdir(data_root)):
    img = cv2.imread(os.path.join(data_root, img_file))
    img = mmcv.bgr2hsv(img)
    img = img / 255
    psum += img.sum(axis=(0, 1))
    psum_sq += (img ** 2).sum(axis=(0, 1))
    total += np.prod(img.shape[:2])
    
total_mean = psum / total
total_var  = (psum_sq / total) - (total_mean ** 2)
total_std  = np.sqrt(total_var)
print('mean: '  + str(total_mean))
print('std:  '  + str(total_std))
