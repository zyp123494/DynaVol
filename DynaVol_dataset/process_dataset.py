import glob
import os
import numpy as np
import skimage.io
from PIL import Image
from tqdm import tqdm
base_path = "./output"
frames = 60

dataset = ['static','dynamic']
split = ['train','val','test']
for d in dataset:
    for s in split:
        for i in tqdm(range(frames)):
            img = skimage.io.imread(f'{base_path}/{d}/{s}/{str(i).zfill(3)}.png').astype(np.float64)
            mask = np.array(Image.open(f'{base_path}/{d}/{s}/segmentation_{str(i).zfill(5)}.png')).astype(np.float64)
            mask = (mask>0)
            img[:,:,3] = img[:,:,3]* mask
            img = img.astype(np.uint8)
            skimage.io.imsave(f'{base_path}/{d}/{s}/{str(i).zfill(3)}.png',img)


d = "dynamic_4views"
split =["view0","view1","view2","view3"]

os.makedirs(f'{base_path}/{d}/train',exist_ok = True)
for i in tqdm(range(frames)):
    for (j,s) in enumerate(split):
        img = skimage.io.imread(f'{base_path}/{d}/{s}/{str(i).zfill(3)}.png').astype(np.float64)
        mask = Image.open(f'{base_path}/{d}/{s}/segmentation_{str(i).zfill(5)}.png')
        mask_arr = np.array(mask).astype(np.float64)
        mask_arr = (mask_arr>0)
        img[:,:,3] = img[:,:,3]* mask_arr
        img = img.astype(np.uint8)
        skimage.io.imsave(f'{base_path}/{d}/train/{str(i*4 + j).zfill(3)}.png',img)
        mask.save(f'{base_path}/{d}/train/segmentation_{str(i*4 + j).zfill(5)}.png')

