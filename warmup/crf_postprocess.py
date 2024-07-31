import numpy as np
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax
import cv2
import matplotlib.pyplot as plt
import skimage.io
from tqdm import tqdm
import glob
import os
import json
from PIL import Image
    
def gray2rgb(seg):
    from PIL import Image
    import skimage.io
    np.random.seed(200)
    
    PALETTE = [0, 0, 0, 128, 0, 0, 0, 128, 0, 128, 128, 0, 0, 0, 128, 128, 0, 128, 0, 128, 128, 128, 128, 128, 64, 0, 0, 191, 0, 0]
    _palette = ((np.random.random((3*(255-len(PALETTE))))*0.7+0.3)*255).astype(np.uint8).tolist()
    PALETTE += _palette
    img =Image.fromarray(seg.astype(np.uint8),mode = 'P')
    img.putpalette(PALETTE)
    return img

base_path = "./logs/chicken/render_train_fine_last"

data_dir = "/data/ypzhao/hypernerf_dataset/vrig_chicken/vrig-chicken"

# base_path = "./logs/small/8obj/render_train_fine_last"

# data_dir = "/data/8obj/dynamic"

save_path = os.path.join(base_path,"crf_seg")
os.makedirs(save_path,exist_ok = True)

d_nerf = False
if os.path.exists(os.path.join(data_dir, 'dataset.json')):
    print("Assume Nerfies dataset")
    with open(os.path.join(data_dir,"dataset.json"),'r') as f:
        content = json.load(f)
    ids = content["train_ids"]
    if data_dir.split("/")[-2].startswith("interp"):
        ids = ids[::4]
    imgs = [os.path.join(data_dir,"rgb/2x",x+".png") for x in ids ]
elif os.path.exists(os.path.join(data_dir, 'transforms_train.json')):
    print("Assume D-nerf dataset")
    d_nerf = True
    ids = []
    with open(os.path.join(data_dir,"transforms_train.json"),'r') as f:
        content = json.load(f)
    print(len(content['frames']))

    for frame in content['frames']:
        # print(frame)
        ids.append(frame['file_path'])
    imgs = [os.path.join(data_dir, x+'.png') for x in ids]


else:
    raise NotImplementedError


probs = sorted(glob.glob(f"{base_path}/seg_raw*"),key = lambda x: int(x.split("/")[-1].split("_")[-1].split(".")[0]))

"""
对语义分割结果应用CRF后处理。

参数:
- image_path: 原始图像的路径。
- probabilities: 一个H*W*K的numpy数组，包含每个像素属于每个类别的概率。

返回:
- map: H*W的numpy数组，包含每个像素点的最终类别。
"""
for (i, image_path) in tqdm(enumerate(imgs)):

    image = Image.open(image_path)
    w, h = image.size

    probabilities = np.load(probs[i])
    H, W, K = probabilities.shape

    if w!=W or h!=H:
        image = image.resize((W,H), Image.LANCZOS)

    image = (np.array(image)[...,:3]).astype(np.uint8)

    print("shape of image", image.shape)
   
    probabilities = probabilities.transpose(2, 0, 1).reshape((K, -1))

    sxy = 100
    srgb =8
    compat = 3


    d = dcrf.DenseCRF2D(W, H, K)


    # 计算一元势能
    U = unary_from_softmax(probabilities)
    U = np.ascontiguousarray(U)
    d.setUnaryEnergy(U)


    d.addPairwiseGaussian(sxy=(3, 3), compat=3)

  
    d.addPairwiseBilateral(sxy=(sxy,sxy), srgb=(srgb,srgb,srgb), rgbim=image, compat=compat)
    # 推断
    Q = d.inference(10)  

    # 获取最终的分类结果
    seg = np.argmax(Q, axis=0).reshape((H, W))
    
    seg_vis = gray2rgb(seg)
    if d_nerf:
        seg_vis = seg_vis.resize((256, 256), Image.NEAREST)
    seg_vis.save(f'{save_path}/seg_{str(i).zfill(3)}.png')
  
        