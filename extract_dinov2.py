import torch
import torchvision.transforms as T
from PIL import Image
import os
from featup.util import norm, unnorm
from featup.plotting import plot_feats
from tqdm import tqdm
import json
import torch.nn.functional as F
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


input_size = 448
transform = T.Compose([
    T.Resize([input_size,input_size]),
    #T.CenterCrop((input_size, input_size)),
    T.ToTensor(),
    norm
])


upsampler = torch.hub.load('mhamilton723/FeatUp', 'dinov2').cuda()


img_dir = f"/data/hypernerf_dataset/vrig_chicken/vrig-chicken"


save_path = os.path.join(img_dir,"dino_featup_v2")
os.makedirs(save_path, exist_ok = True)



d_nerf = False
if os.path.exists(os.path.join(img_dir, 'dataset.json')):
    print("Assume Nerfies dataset")
    with open(os.path.join(img_dir,"dataset.json"),'r') as f:
        content = json.load(f)
    ids = content["ids"]
    train_ids = content["train_ids"]
elif os.path.exists(os.path.join(img_dir, 'transforms_train.json')):
    print("Assume D-nerf dataset")
    d_nerf = True
    ids = []
    with open(os.path.join(img_dir,"transforms_train.json"),'r') as f:
        content = json.load(f)
    print(len(content['frames']))

    for frame in content['frames']:
        # print(frame)
        ids.append(frame['file_path'])


else:
    raise NotImplementedError


print(len(ids))
with torch.no_grad():
    
    for (i,img_id) in tqdm(enumerate(ids)):
        if d_nerf:
            print(img_id)
            image_path = os.path.join(img_dir, img_id+".png")
            image_tensor = transform(Image.open(image_path).convert("RGB")).unsqueeze(0).cuda()
        else:
            if not img_id in train_ids:
                continue
            try:
                image_path = os.path.join(img_dir, "rgb/2x",img_id+".png")
                image_tensor = transform(Image.open(image_path).convert("RGB")).unsqueeze(0).cuda()
            except:
                image_path = os.path.join(img_dir, "rgb/4x",img_id+".png")
                image_tensor = transform(Image.open(image_path).convert("RGB")).unsqueeze(0).cuda()

        
        #hr_feats = resize(upsampler(image_tensor)).cpu().half()
        #print(upsampler(image_tensor).shape)
        hr_feats = F.interpolate(upsampler(image_tensor),(256,256), mode = "bilinear").cpu().half()
        torch.save(hr_feats, os.path.join(save_path,str(i).zfill(5)+".pth"))
