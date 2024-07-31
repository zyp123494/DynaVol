import warnings

warnings.filterwarnings("ignore")

import json
import os
import random

import numpy as np
import torch
from PIL import Image


class Load_hyper_data():
    def __init__(self, 
                 datadir, 
                 ratio=0.5,
                 use_bg_points=False,
                 add_cam=False):
        from .utils import Camera
        datadir = os.path.expanduser(datadir)
        self.datadir = datadir
        with open(f'{datadir}/scene.json', 'r') as f:
            scene_json = json.load(f)
        with open(f'{datadir}/metadata.json', 'r') as f:
            meta_json = json.load(f)
        # with open(f'{datadir}/dataset.json', 'r') as f:
        #     dataset_json = json.load(f)

        if os.path.exists(f'{datadir}/dataset_interleave.json'):
            print('using interleave')
            with open(f'{datadir}/dataset_interleave.json', 'r') as f:
                dataset_json = json.load(f)
        else:
            with open(f'{datadir}/dataset.json', 'r') as f:
                dataset_json = json.load(f)

        self.near = scene_json['near']
        self.far = scene_json['far']
        self.coord_scale = scene_json['scale']
        self.scene_center = scene_json['center']

        self.all_img = dataset_json['ids']
        self.val_id = dataset_json['val_ids']

        self.add_cam = add_cam
        if len(self.val_id) == 0:
            if datadir.split('/')[-2].startswith('interp'):
                print("Assume interp dataset")
                self.i_train = np.array([i for i in np.arange(len(self.all_img)) if
                                (i%4 == 0)])
                self.i_test = self.i_train+2
                self.i_test = self.i_test[:-1,]
            else: # for hypernerf
                print("Assume misc dataset, use all data to train")
                self.i_train = np.array([i for i in np.arange(len(self.all_img))])
                self.i_test = np.array([i for i in np.arange(len(self.all_img))])
        else:
            #self.add_cam = True
            
            self.train_id = dataset_json['train_ids']
            self.i_test = []
            self.i_train = []
            for i in range(len(self.all_img)):
                id = self.all_img[i]
                if id in self.val_id:
                    self.i_test.append(i)
                if id in self.train_id:
                    self.i_train.append(i)
        assert self.add_cam == add_cam
        
        print('self.i_train',self.i_train)
        print('self.i_test',self.i_test)
        self.all_cam = [meta_json[i]['camera_id'] for i in self.all_img]
        try:
            self.all_time = [meta_json[i]['time_id'] for i in self.all_img]
            max_time = max(self.all_time)
            self.all_time = [meta_json[i]['time_id']/max_time for i in self.all_img]
        except KeyError:
            tmp = self.all_img[0]
            idx = -1
            #print(tmp,idx)
            while tmp[idx].isdigit():
                idx -=1 
                if -idx>len(tmp):
                    break
                #print(tmp,idx)
            print(tmp[idx+1:])
            self.all_time = [int(i[idx+1:]) for i in self.all_img]#[i / (len(self.all_img) - 1)for i in range(len(self.all_img))]
        min_t,max_t = min(self.all_time),max(self.all_time)
        self.all_time = [(x-min_t) / (max_t-min_t) for x in self.all_time]
        self.selected_time = set(self.all_time)
        self.ratio = ratio
        assert min(self.all_time) == 0


        self.selected_time = set(self.all_time)
        self.ratio = ratio


        # all poses
        self.all_cam_params = []
        for im in self.all_img:
            camera = Camera.from_json(f'{datadir}/camera/{im}.json')
            camera = camera.scale(ratio)
            camera.position = camera.position - self.scene_center
            camera.position = camera.position * self.coord_scale
            self.all_cam_params.append(camera)

        #self.all_img = [f'{datadir}/rgb/{int(1/ratio)}x/{i}.png' for i in self.all_img]
        self.h, self.w = self.all_cam_params[0].image_shape

        self.use_bg_points = use_bg_points
        if use_bg_points:
            with open(f'{datadir}/points.npy', 'rb') as f:
                points = np.load(f)
            self.bg_points = (points - self.scene_center) * self.coord_scale
            self.bg_points = torch.tensor(self.bg_points).float()
        print(f'total {len(self.all_img)} images ',
                'use cam =',self.add_cam, 
                'use bg_point=',self.use_bg_points)

        if os.path.exists(f'{datadir}/rgb-raw'):
            print('Using raw rgb')
            self.raw = True
            self.all_img = [f'{datadir}/rgb-raw/{i}.png' for i in self.all_img]
        else:
            self.raw = False
            self.all_img = [f'{datadir}/rgb/{int(1/ratio)}x/{i}.png' for i in self.all_img]

    def load_idx(self, idx,not_dic=False, get_pixel_pos = False):

        all_data = self.load_raw(idx)
        if not_dic == True:
            rays_o = all_data['rays_ori']
            rays_d = all_data['rays_dir']
            viewdirs = all_data['viewdirs']
            rays_color = all_data['rays_color']
            if get_pixel_pos:
                pixels = (all_data["pixels"] * 2 - 1)#.flip(-1)  #switch to H, W
                return  rays_o, rays_d, viewdirs,rays_color, pixels
            return rays_o, rays_d, viewdirs,rays_color
        return all_data

    def load_raw(self, idx):
        image = Image.open(self.all_img[idx])
        if self.raw:
            w, h = image.size
            image = image.resize((int(round(w * self.ratio)), int(round(h*self.ratio))), Image.LANCZOS)
        camera = self.all_cam_params[idx]
        pixels = camera.get_pixel_centers()
        h, w = pixels.shape[:2]
        norm_factor = np.array([w,h]).reshape([1,2])
        
        rays_dir = torch.tensor(camera.pixels_to_rays(pixels)).float().view([-1,3])
        rays_ori = torch.tensor(camera.position[None, :]).float().expand_as(rays_dir)
        rays_color = torch.tensor(np.array(image)).view([-1,3])/255.
        # print(pixels.shape,pixels[...,0].max(), pixels[...,1].max())   #[H, W, 2]   W,H
        # print(self.all_img[idx])
        return {'rays_ori': rays_ori, 
                'rays_dir': rays_dir, 
                'viewdirs':rays_dir / rays_dir.norm(dim=-1, keepdim=True),
                'rays_color': rays_color, 
                "pixels": torch.tensor(pixels.reshape([-1,2]) / norm_factor),
                'near': torch.tensor(self.near).float().view([-1]), 
                'far': torch.tensor(self.far).float().view([-1]),}
