import os, math
import numpy as np
import scipy.signal
from typing import List, Optional

from torch import Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F
from .masked_adam import MaskedAdam

import lib_extra.metrics_jax as metrics_jax 
import matplotlib.pyplot as plt
from skimage.color import label2rgb
import cv2
import imageio


import pdb



''' Misc
'''
mse2psnr = lambda x : -10. * torch.log10(x)
to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)


def create_optimizer_or_freeze_model(model, cfg_train, global_step):
    decay_steps = cfg_train.lrate_decay * 1000
    decay_factor = 0.1 ** (global_step/decay_steps)

    param_group = []
    for k in cfg_train.keys():
        if not k.startswith('lrate_'):
            continue
        k = k[len('lrate_'):]

        if not hasattr(model, k):
            continue

        param = getattr(model, k)
        if param is None:
            print(f'create_optimizer_or_freeze_model: param {k} not exist')
            continue

        lr = getattr(cfg_train, f'lrate_{k}') * decay_factor
        if lr > 0:
            print(f'create_optimizer_or_freeze_model: param {k} lr {lr}')
            if isinstance(param, nn.Module):
                param = param.parameters()
            param_group.append({'params': param, 'lr': lr, 'skip_zero_grad': (k in cfg_train.skip_zero_grad_fields)})
        else:
            print(f'create_optimizer_or_freeze_model: param {k} freeze')
            param.requires_grad = False
    return MaskedAdam(param_group)




def load_model_ours(model_class, ckpt_path):
    ckpt = torch.load(ckpt_path)
    model = model_class(**ckpt['model_kwargs'])
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    return model


def load_pretrained_model_whole_hyper(model_class, num_voxels_motion, timesteps, warp_ray, ckpt_path, world_motion_bound_scale, kwargs):
    # TODO ndc condition
    ckpt = torch.load(ckpt_path)
    kwargs['xyz_min'] = ckpt['model_kwargs']['xyz_min']
    kwargs['xyz_max'] = ckpt['model_kwargs']['xyz_max']
    model = model_class(num_voxels_motion=num_voxels_motion, timesteps=timesteps, warp_ray=warp_ray, world_motion_bound_scale=world_motion_bound_scale, **kwargs)
    used_dict = {k: v for k,v in ckpt['model_state_dict'].items() if k=='density' or k.startswith('_time') or k.startswith('camnet')} #only load V_density
    # pdb.set_trace()
    model.load_state_dict(used_dict, strict=False)
    
    print("Load {} successfully!".format(ckpt_path))
    
    return model

def load_pretrained_model_whole(model_class, num_voxels_motion, timesteps, warp_ray, ckpt_path, world_motion_bound_scale, kwargs):
    # TODO ndc condition
    ckpt = torch.load(ckpt_path)
    kwargs['xyz_min'] = ckpt['model_kwargs']['xyz_min']
    kwargs['xyz_max'] = ckpt['model_kwargs']['xyz_max']
    # pdb.set_trace()
    model = model_class(num_voxels_motion=num_voxels_motion, timesteps=timesteps, warp_ray=warp_ray, world_motion_bound_scale=world_motion_bound_scale, **kwargs)
    used_dict = {k: v for k,v in ckpt['model_state_dict'].items() if k=='density' or k.startswith('_time')} #only load V_density
    # pdb.set_trace()
    model.load_state_dict(used_dict, strict=False)
    
    print("Load {} successfully!".format(ckpt_path))
    
    return model


''' Evaluation metrics (ssim, lpips)
'''
def rgb_ssim(img0, img1, max_val,
             filter_size=11,
             filter_sigma=1.5,
             k1=0.01,
             k2=0.03,
             return_map=False):
    # Modified from https://github.com/google/mipnerf/blob/16e73dfdb52044dcceb47cda5243a686391a6e0f/internal/math.py#L58
    assert len(img0.shape) == 3
    assert img0.shape[-1] == 3
    assert img0.shape == img1.shape

    # Construct a 1D Gaussian blur filter.
    hw = filter_size // 2
    shift = (2 * hw - filter_size + 1) / 2
    f_i = ((np.arange(filter_size) - hw + shift) / filter_sigma)**2
    filt = np.exp(-0.5 * f_i)
    filt /= np.sum(filt)

    # Blur in x and y (faster than the 2D convolution).
    def convolve2d(z, f):
        return scipy.signal.convolve2d(z, f, mode='valid')

    filt_fn = lambda z: np.stack([
        convolve2d(convolve2d(z[...,i], filt[:, None]), filt[None, :])
        for i in range(z.shape[-1])], -1)
    mu0 = filt_fn(img0)
    mu1 = filt_fn(img1)
    mu00 = mu0 * mu0
    mu11 = mu1 * mu1
    mu01 = mu0 * mu1
    sigma00 = filt_fn(img0**2) - mu00
    sigma11 = filt_fn(img1**2) - mu11
    sigma01 = filt_fn(img0 * img1) - mu01

    # Clip the variances and covariances to valid values.
    # Variance must be non-negative:
    sigma00 = np.maximum(0., sigma00)
    sigma11 = np.maximum(0., sigma11)
    sigma01 = np.sign(sigma01) * np.minimum(
        np.sqrt(sigma00 * sigma11), np.abs(sigma01))
    c1 = (k1 * max_val)**2
    c2 = (k2 * max_val)**2
    numer = (2 * mu01 + c1) * (2 * sigma01 + c2)
    denom = (mu00 + mu11 + c1) * (sigma00 + sigma11 + c2)
    ssim_map = numer / denom
    ssim = np.mean(ssim_map)
    return ssim_map if return_map else ssim


__LPIPS__ = {}
def init_lpips(net_name, device):
    assert net_name in ['alex', 'vgg']
    import lpips
    print(f'init_lpips: lpips_{net_name}')
    return lpips.LPIPS(net=net_name, version='0.1').eval().to(device)

def rgb_lpips(np_gt, np_im, net_name, device):
    if net_name not in __LPIPS__:
        __LPIPS__[net_name] = init_lpips(net_name, device)
    gt = torch.from_numpy(np_gt).permute([2, 0, 1]).contiguous().to(device)
    im = torch.from_numpy(np_im).permute([2, 0, 1]).contiguous().to(device)
    return __LPIPS__[net_name](gt, im, normalize=True).item()




class ARI(nn.Module):
  """ARI."""

  def forward(self, pr_seg, gt_seg):
    input_pad =np.ones(pr_seg.shape).astype(np.int64)
    gt_instance = gt_seg.max()+1
    pr_instance = pr_seg.max()+1
    

    
    ari_bg =  metrics_jax.Ari.from_model_output(
      predicted_segmentations=pr_seg, ground_truth_segmentations=gt_seg,
      padding_mask=input_pad,
      ground_truth_max_num_instances=gt_instance,
      predicted_max_num_instances=pr_instance,
      ignore_background=False)
    
    ari_nobg =  metrics_jax.Ari.from_model_output(
      predicted_segmentations=pr_seg, ground_truth_segmentations=gt_seg,
      padding_mask=input_pad, 
      ground_truth_max_num_instances=gt_instance,
      predicted_max_num_instances=pr_instance,
      ignore_background=True)
    
    return ari_bg, ari_nobg

def plot_image(ax, img, label=None):
		ax.imshow(img)
		ax.axis('off')
		ax.set_xticks([])
		ax.set_yticks([])
		if label:
			# ax.set_title(label, fontsize=3, y=-21)
			ax.set_xlabel(label, fontsize=3)
			ax.axis('on')

def vis_seg(vid, pr_masks, gt_masks, savedir): # [seq, H, W]
    '''
    args:
    vid: (L, H, W, C)
    gt_mask: (L, H, W, C)
    '''

    # pdb.set_trace()
    seg_pre_array = []
    savedir = os.path.join(savedir, 'seg')
    os.makedirs(savedir, exist_ok=True)

    T = len(vid)

    for i in range(T):
        plt.close()
        fig, ax = plt.subplots(1, 3, dpi=400)

        # for t in range(T):
        # pdb.set_trace()
        vidgrey = cv2.cvtColor(vid[i], cv2.COLOR_RGB2GRAY)[...,None]
        gt_seg = label2rgb(gt_masks[i], vidgrey)
        pred_seg = label2rgb(pr_masks[i], vidgrey)

        plot_image(ax[0], vid[i], 'original')
        plot_image(ax[1], gt_seg[:,:,0,:], 'gt_seg')
        plot_image(ax[2], pred_seg[:,:,0,:], 'pred_seg')

        plt.savefig(os.path.join(savedir, str(i).zfill(3)+'.png'))

        seperate_save_dir = os.path.join(savedir, 'seperate')
        os.makedirs(seperate_save_dir, exist_ok=True)

        # pdb.set_trace()
        cv2.imwrite(os.path.join(seperate_save_dir, str(i).zfill(3)+'_gt_seg.png'), (gt_seg[:, :, 0, :]*255))
        cv2.imwrite(os.path.join(seperate_save_dir, str(i).zfill(3)+'_pred_seg.png'), (pred_seg[:, :, 0, :]*255))
        seg_pre_array.append(pred_seg[:, :, 0, :]*255)

    # pdb.set_trace()
    seg_pre_array = np.array(seg_pre_array)
    imageio.mimwrite(os.path.join(savedir, 'seg.mp4'), seg_pre_array, fps=10, quality=8)
        
        
