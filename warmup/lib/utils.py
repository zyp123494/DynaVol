import os, math
import numpy as np
import scipy.signal
from typing import List, Optional

from torch import Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F
from .masked_adam import MaskedAdam

import matplotlib.pyplot as plt
from skimage.color import label2rgb
import cv2

from typing import Sequence, Union
Array = Union[np.ndarray, torch.Tensor]

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




''' Evaluation metrics (ssim, lpips)
'''
def rgb_ssim(img0, img1, max_val,
             filter_size=11,
             filter_sigma=1.5,
             k1=0.01,
             k2=0.03,
             return_map=False):
    
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


        
        
def create_gradient_grid(
    samples_per_dim: Sequence[int], value_range: Sequence[float] = (-1.0, 1.0)
    ) -> Array:
    """Creates a tensor with equidistant entries from -1 to +1 in each dim
    
    Args:
        samples_per_dim: Number of points to have along each dimension.
        value_range: In each dimension, points will go from range[0] to range[1]
    
    Returns:
        A tensor of shape [samples_per_dim] + [len(samples_per_dim)].
    """

    s = [np.linspace(value_range[0], value_range[1], n) for n in samples_per_dim]
    pe = np.stack(np.meshgrid(*s, sparse=False, indexing="ij"), axis=-1)
    return np.array(pe)

def convert_to_fourier_features(inputs: Array, basis_degree: int) -> Array:
    """Convert inputs to Fourier features, e.g. for positional encoding."""

    # inputs.shape = (..., n_dims).
    # inputs should be in range [-pi, pi] or [0, 2pi].
    n_dims = inputs.shape[-1]

    # Generate frequency basis
    freq_basis = np.concatenate( # shape = (n_dims, n_dims * basis_degree)
        [2**i * np.eye(n_dims) for i in range(basis_degree)], 1)
    
    # x.shape = (..., n_dims * basis_degree)
    x = inputs @ freq_basis # Project inputs onto frequency basis.

    # Obtain Fourier feaures as [sin(x), cos(x)] = [sin(x), sin(x + 0.5 * pi)].
    return np.sin(np.concatenate([x, x + 0.5 * np.pi], axis=-1))
