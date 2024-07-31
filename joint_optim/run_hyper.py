import time

import os, sys, copy, glob, json, time, random, argparse
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
from shutil import copyfile
from tqdm import tqdm, trange
import skimage.io
from PIL import Image
import mmcv
import imageio
import numpy as np
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib import utils
from lib.utils import get_linear_noise_func
from lib import voxelMlp_hyper as VoxelMlp

from lib import voxelMlp_syn as VoxelMlp_syn

from lib.load_data import load_data
# from post_process import post_process
from pytorch_msssim import ms_ssim
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import pdb
import glob
from torch.utils.data import Dataset
from lib.voxelMlp_hyper import sin_emb
from torch_efficient_distloss import flatten_eff_distloss






def config_parser():
    '''Define command line arguments
    '''

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', required=True,
                        help='config file path')
    parser.add_argument("--seed", type=int, default=0,
                        help='Random seed')
    parser.add_argument("--no_reload", action='store_true',
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--no_reload_optimizer", action='store_true',
                        help='do not reload optimizer state from saved ckpt')
    parser.add_argument("--ft_path", type=str, default='',
                        help='specific weights npy file to reload for coarse network')
    parser.add_argument("--export_bbox_and_cams_only", type=str, default='',
                        help='export scene bbox and camera poses for debugging and 3d visualization')
    parser.add_argument("--export_coarse_only", type=str, default='')

    # testing options
    parser.add_argument("--render_only", action='store_true',
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_test", action='store_true')
    parser.add_argument("--render_train", action='store_true')
    parser.add_argument("--render_static", action='store_true')
    parser.add_argument("--render_video", action='store_true')
    parser.add_argument("--render_video_factor", type=int, default=0,
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')
    parser.add_argument("--eval_ssim", action='store_true')
    parser.add_argument("--eval_lpips_alex", action='store_true')
    parser.add_argument("--eval_lpips_vgg", action='store_true')
    parser.add_argument("--bs", type=int, default=4096)
    parser.add_argument("--step_to_half", type=int, default=19000,
                        help='The iteration when fp32 becomes fp16')

    # logging/saving options
    parser.add_argument("--i_print",   type=int, default=500,
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_weights", type=int, default=5000,
                        help='frequency of weight ckpt saving')

    parser.add_argument('--eval_ari', action='store_true')
    parser.add_argument('--per_slot', action='store_true')
    return parser


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

@torch.no_grad()
def render_viewpoints_hyper(model, data_class, ndc, render_kwargs, test=True,  slot_idx = -1,
                                all=True, savedir=None, eval_psnr=False,eval_lpips_alex= True, eval_lpips_vgg=True, eval_ssim=True):
    
    rgbs = []
    rgbs_gt =[]
    rgbs_tensor =[]
    rgbs_gt_tensor =[]
    depths = []
    psnrs = []
    segmentations = []
    ms_ssims =[]
    lpips_alex = []
    lpips_vgg = []
    segmentations_raw = []


    if test:
        if all:
            idx = data_class.i_test
        else:
            idx = data_class.i_test[::16]
    else:
        if all:
            idx = data_class.i_train
        else:
            idx = data_class.i_train[::len(data_class.i_train)//16]
 

    for i in tqdm(idx):
       

        render_kwargs['segmentation']=True
        keys = ['rgb_marched', 'depth','segmentation']
      
        rays_o, rays_d, viewdirs,rgb_gt = data_class.load_idx(i, not_dic=True)
        #print(render_kwargs.get('segmentation', True))
        time_one = data_class.all_time[i]*torch.ones_like(rays_o[:,0:1])
        cam_one = data_class.all_cam[i]*torch.ones_like(rays_o[:,0:1])
        bacth_size = 3000
        render_result_chunks = [
            {k: v for k, v in model(ro, rd, vd,ts, cams,slot_idx = slot_idx,**render_kwargs).items() if k in keys}
            for ro, rd, vd ,ts,cams in zip(rays_o.split(bacth_size, 0), rays_d.split(bacth_size, 0),
                                             viewdirs.split(bacth_size, 0),time_one.split(bacth_size, 0),cam_one.split(bacth_size, 0))
        ]
        render_result = {
            k: torch.cat([ret[k] for ret in render_result_chunks]).reshape(data_class.h,data_class.w,-1)
            for k in render_result_chunks[0].keys()
        }
        rgb_gt = rgb_gt.reshape(data_class.h,data_class.w,-1).cpu().numpy()
        rgb = render_result['rgb_marched'].cpu().numpy()
        depth = render_result['depth'].cpu().numpy()
       
        
      
        rgbs.append(rgb)
        depths.append(depth)
        rgbs_gt.append(rgb_gt)
    
        if eval_psnr:
            p = -10. * np.log10(np.mean(np.square(rgb - rgb_gt)))
            psnrs.append(p)
            rgbs_tensor.append(torch.from_numpy(np.clip(rgb,0,1)).reshape(-1,data_class.h,data_class.w))
            rgbs_gt_tensor.append(torch.from_numpy(np.clip(rgb_gt,0,1)).reshape(-1,data_class.h,data_class.w))
        
        if eval_lpips_alex:
            lpips_alex.append(utils.rgb_lpips(rgb, rgb_gt, net_name = 'alex', device = rays_o.device))
        if eval_lpips_vgg:
            lpips_vgg.append(utils.rgb_lpips(rgb, rgb_gt, net_name = 'vgg', device = rays_o.device))
            #ms_ssims.append(utils.rgb_ssim(rgb, rgb_gt, max_val=1))
        if i==0:
            print('Testing', rgb.shape)

        if render_kwargs.get('segmentation', True):
            seg = render_result['segmentation'].cpu().numpy()
            segmentations_raw.append(seg.copy())
            
            segmentations.append(seg.copy().argmax(-1)[...,None])
        #break
  

    if len(segmentations):
       # seg_vis = []
        for (idx,seg) in enumerate(segmentations):
            seg = gray2rgb(seg[...,0])
            seg.save(os.path.join(savedir, f"seg_{str(idx)}.png"))
    del segmentations



    
    if eval_ssim: 
        rgbs_tensor = torch.stack(rgbs_tensor,0)
        rgbs_gt_tensor = torch.stack(rgbs_gt_tensor,0)
        ms_ssims = ms_ssim(rgbs_gt_tensor, rgbs_tensor, data_range=1, size_average=True )
    
   

    f1 = open(os.path.join(savedir, 'result.txt'), 'w')
    if len(psnrs):
        print('Testing psnr', np.mean(psnrs), '(avg)')
        f1.write('Testing psnr:' + str(np.mean(psnrs)) + '(avg)\n')
        # print('Testing ms_ssims', np.mean(ms_ssims), '(avg)')
        np.save(os.path.join(savedir, 'psnr.npy'), np.array(psnrs))
        # if writer is not None:
        #     writer.add_scalar('test/psnr', np.mean(psnrs), gs)
        
        if eval_ssim: 
            print('Testing ms_ssims', ms_ssims, '(avg)')
            f1.write('Testing ms_ssims' + str(ms_ssims) + '(avg)\n')
            np.save(os.path.join(savedir, 'ssim.npy'), ms_ssims)
            # if writer is not None:
            #     writer.add_scalar('test/ssim', ms_ssims, gs)

        if eval_lpips_alex:
            print('Testing lpips_alex', np.mean(lpips_alex), '(avg)')
            f1.write('Testing lpips_alex' + str(np.mean(lpips_alex)) + '(avg)\n')
        if eval_lpips_vgg:
            print('Testing lpips_vgg', np.mean(lpips_vgg), '(avg)')
            f1.write('Testing lpips_vgg' + str(np.mean(lpips_vgg)) + '(avg)\n')

    f1.close()
    
    if len(psnrs):
        print('Testing psnr', np.mean(psnrs), '(avg)')
        #print('Testing ms_ssims', np.mean(ms_ssims), '(avg)')

    if savedir is not None:
        print(f'Writing images to {savedir}')
        for i in trange(len(rgbs)):
            rgb8 = utils.to8b(rgbs[i])
            filename = os.path.join(savedir, '{:03d}.png'.format(i))
            rgb8 = Image.fromarray(rgb8)
            rgb8.save(filename)

     
    rgbs = np.array(rgbs)
    depths = np.array(depths)
    return rgbs,depths


@torch.no_grad()
def render_viewpoints(model, render_poses, HW, Ks, ndc, render_kwargs,
                      gt_imgs=None, savedir=None, test_times=None, render_factor=0, eval_psnr=False, slot_idx = -1,
                      eval_ssim=False, eval_lpips_alex=False, eval_lpips_vgg=False,batch = None):
    '''Render images for the given viewpoints; run evaluation if gt given.
    '''
    assert len(render_poses) == len(HW) and len(HW) == len(Ks)

    ari_metrics = utils.ARI()

    if batch is not None:
        gt_segmentations,video = batch
        assert len(render_poses) == gt_segmentations.shape[1]
    
    if render_factor!=0:
        HW = np.copy(HW)
        Ks = np.copy(Ks)
        HW //= render_factor
        Ks[:, :2, :3] //= render_factor
    rgbs = []
    depths = []
    psnrs = []
    ssims = []
    lpips_alex = []
    lpips_vgg = []
    segmentations = []

    for i, c2w in enumerate(tqdm(render_poses)):
        if slot_idx == -1:

            render_kwargs['segmentation']=True
            keys = ['rgb_marched', 'rgb_direct','depth','segmentation']
        else:
            render_kwargs['segmentation']=False
            keys = ['rgb_marched', 'rgb_direct','depth']

        H, W = HW[i]
        K = Ks[i]
        rays_o, rays_d, viewdirs = VoxelMlp.get_rays_of_a_view(
                H, W, K, c2w, ndc)
        #keys = ['rgb_marched', 'depth']
        rays_o = rays_o.flatten(0,-2)
        rays_d = rays_d.flatten(0,-2)
        viewdirs = viewdirs.flatten(0,-2)
        time_one = test_times[i]*torch.ones_like(rays_o[:,0:1])
        bacth_size=1000
        render_result_chunks = [
            {k: v for k, v in model(ro, rd, vd,ts,slot_idx = slot_idx, **render_kwargs).items() if k in keys}
            for ro, rd, vd ,ts in zip(rays_o.split(bacth_size, 0), rays_d.split(bacth_size, 0), viewdirs.split(bacth_size, 0),time_one.split(bacth_size, 0))
        ]
        render_result = {
            k: torch.cat([ret[k] for ret in render_result_chunks]).reshape(H,W,-1)
            for k in render_result_chunks[0].keys()
        }
        rgb = render_result['rgb_marched'].cpu().numpy()
        depth = render_result['depth'].cpu().numpy()

        rgbs.append(rgb)
        depths.append(depth)

        

        if i==0:
            print('Testing', rgb.shape)

        if render_kwargs.get('segmentation', True):
            seg = render_result['segmentation'].cpu().numpy()
            
            segmentations.append(seg.copy().argmax(-1)[...,None])

        if gt_imgs is not None and render_factor == 0:
            if eval_psnr:
                p = -10. * np.log10(np.mean(np.square(rgb - gt_imgs[i])))
                psnrs.append(p)
            if eval_ssim:
                ssims.append(utils.rgb_ssim(rgb, gt_imgs[i], max_val=1))
            if eval_lpips_alex:
                lpips_alex.append(utils.rgb_lpips(rgb, gt_imgs[i], net_name = 'alex', device = c2w.device))
            if eval_lpips_vgg:
                lpips_vgg.append(utils.rgb_lpips(rgb, gt_imgs[i], net_name = 'vgg', device = c2w.device))

    
    if len(segmentations):
       # seg_vis = []
        for (idx,seg) in enumerate(segmentations):
            seg = gray2rgb(seg[...,0])
            seg.save(os.path.join(savedir, f"seg_{str(idx)}.png"))
    #del segmentations
    
 

    
    

    f1 = open(os.path.join(savedir, 'result.txt'), 'w')


    if len(psnrs):
        if eval_psnr: print('Testing psnr', np.mean(psnrs), '(avg)'), f1.write('Testing psnr:' + str(np.mean(psnrs)) + '(avg)\n')
        if eval_ssim: print('Testing ssim', np.mean(ssims), '(avg)'),  f1.write('Testing ms_ssims' + str(np.mean(ssims)) + '(avg)\n')
        if eval_lpips_vgg: print('Testing lpips (vgg)', np.mean(lpips_vgg), '(avg)'), f1.write('Testing lpips_vgg' + str(np.mean(lpips_vgg)) + '(avg)\n')
        if eval_lpips_alex: print('Testing lpips (alex)', np.mean(lpips_alex), '(avg)'), f1.write('Testing lpips_alex' + str(np.mean(lpips_alex)) + '(avg)\n')

    if savedir is not None:
        print(f'Writing images to {savedir}')
        for i in trange(len(rgbs)):
            rgb8 = utils.to8b(rgbs[i])
            filename = os.path.join(savedir, '{:03d}.png'.format(i))
            imageio.imwrite(filename, rgb8)

        # for i in trange(len(rgb_directs)):
        #     rgb8 = utils.to8b(rgb_directs[i])
        #     filename = os.path.join(savedir, 'direct_{:03d}.png'.format(i))
        #     rgb8 = Image.fromarray(rgb8)
        #     rgb8.save(filename)

    if batch is not None and render_kwargs.get('segmentation', True) and slot_idx==-1:
        segmentations = np.stack(segmentations, axis=0)[None, ..., 0]
        ari, ari_fg = ari_metrics(segmentations, gt_segmentations)
        print('Testing ari/ari-fg', ari, ari_fg)
        f1.write('Testing ari/ari-fg:' + str(ari) + str(ari_fg))


    rgbs = np.array(rgbs)
    depths = np.array(depths)
    return rgbs, depths



def seed_everything():
    '''Seed everything for better reproducibility.
    (some pytorch operation is non-deterministic like the backprop of grid_samples)
    '''
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)


def load_everything(args, cfg):
    '''Load images / poses / camera settings / data split.
    '''
    data_dict = load_data(cfg.data)
    if cfg.data.dataset_type == 'hyper_dataset':
        kept_keys = {
            'data_class',
            'near', 'far',
            'i_train', 'i_val', 'i_test',}
        for k in list(data_dict.keys()):
            if k not in kept_keys:
                data_dict.pop(k)
        return data_dict

    # remove useless field
    kept_keys = {
            'hwf', 'HW', 'Ks', 'near', 'far',
            'i_train', 'i_val', 'i_test', 'irregular_shape',
            'poses', 'render_poses', 'images','times','render_times'}
    for k in list(data_dict.keys()):
        if k not in kept_keys:
            data_dict.pop(k)

    # construct data tensor
    if data_dict['irregular_shape']:
        data_dict['images'] = [torch.FloatTensor(im, device='cpu') for im in data_dict['images']]
    else:
        data_dict['images'] = torch.FloatTensor(data_dict['images'], device = 'cpu')
    data_dict['poses'] = torch.Tensor(data_dict['poses'])
    return data_dict


def compute_bbox_by_cam_frustrm(args, cfg, HW, Ks, poses, i_train, near, far, **kwargs):
    print('compute_bbox_by_cam_frustrm: start')
    xyz_min = torch.Tensor([np.inf, np.inf, np.inf])
    xyz_max = -xyz_min
    for (H, W), K, c2w in zip(HW[i_train], Ks[i_train], poses[i_train]):
        rays_o, rays_d, viewdirs = VoxelMlp.get_rays_of_a_view(
                H=H, W=W, K=K, c2w=c2w,
                ndc=cfg.data.ndc, inverse_y=cfg.data.inverse_y,
                flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y)
        if cfg.data.ndc:
            pts_nf = torch.stack([rays_o+rays_d*near, rays_o+rays_d*far])
        else:
            pts_nf = torch.stack([rays_o+viewdirs*near, rays_o+viewdirs*far])
        xyz_min = torch.minimum(xyz_min, pts_nf.amin((0,1,2)))
        xyz_max = torch.maximum(xyz_max, pts_nf.amax((0,1,2)))
    print('compute_bbox_by_cam_frustrm: xyz_min', xyz_min)
    print('compute_bbox_by_cam_frustrm: xyz_max', xyz_max)
    print('compute_bbox_by_cam_frustrm: finish')
    return xyz_min, xyz_max

def compute_bbox_by_cam_frustrm_hyper(args, cfg,data_class):
    print('compute_bbox_by_cam_frustrm: start')
    xyz_min = torch.Tensor([np.inf, np.inf, np.inf])
    xyz_max = -xyz_min
    for i in tqdm(data_class.i_train[:200]):
        rays_o, _, viewdirs,_ = data_class.load_idx(i,not_dic=True)
        pts_nf = torch.stack([rays_o+viewdirs*data_class.near, rays_o+viewdirs*data_class.far])
        xyz_min = torch.minimum(xyz_min, pts_nf.amin((0,1,2)))
        xyz_max = torch.maximum(xyz_max, pts_nf.amax((0,1,2)))
    print('compute_bbox_by_cam_frustrm: xyz_min', xyz_min)
    print('compute_bbox_by_cam_frustrm: xyz_max', xyz_max)
    print('compute_bbox_by_cam_frustrm: finish')
    return xyz_min, xyz_max


@torch.no_grad()
def compute_bbox_by_coarse_geo(model, thres,grad):
    print('compute_bbox_by_coarse_geo: start')
    eps_time = time.time()
    interp = torch.stack(torch.meshgrid(
        torch.linspace(0, 1, model.density.shape[2]),
        torch.linspace(0, 1, model.density.shape[3]),
        torch.linspace(0, 1, model.density.shape[4]),
    ), -1)
    dense_xyz = model.xyz_min * (1-interp) + model.xyz_max * interp
    density = model.grid_sampler(dense_xyz, model.density)
    alpha = model.activate_density(density.flatten(end_dim=-2)).view(density.shape[0], density.shape[1], density.shape[2])
    mask = (alpha > thres) * (grad > 1e-5)
    active_xyz = dense_xyz[mask]
    xyz_min = active_xyz.amin(0)
    xyz_max = active_xyz.amax(0)
    print('compute_bbox_by_coarse_geo: xyz_min', xyz_min)
    print('compute_bbox_by_coarse_geo: xyz_max', xyz_max)
    eps_time = time.time() - eps_time
    print('compute_bbox_by_coarse_geo: finish (eps time:', eps_time, 'secs)')
    return xyz_min, xyz_max


def scene_rep_reconstruction(args, cfg, cfg_model, cfg_train,writer, data_dict, stage,coarse_ckpt_path=None):

    # init
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # if abs(cfg_model.world_bound_scale - 1) > 1e-9:
    #     xyz_shift = (xyz_max - xyz_min) * (cfg_model.world_bound_scale - 1) / 2
    #     xyz_min -= xyz_shift
    #     xyz_max += xyz_shift

    if cfg.data.dataset_type !='hyper_dataset':
        HW, Ks, near, far, i_train, i_val, i_test, poses, render_poses, images ,times,render_times= [
            data_dict[k] for k in [
                'HW', 'Ks', 'near', 'far', 'i_train', 'i_val', 'i_test', 'poses', 
                'render_poses', 'images',
                'times','render_times'
            ]
        ]
        times = torch.Tensor(times)
        times_i_train = times[i_train].to('cpu' if cfg.data.load2gpu_on_the_fly else device)
    else:
        data_class = data_dict['data_class']
        near = data_class.near
        far = data_class.far
        i_train = data_class.i_train
        i_test = data_class.i_test



    # find whether there is existing checkpoint path
    last_ckpt_path = os.path.join(cfg.basedir, cfg.expname, f'{stage}_last.tar')
    if args.no_reload:
        reload_ckpt_path = None
    elif args.ft_path:
        reload_ckpt_path = args.ft_path
    elif os.path.isfile(last_ckpt_path):
        reload_ckpt_path = last_ckpt_path
    else:
        reload_ckpt_path = None

    # init model and optimizer
    # pdb.set_trace()
    # TODO train from scratch
  
    start = 0
    model_kwargs = copy.deepcopy(cfg_model)
    

  
   
    num_voxels = model_kwargs.pop('num_voxels')

    if len(cfg_train.pg_scale) :
        num_voxels = int(num_voxels / (2**len(cfg_train.pg_scale)))

    ckpt = torch.load(os.path.join(cfg_train.warmup_model_path,"fine_last_n.tar"))

    model_kwargs["max_instances"] = ckpt["model_kwargs"]["max_instances"]

    if cfg.data.dataset_type == 'dnerf':
        model = VoxelMlp_syn.VoxelMlp(
            xyz_min=ckpt['model_kwargs']['xyz_min'], xyz_max=ckpt['model_kwargs']['xyz_max'],
            num_voxels=num_voxels,
            **model_kwargs)
    else:
        model = VoxelMlp.VoxelMlp(
            xyz_min=ckpt['model_kwargs']['xyz_min'], xyz_max=ckpt['model_kwargs']['xyz_max'],
            num_voxels=num_voxels,
            **model_kwargs)
    
    load_dict = {k: v for k,v in ckpt['model_state_dict'].items() if (not 'decoder' in k) and  (not 'featurenet' in k)} # and (not 'feature' in k)}# 
    model.load_state_dict(load_dict,strict = False)
    #model.load_state_dict(ckpt["model_state_dict"],strict = False)
    with torch.no_grad():
        #since we use softmax, we need to scale it
        model.seg_mask *= 5
    model = model.to(device)
    


    

    optimizer = utils.create_optimizer_or_freeze_model(model, cfg_train, global_step=0)

   
    render_kwargs = {
        'near': near,
        'far': far,
        'bg': 1 if cfg.data.white_bkgd else 0,
        'stepsize': cfg_model.stepsize,
    }

    
    def gather_training_rays_hyper(dino = False):
        now_device = 'cpu'  if cfg.data.load2gpu_on_the_fly else device
        if dino:
            N = len(data_class.i_train)
            K = data_class.h*data_class.w
            rgb_tr = torch.zeros([N,K,3], device=now_device)
            rays_o_tr = torch.zeros_like(rgb_tr)
            rays_d_tr = torch.zeros_like(rgb_tr)
            viewdirs_tr = torch.zeros_like(rgb_tr)
            times_tr = torch.ones([N,K,1], device=now_device)
            cam_tr = torch.ones([N,K,1], device=now_device)
            pixels_tr = torch.zeros([N,K,2], device = now_device)
            img_id_tr = torch.ones([N,K,1], device = now_device)
            imsz = []
            top = 0
            for (idx,i) in enumerate(data_class.i_train):
                rays_o, rays_d, viewdirs,rgb, pixels = data_class.load_idx(i,not_dic=True, get_pixel_pos = True)
                n = rgb.shape[0]
                if data_class.add_cam:
                    cam_tr[idx] = cam_tr[idx]*data_class.all_cam[i]
                times_tr[idx] = times_tr[idx]*data_class.all_time[i]
                rgb_tr[idx].copy_(rgb)
                rays_o_tr[idx].copy_(rays_o.to(now_device))
                rays_d_tr[idx].copy_(rays_d.to(now_device))
                viewdirs_tr[idx].copy_(viewdirs.to(now_device))
                pixels_tr[idx].copy_(pixels.to(now_device))
                img_id_tr[idx] *= i
                imsz.append(n)
                
          

        else:
        
            N = len(data_class.i_train)*data_class.h*data_class.w
            rgb_tr = torch.zeros([N,3], device=now_device)
            rays_o_tr = torch.zeros_like(rgb_tr)
            rays_d_tr = torch.zeros_like(rgb_tr)
            viewdirs_tr = torch.zeros_like(rgb_tr)
            times_tr = torch.ones([N,1], device=now_device)
            cam_tr = torch.ones([N,1], device=now_device)
            pixels_tr = torch.zeros([N,2], device = now_device)
            img_id_tr = torch.ones([N,1], device = now_device)
            imsz = []
            top = 0
            for i in data_class.i_train:
                rays_o, rays_d, viewdirs,rgb, pixels = data_class.load_idx(i,not_dic=True, get_pixel_pos = True)
                n = rgb.shape[0]
                if data_class.add_cam:
                    cam_tr[top:top+n] = cam_tr[top:top+n]*data_class.all_cam[i]
                times_tr[top:top+n] = times_tr[top:top+n]*data_class.all_time[i]
                rgb_tr[top:top+n].copy_(rgb)
                rays_o_tr[top:top+n].copy_(rays_o.to(now_device))
                rays_d_tr[top:top+n].copy_(rays_d.to(now_device))
                viewdirs_tr[top:top+n].copy_(viewdirs.to(now_device))
                pixels_tr[top:top+n].copy_(pixels.to(now_device))
                img_id_tr[top:top+n] *= i
                imsz.append(n)
                top += n
            assert top == N
        index_generator = VoxelMlp.batch_indices_generator(len(rgb_tr), cfg_train.N_rand)
        batch_index_sampler = lambda: next(index_generator)
        return rgb_tr, times_tr,cam_tr,rays_o_tr, rays_d_tr, viewdirs_tr, pixels_tr, img_id_tr,imsz, batch_index_sampler

    def gather_training_rays():
        now_device = 'cpu'  if cfg.data.load2gpu_on_the_fly else device
        if data_dict['irregular_shape']:
            rgb_tr_ori = [images[i].to('cpu' if cfg.data.load2gpu_on_the_fly else device) for i in i_train]
        else:
            rgb_tr_ori = images[i_train].to('cpu' if cfg.data.load2gpu_on_the_fly else device)
        #print(rgb_tr_ori.shape,images.shape, i_train.shape, times_i_train)

        if cfg_train.ray_sampler == 'in_maskcache':
            print('cfg_train.ray_sampler =in_maskcache')
            rgb_tr, times_flaten,rays_o_tr, rays_d_tr, viewdirs_tr, imsz = VoxelMlp.get_training_rays_in_maskcache_sampling(
                    rgb_tr_ori=rgb_tr_ori,times=times_i_train,
                    train_poses=poses[i_train],
                    HW=HW[i_train], Ks=Ks[i_train],
                    ndc=cfg.data.ndc, 
                    model=model, render_kwargs=render_kwargs)
            print(rays_o_tr.shape)
        elif cfg_train.ray_sampler == 'flatten':
            print('cfg_train.ray_sampler =flatten')
            rgb_tr, times_flaten,rays_o_tr, rays_d_tr, viewdirs_tr, imsz =VoxelMlp.get_training_rays_flatten(
                rgb_tr_ori=rgb_tr_ori,times=times_i_train,
                train_poses=poses[i_train],
                HW=HW[i_train], Ks=Ks[i_train], ndc=cfg.data.ndc,)
        else:
            print('cfg_train.ray_sampler =random')
            rgb_tr, times_flaten,rays_o_tr, rays_d_tr, viewdirs_tr, imsz = VoxelMlp.get_training_rays(
                rgb_tr=rgb_tr_ori,times=times_i_train,
                train_poses=poses[i_train],
                HW=HW[i_train], Ks=Ks[i_train], ndc=cfg.data.ndc,)
        index_generator = VoxelMlp.batch_indices_generator(len(rgb_tr), cfg_train.N_rand)
        batch_index_sampler = lambda: next(index_generator)
        return rgb_tr,times_flaten, rays_o_tr, rays_d_tr, viewdirs_tr, imsz, batch_index_sampler
                


    if cfg.data.dataset_type !='hyper_dataset':
        rgb_tr,times_flaten, rays_o_tr, rays_d_tr, viewdirs_tr,imsz, batch_index_sampler = gather_training_rays()
    else:
        rgb_tr,times_flaten,cam_tr, rays_o_tr, rays_d_tr, viewdirs_tr,_, _, imsz, batch_index_sampler = gather_training_rays_hyper()
        



    torch.cuda.empty_cache()
    psnr_lst = []
    time0 = time.time()
    global_step = -1

  

    pseudo_segs = sorted(glob.glob(f"{cfg_train.warmup_model_path}/render_train_fine_last/crf_seg/seg*"),key = lambda x: int(x.split("/")[-1].split("_")[-1].split(".")[0]))
    pseudo_segs = [np.array(Image.open(x)).reshape([-1]) for x in  pseudo_segs]
    pseudo_segs = np.concatenate(pseudo_segs, axis = 0) #[N]
    pseudo_segs = torch.from_numpy(pseudo_segs).to("cpu" if cfg.data.load2gpu_on_the_fly else device).long()
    

    if cfg.data.dataset_type == 'hyper_dataset':
        smooth_term = get_linear_noise_func(lr_init=0.01, lr_final=1e-15, lr_delay_mult=0.01, max_steps=15000)
        frame_interval = np.unique(data_class.all_time)
        frame_interval = frame_interval[1] - frame_interval[0]
        print(frame_interval)

    for global_step in trange(1+start, 1+cfg_train.N_iters):

        


        #random sample rays
        if cfg_train.ray_sampler in ['flatten', 'in_maskcache'] or cfg.data.dataset_type =='hyper_dataset':
            sel_i = batch_index_sampler()
            target = rgb_tr[sel_i]
            rays_o = rays_o_tr[sel_i]
            rays_d = rays_d_tr[sel_i]
            viewdirs = viewdirs_tr[sel_i]
            times_sel = times_flaten[sel_i]
            pseudo_seg = pseudo_segs[sel_i]
            if cfg.data.dataset_type == 'hyper_dataset':
                if data_class.add_cam == True:
                    cam_sel = cam_tr[sel_i]
                    cam_sel = cam_sel.to(device)
                    render_kwargs.update({'cam_sel':cam_sel})
                if data_class.use_bg_points == True:
                    sel_idx = torch.randint(data_class.bg_points.shape[0], [cfg_train.N_rand//3])
                    bg_points_sel = data_class.bg_points[sel_idx]
                    bg_points_sel = bg_points_sel.to(device)
                    render_kwargs.update({'bg_points_sel':bg_points_sel})
        elif cfg_train.ray_sampler == 'random':
            sel_b = torch.randint(rgb_tr.shape[0], [cfg_train.N_rand])
            sel_r = torch.randint(rgb_tr.shape[1], [cfg_train.N_rand])
            sel_c = torch.randint(rgb_tr.shape[2], [cfg_train.N_rand])
            target = rgb_tr[sel_b, sel_r, sel_c]
            rays_o = rays_o_tr[sel_b, sel_r, sel_c]
            rays_d = rays_d_tr[sel_b, sel_r, sel_c]
            viewdirs = viewdirs_tr[sel_b, sel_r, sel_c]
            times_sel = times_flaten[sel_b, sel_r, sel_c]

        else:
            raise NotImplementedError

        if cfg.data.load2gpu_on_the_fly:
            target = target.to(device)
            rays_o = rays_o.to(device)
            rays_d = rays_d.to(device)
            viewdirs = viewdirs.to(device)
            times_sel = times_sel.to(device)
            pseudo_seg = pseudo_seg.to(device)
          

       

        # volume rendering
        if cfg.data.dataset_type == 'hyper_dataset':
            times_sel += torch.randn(times_sel.shape[0], 1, device='cuda') * frame_interval * smooth_term(global_step)
        render_result = model(rays_o, rays_d, viewdirs, times_sel, global_step=global_step,is_training = True, **render_kwargs)
      

 
       

        # gradient descent step
        optimizer.zero_grad(set_to_none = True)
        loss = cfg_train.weight_main * F.mse_loss(render_result['rgb_marched'], target)

       
        psnr = utils.mse2psnr(loss.detach())


       
        loss += F.nll_loss(torch.log(render_result['segmentation']+1e-10), pseudo_seg, ignore_index = 0)
        
        if cfg.data.dataset_type =='hyper_dataset':
            if data_class.use_bg_points == True:
                loss = loss+F.mse_loss(render_result['bg_points_delta'],bg_points_sel)
        if cfg_train.weight_entropy_last > 0:
            pout = render_result['alphainv_last'].clamp(1e-6, 1-1e-6)
            entropy_last_loss = -(pout*torch.log(pout) + (1-pout)*torch.log(1-pout)).mean()
            loss += cfg_train.weight_entropy_last * entropy_last_loss
        if cfg_train.weight_rgbper > 0:
            rgbper = (render_result['raw_rgb'] - target[render_result['ray_id']]).pow(2).sum(-1)
            rgbper_loss = (rgbper * render_result['weights'].detach()).sum() / len(rays_o)
            loss += cfg_train.weight_rgbper * rgbper_loss
        
        if cfg_train.weight_distortion > 0:
            n_max = render_result['n_max']
            s = render_result['s']
            w = render_result['weights']
            ray_id = render_result['ray_id']
            loss_distortion = flatten_eff_distloss(w, s.reshape([-1]), 1/n_max, ray_id)
            loss += cfg_train.weight_distortion * loss_distortion

        
        loss.backward()
        

        if global_step<cfg_train.tv_before and global_step>cfg_train.tv_after and global_step%cfg_train.tv_every==0:
            if cfg_train.weight_tv_feature>0:
                model.feature_total_variation_add_grad(
                    cfg_train.weight_tv_feature/len(rays_o), global_step<cfg_train.tv_feature_before)

            
            if cfg_train.weight_tv_density > 0:
                model.density_total_variation_add_grad(
                    cfg_train.weight_tv_density/len(rays_o), global_step<cfg_train.tv_feature_before
                )
           
        optimizer.step()
        psnr_lst.append(psnr.item())
        # update lr
        decay_steps = cfg_train.lrate_decay * 1000
        decay_factor = 0.1 ** (1/decay_steps)
        for i_opt_g, param_group in enumerate(optimizer.param_groups):
            # if not param_group['is_dino']:
            param_group['lr'] = param_group['lr'] * decay_factor         

        # check log & save
        if (global_step+1)%args.i_print==0:
            eps_time = time.time() - time0
            eps_time_str = f'{eps_time//3600:02.0f}:{eps_time//60%60:02.0f}:{eps_time%60:02.0f}'
            tqdm.write(f'scene_rep_reconstruction ({stage}): iter {global_step:6d} / '
                       f'Loss: {loss.item():.9f} / PSNR: {np.mean(psnr_lst):5.2f} / '
                       f'Eps: {eps_time_str}')
            psnr_lst = []

        if (global_step+1)%args.i_weights==0:
            path = os.path.join(cfg.basedir, cfg.expname, f'{stage}_{global_step:06d}.tar')
            render_viewpoints_kwargs = {
                'model': model,
                'ndc': cfg.data.ndc,
                'render_kwargs': {
                    'near': near,
                    'far': far,
                    'bg': 1 if cfg.data.white_bkgd else 0,
                    'stepsize': cfg_model.stepsize,

                    },
                }

            torch.save({
                'global_step': global_step,
                'model_kwargs': model.get_kwargs(),
                'model_state_dict': model.state_dict(),
            }, path)

            testsavedir = os.path.join(cfg.basedir, cfg.expname, f'{global_step}-test')
            if os.path.exists(testsavedir) == False:
                os.makedirs(testsavedir)
            
            if cfg.data.dataset_type != 'hyper_dataset': 
                rgbs,disps = render_viewpoints(
                    render_poses=data_dict['poses'][data_dict['i_train']],
                    HW=data_dict['HW'][data_dict['i_train']],
                    Ks=data_dict['Ks'][data_dict['i_train']],
                    gt_imgs=[data_dict['images'][i].cpu().numpy() for i in data_dict['i_train']],
                    savedir=testsavedir,
                    test_times=data_dict['times'][data_dict['i_train']],
                    eval_psnr=True, eval_ssim=args.eval_ssim, eval_lpips_alex=args.eval_lpips_alex, eval_lpips_vgg=args.eval_lpips_vgg,
                    **render_viewpoints_kwargs)
            else:
                rgbs,disps = render_viewpoints_hyper(
                        data_class=data_class,
                        savedir=testsavedir, all=False, test=True, eval_psnr=True,
                        **render_viewpoints_kwargs)

            testsavedir = os.path.join(cfg.basedir, cfg.expname, f'{global_step}-train')
            if os.path.exists(testsavedir) == False:
                os.makedirs(testsavedir)
            
            if cfg.data.dataset_type != 'hyper_dataset': 
                rgbs,disps = render_viewpoints(
                    render_poses=data_dict['poses'][data_dict['i_train']],
                    HW=data_dict['HW'][data_dict['i_train']],
                    Ks=data_dict['Ks'][data_dict['i_train']],
                    gt_imgs=[data_dict['images'][i].cpu().numpy() for i in data_dict['i_train']],
                    savedir=testsavedir,
                    test_times=data_dict['times'][data_dict['i_train']],
                    eval_psnr=True, eval_ssim=args.eval_ssim, eval_lpips_alex=args.eval_lpips_alex, eval_lpips_vgg=args.eval_lpips_vgg,
                    **render_viewpoints_kwargs)

            else:
                rgbs,disps = render_viewpoints_hyper(
                        data_class=data_class,
                        savedir=testsavedir, all=False, test=False, eval_psnr=True,
                        **render_viewpoints_kwargs)
            print(f'scene_rep_reconstruction ({stage}): saved checkpoints at', path)

   

    if global_step != -1:
        torch.save({
            'global_step': global_step,
            'model_kwargs': model.get_kwargs(),
            'model_state_dict': model.state_dict(),
        }, last_ckpt_path)
        print(f'scene_rep_reconstruction ({stage}): saved checkpoints at', last_ckpt_path)


def train(args, cfg, data_dict=None):

    # init
    print('train: start')
    eps_fine = time.time()
    os.makedirs(os.path.join(cfg.basedir, cfg.expname), exist_ok=True)
    with open(os.path.join(cfg.basedir, cfg.expname, 'args.txt'), 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    cfg.dump(os.path.join(cfg.basedir, cfg.expname, 'config.py'))

    # fine detail reconstruction
    log_path = os.path.join('tb', cfg.basedir+"/logs/"+cfg.expname+'_'+time.strftime('%Y%m%d%H%M%S', time.localtime()))
    os.makedirs(log_path, exist_ok=True)
    writer = SummaryWriter(log_path)

    # xyz_min, xyz_max = compute_bbox_by_cam_frustrm_hyper(args=args, cfg=cfg, data_class = data_dict['data_class'])


    scene_rep_reconstruction(
            args=args, cfg=cfg,
            cfg_model=cfg.fine_model_and_render, cfg_train=cfg.fine_train,
            data_dict=data_dict, stage='fine', writer=writer)#, xyz_min = xyz_min, xyz_max = xyz_max)

    eps_fine = time.time() - eps_fine
    eps_time_str = f'{eps_fine//3600:02.0f}:{eps_fine//60%60:02.0f}:{eps_fine%60:02.0f}'
    print('train: fine detail reconstruction in', eps_time_str)



            

# 


if __name__=='__main__':

    # load setup
    parser = config_parser()
    args = parser.parse_args()
    cfg = mmcv.Config.fromfile(args.config)

    # init enviroment
    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    seed_everything()
    data_dict = None
    # pdb.set_trace()
    # load images / poses / camera settings / data split
    data_dict= load_everything(args=args, cfg=cfg)

    

    # train
    if not args.render_only:
        train(args, cfg, data_dict)

    # load model for rendring
    if args.render_test or args.render_train or args.render_video:
        if args.ft_path:
            ckpt_path = args.ft_path
        else:
            ckpt_path = os.path.join(cfg.basedir, cfg.expname, 'fine_last.tar')
        ckpt_name = ckpt_path.split('/')[-1][:-4]
     
        model_class = VoxelMlp_syn.VoxelMlp if cfg.data.dataset_type=='dnerf' else VoxelMlp.VoxelMlp
        model = utils.load_model(model_class, ckpt_path).to(device)
        stepsize = cfg.fine_model_and_render.stepsize
        render_viewpoints_kwargs = {
            'model': model,
            'ndc': cfg.data.ndc,
            'render_kwargs': {
                'near': data_dict['near'],
                'far': data_dict['far'],
                'bg': 1 if cfg.data.white_bkgd else 0,
                'stepsize': stepsize,
                'render_depth': True,
                'segmentation': True
            }}
  
    

    num_slots = model.seg_mask.shape[1] if args.per_slot else 0
        # render trainset and eval
    if args.render_train:
       
        if args.eval_ari and cfg.data.dataset_type == 'dnerf': 
            import glob
            from PIL import Image
            gt_seg = sorted(glob.glob(f'{cfg.data.datadir}/train/segmentation*.png'))
            gt_seg = [np.array(Image.open(x).resize((256,256),Image.NEAREST)) for x in gt_seg]
            gt_seg = np.stack(gt_seg, 0)[None,...]

            video = sorted(glob.glob(f'{cfg.data.datadir}/train/[0-9][0-9][0-9].png'))
            video = [np.array(Image.open(x).convert('RGB').resize((256,256),Image.BICUBIC)) for x in video]
            video = np.stack(video, 0)[None,...].astype(np.uint8)
            #video = (video / 255.0).astype(np.float64)
            batch = (gt_seg,video)

            print(gt_seg.shape,video.shape)
        else:
            batch = None   

                
        
        
        for i in range(-1, num_slots):
            if i == -1:
                testsavedir = os.path.join(cfg.basedir, cfg.expname, f'render_train_{ckpt_name}')

            else:
                testsavedir = os.path.join(cfg.basedir, cfg.expname, f'render_train_{ckpt_name}_slot{i}')
            os.makedirs(testsavedir, exist_ok=True)
            with torch.no_grad():
                if cfg.data.dataset_type == 'hyper_dataset':
                    rgbs, depths = render_viewpoints_hyper(
                        data_class=data_dict['data_class'],
                        savedir=testsavedir, all =True, test=False,
                        eval_psnr=True,slot_idx = i,
                        **render_viewpoints_kwargs)
                else:
                    rgbs,depths = render_viewpoints(
                        render_poses=data_dict['poses'][data_dict['i_train']],
                        HW=data_dict['HW'][data_dict['i_train']],
                        Ks=data_dict['Ks'][data_dict['i_train']],
                        gt_imgs=[data_dict['images'][i].cpu().numpy() for i in data_dict['i_train']],
                        savedir=testsavedir,
                        test_times=data_dict['times'][data_dict['i_train']],
                        eval_psnr=True, eval_ssim=True, eval_lpips_alex=True, eval_lpips_vgg=True,
                        slot_idx = i,batch = batch,
                        **render_viewpoints_kwargs)
        
            rgb_video = 'video.rgb.mp4'
            depth_video = 'video.depth.mp4'
            shape_video = 'video.shape.mp4'
            imageio.mimwrite(os.path.join(testsavedir, rgb_video), utils.to8b(rgbs), fps=10, quality=8)
            imageio.mimwrite(os.path.join(testsavedir, depth_video), utils.to8b(1 - depths / np.max(depths)), fps=10, quality=8)
        # imageio.mimwrite(os.path.join(testsavedir, shape_video), utils.to8b(shapes), fps=10, quality=8)


    # render testset and eval
    if args.render_test:
       
        # num_views = int(len(data_dict['i_test']) / imgs_perview) 
        if args.eval_ari and cfg.data.dataset_type == 'dnerf': 
            # video = torch.from_numpy(np.load(os.path.join(cfg.data.datadir, 'test', "video.npy"))).type(torch.float32).to(device,non_blocking = True).unsqueeze(0)
        
            import glob
            from PIL import Image
            gt_seg = sorted(glob.glob(f'{cfg.data.datadir}/test/segmentation*.png'))
            gt_seg = [np.array(Image.open(x).resize((256,256),Image.NEAREST)) for x in gt_seg]
            gt_seg = np.stack(gt_seg, 0)[None,...]

            video = sorted(glob.glob(f'{cfg.data.datadir}/test/[0-9][0-9][0-9].png'))
            video = [np.array(Image.open(x).convert('RGB').resize((256,256),Image.BICUBIC)) for x in video]
            video = np.stack(video, 0)[None,...].astype(np.uint8)
            #video = (video / 255.0).astype(np.float64)
            batch = (gt_seg,video)
        else:
            batch = None

      
        for i in range(-1, num_slots):
            if i == -1:
                testsavedir = os.path.join(cfg.basedir, cfg.expname, f'render_test_{ckpt_name}')
            else:
                testsavedir = os.path.join(cfg.basedir, cfg.expname, f'render_test_{ckpt_name}_slot{i}')
            os.makedirs(testsavedir, exist_ok=True)
            with torch.no_grad():
                if cfg.data.dataset_type == "hyper_dataset":
                    rgbs, depths = render_viewpoints_hyper(
                        data_class=data_dict['data_class'],
                        savedir=testsavedir, all=(i==-1), test=True,
                        eval_psnr=True,slot_idx = i, 
                        **render_viewpoints_kwargs)
                else:
                    rgbs,depths = render_viewpoints(
                        render_poses=data_dict['poses'][data_dict['i_test']],
                        HW=data_dict['HW'][data_dict['i_test']],
                        Ks=data_dict['Ks'][data_dict['i_test']],
                        gt_imgs=[data_dict['images'][i].cpu().numpy() for i in data_dict['i_test']],
                        savedir=testsavedir,
                        test_times=data_dict['times'][data_dict['i_test']],
                        eval_psnr=True, eval_ssim=True, eval_lpips_alex=True, eval_lpips_vgg=True,
                         slot_idx = i,batch = batch,
                        **render_viewpoints_kwargs)
            rgb_video = 'video.rgb.mp4'
            depth_video = 'video.depth.mp4'
            shape_video = 'video.shape.mp4'
            imageio.mimwrite(os.path.join(testsavedir, rgb_video), utils.to8b(rgbs), fps=10, quality=8)
            imageio.mimwrite(os.path.join(testsavedir, depth_video), utils.to8b(1 - depths / np.max(depths)), fps=10, quality=8)


    
    print('Done')

