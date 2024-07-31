import time

import os, sys, copy, glob, json, time, random, argparse
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
from lib import voxelMlp_hyper as VoxelMlp
from lib import voxelMlp_syn as VoxelMlp_syn

from lib.load_data import load_data
from post_process import post_process
from pytorch_msssim import ms_ssim
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import pdb
import glob
from torch.utils.data import Dataset
from lib.utils import get_linear_noise_func
from lib.voxelMlp_hyper import sin_emb
from torch_efficient_distloss import flatten_eff_distloss





class DinoDataSet(Dataset):
    def __init__(self, data_path, train_ids,load_to_memory = False):  #0.07
        self.names = [f"{data_path}/{str(id).zfill(5)}.pth" for id in train_ids]
        self.id2indx = np.zeros([max(train_ids)+1]) - 1  
        self.id2indx[np.array(train_ids)] = np.arange(len(train_ids))  
        self.id2indx = self.id2indx.astype(np.int32)

        self.load_to_memory = load_to_memory
        if load_to_memory:
            self.dinos = [torch.load(x) for x in self.names]
        #print(self.names)

    def __len__(self):
        return len(self.names)


    def __getitem__(self, index):
        index = self.id2indx[index]
        assert index >=0
        if self.load_to_memory:
            return self.dinos[index].float()
        return torch.load(self.names[index]).float()

# def get_dino_loss(pred_dino, pixels, img_id, dino_features, time_id, model):
def get_dino_loss(pred_dino, pixels, img_id,  time_id, model):
    #pred_dino [N,16]
    #pixels [N, 2]
    #img_id[N,1]
    img_id = img_id.squeeze(-1).long()
    loss = 0
    N = 4096
    count = 0

    loss_func = torch.nn.L1Loss()
    #loss_func = SelectiveMSELoss()

    assert torch.unique(img_id).shape[0] ==1
    i = img_id[0]
    # if i!=0:
    #     return 0
    pixel = pixels[img_id == i,:][:N,:]
    if pixel.shape[0] < 2:
        return loss

    dino_feature = dino_dataset[i].to(pixel.device)
    #dino_feature = dino_features[i].to(pixel.device).float()
    dino_sample = F.grid_sample( dino_feature,pixel.reshape([1,1,-1,2]))
    dino_sample = dino_sample.squeeze(0).squeeze(1).permute(1,0) #[N,C]
    
    dino = pred_dino[img_id == i,:][:N]

    dino = model.dino_mlp(dino)
  
    loss = loss_func(dino, dino_sample.detach())
  
    return loss

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
    parser.add_argument("--load_to_memory", action='store_true')
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
    parser.add_argument("--thresh",   type=float, default=1e-3)
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
def render_viewpoints_hyper(model, data_class, ndc, render_kwargs, test=True, 
                                all=True, savedir=None, eval_psnr=False,slot_idx = -1, mask = None,eval_lpips_alex= True, eval_lpips_vgg=True, eval_ssim=True):
    
    rgbs = []
    rgbs_gt =[]
    rgbs_tensor =[]
    rgbs_gt_tensor =[]
    depths = []
    dinos = []
    psnrs = []
    segmentations = []
    ms_ssims =[]
    lpips_alex = []
    lpips_vgg = []
    slots_probs = []
    segmentations_raw = []
    rgb_directs=[]

    #dino_depth = []

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
        
        if mask is not None:
            render_kwargs['segmentation']=True
            keys = ['rgb_marched', 'rgb_direct','depth','segmentation',"dino","slots_prob"]
        else:
            render_kwargs['segmentation']=False
            keys = ['rgb_marched', 'rgb_direct','depth',"dino","slots_prob"]
        
        rays_o, rays_d, viewdirs,rgb_gt = data_class.load_idx(i, not_dic=True)
        time_one = data_class.all_time[i]*torch.ones_like(rays_o[:,0:1])
        cam_one = data_class.all_cam[i]*torch.ones_like(rays_o[:,0:1])
        bacth_size = 3000
        render_result_chunks = [
            {k: v for k, v in model(ro, rd, vd,ts, cams,slot_idx = slot_idx,mask =mask,**render_kwargs).items() if k in keys}
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
        dino = render_result["dino"].cpu().numpy()
        slots_prob = render_result["slots_prob"].cpu().numpy()
        
        rgb_direct = render_result['rgb_direct'].cpu().numpy()
        rgb_directs.append(rgb_direct)


      

        rgbs.append(rgb)
        depths.append(depth)
        rgbs_gt.append(rgb_gt)
        slots_probs.append(slots_prob.argmax(-1))
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
        for (idx,seg) in enumerate(segmentations):
            seg = gray2rgb(seg[...,0])
            seg.save(os.path.join(savedir, f"seg_{str(idx)}.png"))
    del segmentations
    
    if len(segmentations_raw):
        for (i,x) in enumerate(segmentations_raw):
            np.save(os.path.join(savedir, f"seg_raw_{str(i).zfill(3)}.npy"),x)

    del segmentations_raw

    

    if len(slots_probs):
        for (idx,seg) in enumerate(slots_probs):
            gray2rgb(seg).save(os.path.join(savedir, f"slots_prob_{str(idx)}.png"))

    del slots_probs
    

    
    if eval_ssim: 
        rgbs_tensor = torch.stack(rgbs_tensor,0)
        rgbs_gt_tensor = torch.stack(rgbs_gt_tensor,0)
        ms_ssims = ms_ssim(rgbs_gt_tensor, rgbs_tensor, data_range=1, size_average=True )

    f1 = open(os.path.join(savedir, 'result.txt'), 'w')
    if len(psnrs):
        print('Testing psnr', np.mean(psnrs), '(avg)')
        f1.write('Testing psnr:' + str(np.mean(psnrs)) + '(avg)\n')
        np.save(os.path.join(savedir, 'psnr.npy'), np.array(psnrs))

        
        if eval_ssim: 
            print('Testing ms_ssims', ms_ssims, '(avg)')
            f1.write('Testing ms_ssims' + str(ms_ssims) + '(avg)\n')
            np.save(os.path.join(savedir, 'ssim.npy'), ms_ssims)

        if eval_lpips_alex:
            print('Testing lpips_alex', np.mean(lpips_alex), '(avg)')
            f1.write('Testing lpips_alex' + str(np.mean(lpips_alex)) + '(avg)\n')
        if eval_lpips_vgg:
            print('Testing lpips_vgg', np.mean(lpips_vgg), '(avg)')
            f1.write('Testing lpips_vgg' + str(np.mean(lpips_vgg)) + '(avg)\n')

    f1.close()
    
    if len(psnrs):
        print('Testing psnr', np.mean(psnrs), '(avg)')


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
                      gt_imgs=None, savedir=None, test_times=None, render_factor=0, eval_psnr=False,mask = None, slot_idx = -1,
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
    segmentations_raw = []
    segmentations = []
    rgb_directs=[]
    

    for i, c2w in enumerate(tqdm(render_poses)):
        if mask is not None and slot_idx == -1:

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
            {k: v for k, v in model(ro, rd, vd,ts,slot_idx = slot_idx,mask =mask, **render_kwargs).items() if k in keys}
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

       
        rgb_direct = render_result['rgb_direct'].cpu().numpy()
        rgb_directs.append(rgb_direct)



        if i==0:
            print('Testing', rgb.shape)

        if render_kwargs.get('segmentation', True):
            seg = render_result['segmentation'].cpu().numpy()
            segmentations_raw.append(seg.copy())
            
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
    
    if len(segmentations_raw):
        for (i,x) in enumerate(segmentations_raw):
            np.save(os.path.join(savedir, f"seg_raw_{str(i).zfill(3)}.npy"),x)

    del segmentations_raw

    

    
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

        for i in trange(len(rgb_directs)):
            rgb8 = utils.to8b(rgb_directs[i])
            filename = os.path.join(savedir, 'direct_{:03d}.png'.format(i))
            rgb8 = Image.fromarray(rgb8)
            rgb8.save(filename)

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
                ndc=cfg.data.ndc)
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




def scene_rep_reconstruction(args, cfg, cfg_model, cfg_train,writer,xyz_min, xyz_max, data_dict, stage,coarse_ckpt_path=None):

    # init
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if abs(cfg_model.world_bound_scale - 1) > 1e-9:
        xyz_shift = (xyz_max - xyz_min) * (cfg_model.world_bound_scale - 1) / 2
        xyz_min -= xyz_shift
        xyz_max += xyz_shift

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
   
  
    start = 0
    model_kwargs = copy.deepcopy(cfg_model)
    

  
   
    num_voxels = model_kwargs.pop('num_voxels')

    if len(cfg_train.pg_scale) :
        num_voxels = int(num_voxels / (2**len(cfg_train.pg_scale)))


    if cfg.data.dataset_type == 'dnerf':
        model = VoxelMlp_syn.VoxelMlp(
            xyz_min=xyz_min, xyz_max=xyz_max,
            num_voxels=num_voxels,
            **model_kwargs)
    else:
        model = VoxelMlp.VoxelMlp(
            xyz_min=xyz_min, xyz_max=xyz_max,
            num_voxels=num_voxels,
            **model_kwargs)
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
        _,times_flaten_dino ,_, rays_o_tr_dino, rays_d_tr_dino, _,pixels_tr_dino, img_id_tr_dino, _,_ = gather_training_rays_hyper(dino = True)


    
    torch.cuda.empty_cache()
    psnr_lst = []
    time0 = time.time()
    global_step = -1

    
   # print(cfg.data)
    global dino_dataset
    if cfg.data.dataset_type != 'hyper_dataset':
        dino_dataset = None
        if model_kwargs.maskout_near_cam_vox:
            print("maskout near vox")
            model.maskout_near_cam_vox(poses[i_train,:3,3], near)
    else:
        dino_dataset = DinoDataSet(f"{data_class.datadir}/dino_featup_v2", i_train,load_to_memory = args.load_to_memory)
        model.dino_mlp = nn.Linear(model.dino_channel, dino_dataset[i_train[0]].shape[1])
    
    if cfg.data.dataset_type == 'hyper_dataset':
        smooth_term = get_linear_noise_func(lr_init=0.1, lr_final=1e-15, lr_delay_mult=0.01, max_steps=15000)
        frame_interval = np.unique(data_class.all_time)
        frame_interval = frame_interval[1] - frame_interval[0]
        print("frame_interval",frame_interval)

    for global_step in trange(start, cfg_train.N_iters):

        if global_step == args.step_to_half:
            model.feature.data=model.feature.data.half()
            model.density.data = model.density.data.half()
        
        if global_step in cfg_train.pg_scale:
            n_rest_scales = len(cfg_train.pg_scale)-cfg_train.pg_scale.index(global_step)-1
            cur_voxels = int(cfg_model.num_voxels / (2**n_rest_scales))
            
            model.scale_volume_grid(cur_voxels)
            
            optimizer = utils.create_optimizer_or_freeze_model(model, cfg_train, global_step=0)

           


        #random sample rays
        if cfg_train.ray_sampler in ['flatten', 'in_maskcache'] or cfg.data.dataset_type =='hyper_dataset':
            sel_i = batch_index_sampler()
            target = rgb_tr[sel_i]
            rays_o = rays_o_tr[sel_i]
            rays_d = rays_d_tr[sel_i]
            viewdirs = viewdirs_tr[sel_i]
            times_sel = times_flaten[sel_i]
            
            

            if cfg.data.dataset_type == 'hyper_dataset':
                img_i =np.random.choice(pixels_tr_dino.shape[0])
                sel_r = torch.randint(pixels_tr_dino.shape[1], [cfg_train.N_rand]).to(pixels_tr_dino.device)
                rays_o_dino = rays_o_tr_dino[img_i, sel_r]#, sel_c]
                rays_d_dino = rays_d_tr_dino[img_i, sel_r]#, sel_c]
                times_sel_dino = times_flaten_dino[img_i,sel_r]
                pixels_dino = pixels_tr_dino[img_i,sel_r]
                img_id_dino = img_id_tr_dino[img_i,sel_r]

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
           
            img_num = min(max(int(cfg_train.start_ratio * rgb_tr.shape[0]), int(global_step / cfg_train.increase_until * rgb_tr.shape[0])), rgb_tr.shape[0])
            sel_b = torch.randint(img_num, [cfg_train.N_rand])
            
            sel_r = torch.randint(rgb_tr.shape[1], [cfg_train.N_rand])
            sel_c = torch.randint(rgb_tr.shape[2], [cfg_train.N_rand])
            target = rgb_tr[sel_b, sel_r, sel_c]
            rays_o = rays_o_tr[sel_b, sel_r, sel_c]
            rays_d = rays_d_tr[sel_b, sel_r, sel_c]
            viewdirs = viewdirs_tr[sel_b, sel_r, sel_c]
            times_sel = times_flaten[sel_b, sel_r, sel_c]
        elif cfg_train.ray_sampler == 'sequential_1im_fixed':

            img_i = global_step % rgb_tr.shape[0]

            img_i = torch.tensor(img_i,device = "cpu" if cfg.data.load2gpu_on_the_fly else device)
            sel_r = torch.randint(rgb_tr.shape[1], [cfg_train.N_rand]).to("cpu" if cfg.data.load2gpu_on_the_fly else device)
            sel_c = torch.randint(rgb_tr.shape[2], [cfg_train.N_rand]).to("cpu" if cfg.data.load2gpu_on_the_fly else device)
            target = rgb_tr[img_i, sel_r, sel_c]
            rays_o = rays_o_tr[img_i, sel_r, sel_c]
            rays_d = rays_d_tr[img_i, sel_r, sel_c]
            viewdirs = viewdirs_tr[img_i, sel_r, sel_c]
            times_sel = times_flaten[img_i, sel_r, sel_c]
       

        else:
            raise NotImplementedError

        if cfg.data.load2gpu_on_the_fly:
            target = target.to(device)
            rays_o = rays_o.to(device)
            rays_d = rays_d.to(device)
            viewdirs = viewdirs.to(device)
            times_sel = times_sel.to(device)
            if cfg.data.dataset_type == 'hyper_dataset':
                rays_o_dino = rays_o_dino.to(device)
                rays_d_dino = rays_d_dino.to(device)
                times_sel_dino = times_sel_dino.to(device)
                pixels_dino = pixels_dino.to(device)  #[N,2]
                img_id_dino = img_id_dino.to(device)   #[N,1]

        # volume rendering
        if cfg.data.dataset_type == 'hyper_dataset':
            times_sel += torch.randn(times_sel.shape[0], 1, device='cuda') * frame_interval * smooth_term(global_step)
        render_result = model(rays_o, rays_d, viewdirs, times_sel, global_step=global_step,is_training = True, **render_kwargs)
        if cfg.data.dataset_type == 'hyper_dataset' and global_step >= cfg_train.dino_start_training:
            render_result_dino = model.forward_dino(rays_o_dino, rays_d_dino, times_sel_dino, global_step=global_step,is_training = True, **render_kwargs)

 

        loss_dino = 0
        loss_entropy = 0
     
        if cfg.data.dataset_type == 'hyper_dataset' and global_step >= cfg_train.dino_start_training:
            loss_dino = get_dino_loss(render_result_dino["dino"], pixels_dino, img_id_dino, i_train[global_step%len(i_train)], model)
            writer.add_scalar("dino_loss", loss_dino.item(), global_step)
        

            loss_entropy = render_result_dino["loss_entropy"]
            writer.add_scalar("entropy loss", loss_entropy.item(),global_step)

       

        # gradient descent step
        optimizer.zero_grad(set_to_none = True)
        loss = cfg_train.weight_main * F.mse_loss(render_result['rgb_marched'], target)

       
        psnr = utils.mse2psnr(loss.detach())

        if cfg.data.dataset_type == 'hyper_dataset' and global_step >= cfg_train.dino_start_training:
            loss += cfg_train.weight_dino *loss_dino
            loss += cfg_train.weight_entropy*loss_entropy
        loss += render_result["cycle_loss"]

        
        loss += cfg_train.weight_main * F.mse_loss(render_result['rgb_direct'], target)
       
        
        if cfg.data.dataset_type =='hyper_dataset':
            if data_class.use_bg_points == True:
                loss = loss+F.mse_loss(render_result['bg_points_delta'],bg_points_sel)
        
        if cfg_train.weight_entropy_last > 0:
            pout = render_result['alphainv_last'].clamp(1e-6, 1-1e-6)
            entropy_last_loss = -(pout*torch.log(pout) + (1-pout)*torch.log(1-pout))#.mean()
            loss += cfg_train.weight_entropy_last * entropy_last_loss.mean()


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

            if cfg_train.weight_tv_dino>0:
                model.dino_total_variation_add_grad(
                    cfg_train.weight_tv_dino/len(rays_o), global_step<cfg_train.tv_feature_before)
            
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
            param_group['lr'] = param_group['lr'] * decay_factor    
    
        if (global_step+1)%args.i_print==0:
            eps_time = time.time() - time0
            eps_time_str = f'{eps_time//3600:02.0f}:{eps_time//60%60:02.0f}:{eps_time%60:02.0f}'
            tqdm.write(f'scene_rep_reconstruction ({stage}): iter {global_step:6d} / '
                       f'Loss: {loss.item():.9f} / PSNR: {np.mean(psnr_lst):5.2f} / '
                       f'Eps: {eps_time_str}')
            psnr_lst = []

        if (global_step+1)%args.i_weights==0:
            path = os.path.join(cfg.basedir, cfg.expname, f'{stage}_{global_step:06d}.tar')
            # model.motion = torch.nn.Parameter(torch.cat(model.motion_list))     
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
            },path)

            testsavedir = os.path.join(cfg.basedir, cfg.expname, f'{global_step}-test')
            if os.path.exists(testsavedir) == False:
                os.makedirs(testsavedir)
            
            if cfg.data.dataset_type != 'hyper_dataset': 
                rgbs,disps = render_viewpoints(
                    render_poses=data_dict['poses'][data_dict['i_test']],
                    HW=data_dict['HW'][data_dict['i_test']],
                    Ks=data_dict['Ks'][data_dict['i_test']],
                    gt_imgs=[data_dict['images'][i].cpu().numpy() for i in data_dict['i_test']],
                    savedir=testsavedir,
                    test_times=data_dict['times'][data_dict['i_test']],
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

    if cfg.data.dataset_type == 'hyper_dataset':
        xyz_min, xyz_max = compute_bbox_by_cam_frustrm_hyper(args = args, cfg = cfg,data_class = data_dict['data_class'])
    else:
        xyz_min, xyz_max = compute_bbox_by_cam_frustrm(args = args, cfg = cfg, **data_dict)
  

    scene_rep_reconstruction(
            args=args, cfg=cfg,
            cfg_model=cfg.fine_model_and_render, cfg_train=cfg.fine_train,
            data_dict=data_dict, stage='fine', writer=writer, xyz_min = xyz_min, xyz_max = xyz_max)

    eps_fine = time.time() - eps_fine
    eps_time_str = f'{eps_fine//3600:02.0f}:{eps_fine//60%60:02.0f}:{eps_fine%60:02.0f}'
    print('train: fine detail reconstruction in', eps_time_str)




def get_importance(args, cfg, data_dict, model):
    stepsize = cfg.fine_model_and_render.stepsize
    
    render_kwargs={
        'near': data_dict['near'],
        'far': data_dict['far'],
        'bg': 1 if cfg.data.white_bkgd else 0,
        'stepsize': stepsize,
        'render_depth': False,
    }
        
    
    ndc = cfg.data.ndc
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    
    print("xyzmin,max",model.xyz_min,model.xyz_max)

    eps_render = time.time()
    pseudo_grid = torch.ones_like(model.density)
    pseudo_grid.requires_grad = True

    for i, c2w in enumerate(tqdm(data_dict['poses'][data_dict['i_train']])):

        H, W = data_dict['HW'][data_dict['i_train']][i]
        K = data_dict['Ks'][data_dict['i_train']][i]
        
        rays_o, rays_d, viewdirs = VoxelMlp.get_rays_of_a_view(
                H, W, K, c2w, ndc,)
        keys = ['rgb_marched']
        rays_o = rays_o.flatten(0,-2)
        rays_d = rays_d.flatten(0,-2)
        viewdirs = viewdirs.flatten(0,-2)

        frame_time = data_dict['times'][data_dict['i_train']][i] * torch.ones_like(rays_o[:,0:1])
        frame_time = frame_time.to(device)
    

       
        bacth_size = 1000

        for ro, rd, vd in zip(rays_o.split(1024, 0), rays_d.split(1024, 0), viewdirs.split(1024, 0)):
            ret = model.forward_imp(ro, rd, vd, frame_time, i, start=(frame_time==0), pseudo_grid = pseudo_grid, training_flag=False, stc_data=False,**render_kwargs)

            if (ret['weights'].size(0) !=0) and (ret['sampled_pseudo_grid'].size(0) !=0):
                (ret['weights'].detach()*ret['sampled_pseudo_grid']).sum().backward()
           


    model.density.grad = None
    return pseudo_grid.grad.clone()

def get_importance_hyper(args, cfg, data_class, model):
    stepsize = cfg.fine_model_and_render.stepsize
    
    render_kwargs={
        'near': data_dict['near'],
        'far': data_dict['far'],
        'bg': 1 if cfg.data.white_bkgd else 0,
        'stepsize': stepsize,
        'render_depth': False,
        'num_slots':cfg.fine_model_and_render.max_instances,
    }
        
    
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    
    print("xyzmin,max",model.xyz_min,model.xyz_max)
    idx = data_class.i_train[:]
    eps_render = time.time()
    pseudo_grid = torch.ones([1,1,*model.feature.shape[2:]],device = model.feature.device)
    pseudo_grid.requires_grad = True
    for i in tqdm(idx[:200]):

        rays_o, rays_d, viewdirs,rgb_gt = data_class.load_idx(i, not_dic=True)
        keys = ['rgb_marched', 'depth']
        time_one = data_class.all_time[i]*torch.ones_like(rays_o[:,0:1])
        cam_one = data_class.all_cam[i]*torch.ones_like(rays_o[:,0:1])
        bacth_size = 1000

        for ro, rd, vd ,ts,cams in zip(rays_o.split(bacth_size, 0), rays_d.split(bacth_size, 0),
                                             viewdirs.split(bacth_size, 0),time_one.split(bacth_size, 0),cam_one.split(bacth_size, 0)):
            ret = model.forward_imp(ro, rd, vd,ts, cams,pseudo_grid = pseudo_grid,**render_kwargs)
            if (ret['weights'].size(0) !=0) and (ret['sampled_pseudo_grid'].size(0) !=0):
                (ret['weights'].detach()*ret['sampled_pseudo_grid']).sum().backward()
            i += 1


    model.feature.grad = None
    return pseudo_grid.grad.clone()
            

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

            },
        }

    frame_times = []
    if cfg.data.dataset_type == 'hyper_dataset':
        for idx in data_dict['data_class'].i_train:
            frame_times.append(data_dict['data_class'].all_time[idx])
    else:
        frame_times = data_dict['times'][data_dict['i_train']].tolist()
    frame_times = list(set(frame_times))
    frame_times = sorted(frame_times)


    print("note, use 100 frame here")
    frame_times = frame_times[::max(1,len(frame_times)//100)]
    dx = model.get_dynamics(frame_times = frame_times)  #[59,3,X,Y,Z]

       
    world_pos = (model.world_pos[0].cpu() + dx[0]).numpy()  #[3,x,y,z]



    dx = (dx - dx[0:1]).numpy() 
    voxel_size = model.voxel_size.item()
    


    rgb = model.get_rgb().detach().squeeze(0).cpu().numpy()


    path = os.path.join(cfg.basedir, cfg.expname, 'importance.pth')
    if os.path.exists(path):
        imp = torch.load(path)
    else:
        if cfg.data.dataset_type != 'hyper_dataset':
            imp = get_importance(args, cfg, data_dict, model)
        else:
            imp = get_importance_hyper(args, cfg, data_dict['data_class'], model)
        torch.save(imp,path)

    

    importance = imp.clone().cuda().reshape([-1])
    importance_prune =  0.9999 if cfg.data.dataset_type == 'hyper_dataset' else 0.999
    percent_sum = importance_prune
    vals,idx = sorted_importance = torch.sort(importance+(1e-6))
    cumsum_val = torch.cumsum(vals, dim=0)
    split_index = ((cumsum_val/vals.sum()) > (1-percent_sum)).nonzero().min()
    split_val_nonprune = vals[split_index]
    percent_point = (importance+(1e-6)>= vals[split_index]).sum()/importance.numel()
    print(f'{percent_point*100:.2f}% of most important points contribute over {(percent_sum)*100:.2f}% importance ')
    if cfg.data.dataset_type !='hyper_dataset':
        with torch.no_grad():
            model.density[imp < split_val_nonprune] = -5

    imp = imp[0,0].cpu().numpy()

    if cfg.data.dataset_type !='hyper_dataset':
        slots_prob = None
        num_slots = cfg.fine_model_and_render.max_instances
    else:
        dino = model.dino[0].permute(1,2,3,0)
        slots = model.slots
        tem =F.softplus(model.temperature)
        slots_prob =F.softmax(dino @ slots.permute(1,0) /tem,dim=-1)  #[H,W,D,K]
        if slots_prob.shape[:3] != model.density.shape[2:]:
            slots_prob = slots_prob.permute(3, 0, 1, 2)  #[K,H,W,D]
            slots_prob = F.interpolate(slots_prob.unsqueeze(0), size=model.density.shape[2:], mode='trilinear', align_corners = True)
            slots_prob = slots_prob.squeeze(0)
            slots_prob = slots_prob.permute(1, 2, 3, 0)
        slots_prob = slots_prob.detach().cpu().numpy()
        num_slots = slots.shape[0]
   
  
        
    if args.per_slot:
        masks= post_process(model.density, model.act_shift, num_slots, world_pos, dx, rgb, split_val_nonprune.item(),args.thresh,grad = imp,
                            dino_label = slots_prob,voxel_size = voxel_size, dataset_type = cfg.data.dataset_type, cluster_args = cfg.cluster if hasattr(cfg,"cluster") else None).to(model.density.device).float()
        
        num_slots = masks.shape[1]

        path = os.path.join(cfg.basedir, cfg.expname, 'fine_last_n.tar')
        kwargs = model.get_kwargs()
        kwargs['max_instances'] = num_slots
        with torch.no_grad():
            masks_ = masks.to(model.density.device)
            state_dict = model.state_dict()
            state_dict['seg_mask'] = masks_
            
            
        torch.save({
            'model_kwargs': kwargs,
            'model_state_dict': state_dict,
        }, path)


        
    else:
        num_slots = 0
        masks = None
   
    if args.render_train:
        if args.eval_ari and cfg.data.dataset_type =='dnerf': 
            import glob
            from PIL import Image
            gt_seg = sorted(glob.glob(f'{cfg.data.datadir}/train/segmentation*.png'))
            gt_seg = [np.array(Image.open(x).resize((512,512),Image.NEAREST)) for x in gt_seg]
            gt_seg = np.stack(gt_seg, 0)[None,...]

            video = sorted(glob.glob(f'{cfg.data.datadir}/train/[0-9][0-9][0-9].png'))
            video = [np.array(Image.open(x).convert('RGB').resize((512,512),Image.BICUBIC)) for x in video]
            video = np.stack(video, 0)[None,...].astype(np.uint8)
            batch = (gt_seg,video)

            print(gt_seg.shape,video.shape)
        else:
            batch = None   

                
        
      
      
        testsavedir = os.path.join(cfg.basedir, cfg.expname, f'render_train_{ckpt_name}')

        os.makedirs(testsavedir, exist_ok=True)
        with torch.no_grad():
            if cfg.data.dataset_type == 'hyper_dataset':
                rgbs, depths = render_viewpoints_hyper(
                    data_class=data_dict['data_class'],
                    savedir=testsavedir, all =True, test=False,
                    eval_psnr=True,slot_idx = -1, mask = masks,
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
                    mask = masks, slot_idx = -1,batch = batch,
                    **render_viewpoints_kwargs)
    
        rgb_video = 'video.rgb.mp4'
        depth_video = 'video.depth.mp4'
        imageio.mimwrite(os.path.join(testsavedir, rgb_video), utils.to8b(rgbs), fps=10, quality=8)
        imageio.mimwrite(os.path.join(testsavedir, depth_video), utils.to8b(1 - depths / np.max(depths)), fps=10, quality=8)
    


    
    if args.render_test:

        if args.eval_ari and cfg.data.dataset_type =='dnerf': 
            import glob
            from PIL import Image
            gt_seg = sorted(glob.glob(f'{cfg.data.datadir}/test/segmentation*.png'))
            gt_seg = [np.array(Image.open(x).resize((512,512),Image.NEAREST)) for x in gt_seg]
            gt_seg = np.stack(gt_seg, 0)[None,...]

            video = sorted(glob.glob(f'{cfg.data.datadir}/test/[0-9][0-9][0-9].png'))
            video = [np.array(Image.open(x).convert('RGB').resize((512,512),Image.BICUBIC)) for x in video]
            video = np.stack(video, 0)[None,...].astype(np.uint8)
            #video = (video / 255.0).astype(np.float64)
            batch = (gt_seg,video)
        else:
            batch = None

      
        testsavedir = os.path.join(cfg.basedir, cfg.expname, f'render_test_{ckpt_name}')
          
        os.makedirs(testsavedir, exist_ok=True)
        with torch.no_grad():
            if cfg.data.dataset_type == "hyper_dataset":
                rgbs, depths = render_viewpoints_hyper(
                    data_class=data_dict['data_class'],
                    savedir=testsavedir, all=True, test=True,
                    eval_psnr=True,slot_idx = -1, mask = masks,
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
                        mask = masks, slot_idx = -1,batch = batch,
                    **render_viewpoints_kwargs)
        rgb_video = 'video.rgb.mp4'
        depth_video = 'video.depth.mp4'
        
        imageio.mimwrite(os.path.join(testsavedir, rgb_video), utils.to8b(rgbs), fps=10, quality=8)
        imageio.mimwrite(os.path.join(testsavedir, depth_video), utils.to8b(1 - depths / np.max(depths)), fps=10, quality=8)
       


    
    print('Done')

