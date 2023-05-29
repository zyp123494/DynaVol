import os, sys, copy, glob, json, time, random, argparse
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from shutil import copyfile
from tqdm import tqdm, trange

import mmcv
import imageio
import numpy as np
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib import utils
from lib import voxelMlp as VoxelMlp

from lib.load_data import load_data_ours
from lib.load_data import load_data
from post_process import post_process
from torch.utils.tensorboard import SummaryWriter

import pdb


def config_parser():
    '''Define command line arguments
    '''

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', required=True,
                        help='config file path')
    parser.add_argument("--seed", type=int, default=777,
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
    parser.add_argument("--render_video", action='store_true')
    parser.add_argument("--render_video_factor", type=int, default=0,
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')
    parser.add_argument("--eval_ssim", action='store_true')
    parser.add_argument("--eval_lpips_alex", action='store_true')
    parser.add_argument("--eval_lpips_vgg", action='store_true')
    parser.add_argument("--bs", type=int, default=4096)

    # logging/saving options
    parser.add_argument("--i_print",   type=int, default=500,
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_weights", type=int, default=5000,
                        help='frequency of weight ckpt saving')

    parser.add_argument('--eval_ari', action='store_true')
    parser.add_argument('--per_slot', action='store_true',help = 'whether to render per slot result')
    parser.add_argument("--num_slots",   type=int, default=10,help = 'number of slots')
    parser.add_argument("--thresh",   type=float, default=1e-2,help='thresh to determine forground and background.')
    return parser


@torch.no_grad()
def render_viewpoints(model, render_poses, HW, Ks, frame_times, ndc, render_kwargs,
                      gt_imgs=None, savedir=None, render_factor=0,  stc_data=False, bs=4096,
                      eval_ssim=False, eval_lpips_alex=False, slot_idx = -1,mask = None,eval_lpips_vgg=False, writer=None, gs=-1):
    '''Render images for the given viewpoints; run evaluation if gt given.
    '''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    

    if render_factor!=0:
        HW = np.copy(HW)
        Ks = np.copy(Ks)
        HW //= render_factor
        Ks[:, :2, :3] //= render_factor

    rgbs = []
    depths = []
    segmentations = []
    psnrs = []
    ssims = []
    lpips_alex = []
    lpips_vgg = []   
    mses = []

    eps_render = time.time()
    for i, c2w in enumerate(tqdm(render_poses)):

        H, W = HW[i]
        K = Ks[i]
        frame_time = frame_times[i]
    
        frame_time = frame_time.to(device)
        rays_o, rays_d, viewdirs = VoxelMlp.get_rays_of_a_view(
                H, W, K, c2w, ndc, inverse_y=render_kwargs['inverse_y'],
                flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y)
        keys = ['rgb_marched', 'depth', 'segmentation']
        rays_o = rays_o.flatten(0,-2)
        rays_d = rays_d.flatten(0,-2)
        viewdirs = viewdirs.flatten(0,-2)


        render_result_chunks = [
            {k: v for k, v in model(ro, rd, vd, frame_time, i, start=(frame_time==0), training_flag=False, stc_data=stc_data, slot_idx = slot_idx,mask =mask,**render_kwargs).items() if k in keys}
            for ro, rd, vd in zip(rays_o.split(bs, 0), rays_d.split(bs, 0), viewdirs.split(bs, 0))
        ]

        render_result = {
            k: torch.cat([ret[k] for ret in render_result_chunks]).reshape(H,W,-1)
            for k in render_result_chunks[0].keys()
        }


        rgb = render_result['rgb_marched'].cpu().numpy()
        depth = render_result['depth'].cpu().numpy()

        rgbs.append(rgb)
        depths.append(depth)

        if render_kwargs.get('segmentation', True):
            seg = render_result['segmentation'].cpu().numpy()
            segmentations.append(seg)

        if i==0:
            print('Testing', rgb.shape)

        if gt_imgs is not None and render_factor==0:
            p = -10. * np.log10(np.mean(np.square(rgb - gt_imgs[i])))
            psnrs.append(p)
            # pdb.set_trace()
            mses.append(F.mse_loss(torch.tensor(rgb), torch.tensor(gt_imgs[i])).cpu().numpy())
            if eval_ssim:
                ssims.append(utils.rgb_ssim(rgb, gt_imgs[i], max_val=1))
            if eval_lpips_alex:
                lpips_alex.append(utils.rgb_lpips(rgb, gt_imgs[i], net_name='alex', device=c2w.device))
            if eval_lpips_vgg:
                lpips_vgg.append(utils.rgb_lpips(rgb, gt_imgs[i], net_name='vgg', device=c2w.device))

    eps_render = time.time() - eps_render
    eps_time_str = f'{eps_render//3600:02.0f}:{eps_render//60%60:02.0f}:{eps_render%60:02.0f}'
    print('render: render takes ', eps_time_str)

    if len(psnrs):
        print('Testing psnr', np.mean(psnrs), '(avg)')
        print('Testing mse', np.mean(mses), '(avg)')
        np.save(os.path.join(savedir, 'psnr.npy'), np.array(psnrs))
        if writer is not None:
            writer.add_scalar('test/psnr', np.mean(psnrs), gs)
        if eval_ssim: 
            print('Testing ssim', np.mean(ssims), '(avg)')
            np.save(os.path.join(savedir, 'ssim.npy'), np.array(ssims))
            if writer is not None:
                writer.add_scalar('test/ssim', np.mean(ssims), gs)
        if eval_lpips_vgg: 
            print('Testing lpips (vgg)', np.mean(lpips_vgg), '(avg)')
            if writer is not None:
                writer.add_scalar('test/vgg', np.mean(lpips_vgg), gs)
        if eval_lpips_alex: 
            print('Testing lpips (alex)', np.mean(lpips_alex), '(avg)')
            if writer is not None:
                writer.add_scalar('test/alex', np.mean(lpips_alex), gs)

    if savedir is not None:
        os.makedirs(os.path.join(savedir, 'depth'), exist_ok=True)
        print(f'Writing images to {savedir}')
        for i in trange(len(rgbs)):
            rgb8 = utils.to8b(rgbs[i])
            depth8 = utils.to8b(1 - depths[i] / np.max(depths))
            filename = os.path.join(savedir, '{:03d}.png'.format(i))
            imageio.imwrite(filename, rgb8)


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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_dict = load_data_ours(cfg.data)
    data_dict_static = load_data(cfg.data_static)

    # remove useless field
    kept_keys = {
            'hwf', 'HW', 'Ks', 'near', 'far',
            'i_train', 'i_val', 'i_test', 'irregular_shape',
            'poses', 'render_poses', 'images', 'times', 'render_times', 'random_poses'}
    for k in list(data_dict.keys()):
        if k not in kept_keys:
            data_dict.pop(k)

    # construct data tensor
    if data_dict['irregular_shape']:
        data_dict['images'] = [torch.FloatTensor(im, device='cpu') for im in data_dict['images']]
    else:
        data_dict['images'] = torch.FloatTensor(data_dict['images'], device='cpu')
    data_dict['poses'] = torch.Tensor(data_dict['poses'])
    data_dict['times'] = torch.Tensor(data_dict['times'])
    if data_dict['random_poses'] is not None:
        data_dict['random_poses'] = torch.Tensor(data_dict['random_poses'])
    # data_dict['grids'] = torch.Tensor(data_dict['grids'])
    data_dict['render_poses'] = torch.Tensor(data_dict['render_poses'])

    # static 
    kept_keys = {
            'hwf', 'HW', 'Ks', 'near', 'far',
            'i_train', 'i_val', 'i_test', 'irregular_shape',
            'poses', 'render_poses', 'images'}
    for k in list(data_dict_static.keys()):
        if k not in kept_keys:
            data_dict_static.pop(k)
            
    # construct data tensor
    if data_dict_static['irregular_shape']:
        data_dict_static['images'] = [torch.FloatTensor(im, device='cpu') for im in data_dict_static['images']]
    else:
        data_dict_static['images'] = torch.FloatTensor(data_dict_static['images'], device='cpu')
    data_dict_static['poses'] = torch.Tensor(data_dict_static['poses'])

    return data_dict, data_dict_static


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


def scene_rep_reconstruction(args, cfg, cfg_model, cfg_train, data_dict, data_dict_static, stage, writer, xyz_min=None, xyz_max=None, coarse_ckpt_path=None):

    # init
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    xyz_min, xyz_max = compute_bbox_by_cam_frustrm(args=args, cfg=cfg, **data_dict)
    HW, Ks, near, far, i_train, i_val, i_test, poses, render_poses, images, times, render_times, random_poses, hwf = [
        data_dict[k] for k in [
            'HW', 'Ks', 'near', 'far', 'i_train', 'i_val', 'i_test', 'poses', 'render_poses', 'images', 'times', 'render_times', 'random_poses', 'hwf'#, 'vertices'
        ]
    ]
    frame_times = times[i_train]

    # load used static data
    HW_stc, Ks_stc, near_stc, far_stc, i_train_stc, poses_stc, images_stc = [
        data_dict_static[k] for k in [
            'HW', 'Ks', 'near', 'far', 'i_train', 'poses', 'images'
        ]
    ]

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

    start = 0
    model_kwargs = copy.deepcopy(cfg_model)

    if cfg.data.ndc:
        raise NotImplementedError
    else:
        model_class = VoxelMlp.VoxelMlp
  
    timesteps = model_kwargs.pop('timesteps')   
    warp_ray = model_kwargs.pop('warp_ray')  
    world_motion_bound_scale = model_kwargs.pop('world_motion_bound_scale')  
   
    num_voxels = model_kwargs.pop('num_voxels')
    if len(cfg_train.pg_scale) :
        num_voxels = int(num_voxels / (2**len(cfg_train.pg_scale)))
        

    model_kwargs['xyz_min'] = xyz_min
    model_kwargs['xyz_max'] = xyz_max

    model = model_class(num_voxels = num_voxels, timesteps=timesteps, warp_ray=warp_ray, world_motion_bound_scale=world_motion_bound_scale, **model_kwargs)
    optimizer = utils.create_optimizer_or_freeze_model(model, cfg_train, global_step=0)

    if model_kwargs.maskout_near_cam_vox:
        print("maskout near vox")
        model.maskout_near_cam_vox(poses[i_train,:3,3], near)
    # init rendering setup
    render_kwargs = {
        'near': data_dict['near'],
        'far': data_dict['far'],
        'bg': 1 if cfg.data.white_bkgd else 0,
        'stepsize': cfg_model.stepsize,
        'inverse_y': cfg.data.inverse_y,
        'flip_x': cfg.data.flip_x,
        'flip_y': cfg.data.flip_y,
        'render_depth': False,
        'num_slots':cfg.fine_model_and_render.max_instances
    }

    render_kwargs_stc = {
        'near': data_dict_static['near'],
        'far': data_dict_static['far'],
        'bg': 1 if cfg.data.white_bkgd else 0,
        'stepsize': cfg_model.stepsize,
        'inverse_y': cfg.data.inverse_y,
        'flip_x': cfg.data.flip_x,
        'flip_y': cfg.data.flip_y,
    }

    def gather_static_training_rays():
        
        rgb_tr_ori = images_stc[i_train_stc].to('cpu' if cfg.data.load2gpu_on_the_fly else device)

        # if cfg_train.ray_sampler == 'in_maskcache':
        rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz = VoxelMlp.get_training_rays(
                    rgb_tr=rgb_tr_ori,
                    train_poses=poses_stc[i_train_stc],
                    HW=HW_stc[i_train_stc], Ks=Ks_stc[i_train_stc],
                    ndc=cfg.data.ndc, inverse_y=cfg.data.inverse_y,
                    flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y,
                    )
        
        index_generator = VoxelMlp.batch_indices_generator(len(rgb_tr), cfg_train.N_rand)
        batch_index_sampler = lambda: next(index_generator)
        return rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz, batch_index_sampler

    
    def gather_training_rays():
        if data_dict['irregular_shape']:
            rgb_tr_ori = [images[i].to('cpu' if cfg.data.load2gpu_on_the_fly else device) for i in i_train]
        else:
            rgb_tr_ori = images[i_train].to('cpu' if cfg.data.load2gpu_on_the_fly else device)

        if cfg_train.ray_sampler == 'in_maskcache':
            rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz = VoxelMlp.get_training_rays_in_maskcache_sampling(
                    rgb_tr_ori=rgb_tr_ori,
                    train_poses=poses[i_train],
                    HW=HW[i_train], Ks=Ks[i_train],
                    ndc=cfg.data.ndc, inverse_y=cfg.data.inverse_y,
                    flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y,
                    model=model, render_kwargs=render_kwargs)
        elif cfg_train.ray_sampler == 'flatten':
            rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz = VoxelMlp.get_training_rays_flatten(
                rgb_tr_ori=rgb_tr_ori,
                train_poses=poses[i_train],
                HW=HW[i_train], Ks=Ks[i_train], ndc=cfg.data.ndc, inverse_y=cfg.data.inverse_y,
                flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y)
        else:
            rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz = VoxelMlp.get_training_rays(
                rgb_tr=rgb_tr_ori,
                train_poses=poses[i_train],
                HW=HW[i_train], Ks=Ks[i_train], ndc=cfg.data.ndc, inverse_y=cfg.data.inverse_y,
                flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y)
        index_generator = VoxelMlp.batch_indices_generator(len(rgb_tr), cfg_train.N_rand)
        batch_index_sampler = lambda: next(index_generator)
        return rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz, batch_index_sampler

    rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz, batch_index_sampler = gather_training_rays()
    rgb_tr_stc, rays_o_tr_stc, rays_d_tr_stc, viewdirs_tr_stc, imsz_stc, batch_index_sampler_stc = gather_static_training_rays()


    if cfg.data.load2gpu_on_the_fly:
        rgb_tr = rgb_tr.to(device)
        rays_o_tr = rays_o_tr.to(device)
        rays_d_tr = rays_d_tr.to(device)
        viewdirs_tr = viewdirs_tr.to(device)
        frame_times = frame_times.to(device)

        if not cfg.data.ndc:
            rgb_tr_stc = rgb_tr.to(device)
            rays_o_tr_stc = rays_o_tr.to(device)
            rays_d_tr_stc = rays_d_tr.to(device)
            viewdirs_tr_stc = viewdirs_tr.to(device)

    # view-count-based learning rate
    if cfg_train.pervoxel_lr:
        def per_voxel_init():
            cnt = model.voxel_count_views(
                    rays_o_tr=rays_o_tr, rays_d_tr=rays_d_tr, imsz=imsz, near=near, far=far,
                    stepsize=cfg_model.stepsize, downrate=cfg_train.pervoxel_lr_downrate,
                    irregular_shape=data_dict['irregular_shape'])
            optimizer.set_pervoxel_lr(cnt)
            with torch.no_grad():
                model.density[cnt <= 2] = -100
        per_voxel_init()

    # init decay factor for first stage
    decay_factor = 1
    # GOGO
    torch.cuda.empty_cache()
    psnr_lst = []
    time0 = time.time()
    global_step = -1

    cur_view = 0
    print("weight cycle:",cfg_train.weight_cycle)
    for global_step in trange(start, cfg_train.N_iters):

        if global_step in cfg_train.pg_scale:
            n_rest_scales = len(cfg_train.pg_scale)-cfg_train.pg_scale.index(global_step)-1
            cur_voxels = int(cfg_model.num_voxels / (2**n_rest_scales))   
            model.scale_volume_grid(cur_voxels)
            optimizer = utils.create_optimizer_or_freeze_model(model, cfg_train, global_step=0)
            model.density.data.sub_(1)
    

        # random sample rays
        if cfg_train.ray_sampler in ['flatten', 'in_maskcache']:
            sel_i = batch_index_sampler()
            target = rgb_tr[sel_i]
            rays_o = rays_o_tr[sel_i]
            rays_d = rays_d_tr[sel_i]
            viewdirs = viewdirs_tr[sel_i]
        elif cfg_train.ray_sampler == 'random':
            sel_b = torch.randint(rgb_tr.shape[0], [cfg_train.N_rand])
            sel_r = torch.randint(rgb_tr.shape[1], [cfg_train.N_rand])
            sel_c = torch.randint(rgb_tr.shape[2], [cfg_train.N_rand])
            target = rgb_tr[sel_b, sel_r, sel_c]
            rays_o = rays_o_tr[sel_b, sel_r, sel_c]
            rays_d = rays_d_tr[sel_b, sel_r, sel_c]
            viewdirs = viewdirs_tr[sel_b, sel_r, sel_c]
        elif cfg_train.ray_sampler == 'random_1im':
            # Randomly select one image due to time step.
            if global_step >= cfg_train.precrop_iters_time:
                img_i = np.random.choice(i_train)
            else:
                skip_factor = global_step / float(cfg_train.precrop_iters_time) * len(i_train)
                max_sample = max(int(skip_factor), 3)
                img_i = np.random.choice(i_train[:max_sample])
            # Require i_train order is the same as the above reordering of training images. 
            sel_r = torch.randint(rgb_tr.shape[1], [cfg_train.N_rand])
            sel_c = torch.randint(rgb_tr.shape[2], [cfg_train.N_rand])
            target = rgb_tr[img_i, sel_r, sel_c]
            rays_o = rays_o_tr[img_i, sel_r, sel_c]
            rays_d = rays_d_tr[img_i, sel_r, sel_c]
            viewdirs = viewdirs_tr[img_i, sel_r, sel_c]
            frame_time = frame_times[img_i].to(target.device)
        elif cfg_train.ray_sampler == 'sequential_1im_fixed':

            num_views = frame_times.shape[0] // timesteps
            assert (frame_times.shape[0]%timesteps) == 0

            img_i = global_step % timesteps  #[0-59]
            if img_i == 0 and global_step!= 0:
                cur_view = (cur_view+1) % num_views
            img_i = img_i * num_views + cur_view #[0-239] 

            img_i = torch.tensor(img_i,device = device)
            sel_r = torch.randint(rgb_tr.shape[1], [cfg_train.N_rand])
            sel_c = torch.randint(rgb_tr.shape[2], [cfg_train.N_rand])
            target = rgb_tr[img_i, sel_r, sel_c]
            rays_o = rays_o_tr[img_i, sel_r, sel_c]
            rays_d = rays_d_tr[img_i, sel_r, sel_c]
            viewdirs = viewdirs_tr[img_i, sel_r, sel_c]
            frame_time = frame_times[img_i]
          
        else:
            raise NotImplementedError

        # model for static data
        if not cfg.data.ndc:
            img_i_stc = np.random.choice(i_train_stc[:cfg.data_static.num_train])
            sel_r_stc = torch.randint(rgb_tr_stc.shape[1], [cfg_train.N_rand])
            sel_c_stc = torch.randint(rgb_tr_stc.shape[2], [cfg_train.N_rand])
            img_i_stc = torch.tensor(img_i_stc,device = device)
            target_stc = rgb_tr_stc[img_i_stc, sel_r_stc, sel_c_stc]
            rays_o_stc = rays_o_tr_stc[img_i_stc, sel_r_stc, sel_c_stc]
            rays_d_stc = rays_d_tr_stc[img_i_stc, sel_r_stc, sel_c_stc]
            viewdirs_stc = viewdirs_tr_stc[img_i_stc, sel_r_stc, sel_c_stc]
            frame_time_stc = frame_times[0]

       
        render_result = model(rays_o, rays_d, viewdirs, frame_time, img_i, global_step=global_step, start=(frame_time==0), **render_kwargs)

        # static volume rendering
        if not cfg.data.ndc:
            render_result_stc = model(rays_o_stc, rays_d_stc, viewdirs_stc, frame_time_stc, 0, global_step=global_step, start=True, stc_data=True, **render_kwargs)

        # gradient descent step
        optimizer.zero_grad(set_to_none=True)
        loss = cfg_train.weight_main * F.mse_loss(render_result['rgb_marched'], target)
       
        psnr = utils.mse2psnr(loss.detach())

        if cfg_train.weight_cycle > 0:
            
            loss += cfg_train.weight_cycle * render_result['cycle_loss']
        
        
        loss_static = cfg_train.weight_main * F.mse_loss(render_result_stc['rgb_marched'], target_stc)
    

        if cfg_train.weight_entropy_last > 0:
            pout = render_result['alphainv_last'].clamp(1e-6, 1-1e-6)
            entropy_last_loss = -(pout*torch.log(pout) + (1-pout)*torch.log(1-pout)).mean()
            loss += cfg_train.weight_entropy_last * entropy_last_loss
            
            pout_ = render_result_stc['alphainv_last'].clamp(1e-6, 1-1e-6)
            entropy_last_loss_stc = -(pout_*torch.log(pout_) + (1-pout_)*torch.log(1-pout_)).mean()
            loss_static += cfg_train.weight_entropy_last * entropy_last_loss_stc
        
        
        if cfg_train.weight_rgbper > 0:
            rgbper = (render_result['raw_rgb'] - target[render_result['ray_id']]).pow(2).sum(-1)
            rgbper_loss = (rgbper * render_result['weights'].detach()).sum() / len(rays_o)
            loss += cfg_train.weight_rgbper * rgbper_loss 

            
            rgbper_stc = (render_result_stc['raw_rgb'] - target_stc[render_result_stc['ray_id']]).pow(2).sum(-1)
            rgbper_loss_stc = (rgbper_stc * render_result_stc['weights'].detach()).sum() / len(rays_o_stc)
            loss_static += cfg_train.weight_rgbper * rgbper_loss_stc

        loss += cfg.data_static.num_train / timesteps  * cfg_train.weight_static * loss_static
       
        loss.backward()

        writer.add_scalar('train/loss', loss.item(), global_step)
        if not cfg.data.ndc:
            writer.add_scalar('train/loss_stc', loss_static.item(), global_step)
        writer.add_scalar('train/psnr', psnr, global_step)


        if global_step<cfg_train.tv_before and global_step>cfg_train.tv_after and global_step%cfg_train.tv_every==0:
            if cfg_train.weight_tv_density>0:
                model.density_total_variation_add_grad(
                    cfg_train.weight_tv_density/len(rays_o), global_step<cfg_train.tv_dense_before)
         

        optimizer.step()
        psnr_lst.append(psnr.item())   

        # update lr
        decay_steps = cfg_train.lrate_decay * 1000
        decay_factor = 0.1 ** (1/decay_steps)
        for i_opt_g, param_group in enumerate(optimizer.param_groups):
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
            # model.motion = torch.nn.Parameter(torch.cat(model.motion_list))     
            torch.save({
                'global_step': global_step,
                'model_kwargs': model.get_kwargs(),
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, path)
            print(f'scene_rep_reconstruction ({stage}): saved checkpoints at', path)
            test(args, cfg, str(global_step), model, writer)


    if global_step != -1:
        torch.save({
            'global_step': global_step,
            'model_kwargs': model.get_kwargs(),
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, last_ckpt_path)
        print(f'scene_rep_reconstruction ({stage}): saved checkpoints at', last_ckpt_path)


def train(args, cfg, data_dict, data_dict_static):

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
    log_path = os.path.join('tb', cfg.basedir+cfg.expname+'_'+time.strftime('%Y%m%d%H%M%S', time.localtime()))
    os.makedirs(log_path, exist_ok=True)
    writer = SummaryWriter(log_path)
    scene_rep_reconstruction(
            args=args, cfg=cfg,
            cfg_model=cfg.fine_model_and_render, cfg_train=cfg.fine_train,
            data_dict=data_dict, data_dict_static=data_dict_static, stage='fine', writer=writer)
    eps_fine = time.time() - eps_fine
    eps_time_str = f'{eps_fine//3600:02.0f}:{eps_fine//60%60:02.0f}:{eps_fine%60:02.0f}'
    print('train: fine detail reconstruction in', eps_time_str)


def get_gradient(args, cfg, stage, model):
    stepsize = cfg.fine_model_and_render.stepsize
    
    render_kwargs={
        'near': data_dict['near'],
        'far': data_dict['far'],
        'bg': 1 if cfg.data.white_bkgd else 0,
        'stepsize': stepsize,
        'inverse_y': cfg.data.inverse_y,
        'flip_x': cfg.data.flip_x,
        'flip_y': cfg.data.flip_y,
        'render_depth': False,
        'num_slots':cfg.fine_model_and_render.max_instances,
    }
        
    ndc = cfg.data.ndc
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    
    print("xyzmin,max",model.xyz_min,model.xyz_max)

    eps_render = time.time()
    for i, c2w in enumerate(tqdm(data_dict['poses'][data_dict['i_train']])):

        H, W = data_dict['HW'][data_dict['i_train']][i]
        K = data_dict['Ks'][data_dict['i_train']][i]
        frame_time = data_dict['times'][data_dict['i_train']][i]
        frame_time = frame_time.to(device)
        rays_o, rays_d, viewdirs = VoxelMlp.get_rays_of_a_view(
                H, W, K, c2w, ndc, inverse_y=render_kwargs['inverse_y'],
                flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y)
        keys = ['rgb_marched']
        rays_o = rays_o.flatten(0,-2)
        rays_d = rays_d.flatten(0,-2)
        viewdirs = viewdirs.flatten(0,-2)

        for ro, rd, vd in zip(rays_o.split(1024, 0), rays_d.split(1024, 0), viewdirs.split(1024, 0)):
            render_result = model(ro, rd, vd, frame_time, i, start=(frame_time==0), training_flag=False, stc_data=False,**render_kwargs)
            render_result['rgb_marched'].sum().backward()
    
    return model.density.grad[0][0]


def test(args, cfg, stage, model, writer):
   
    stepsize = cfg.fine_model_and_render.stepsize
    render_viewpoints_kwargs = {
            'model': model,
            'ndc': cfg.data.ndc,
            'render_kwargs': {
                'near': data_dict['near'],
                'far': data_dict['far'],
                'bg': 1 if cfg.data.white_bkgd else 0,
                'stepsize': stepsize,
                'inverse_y': cfg.data.inverse_y,
                'flip_x': cfg.data.flip_x,
                'flip_y': cfg.data.flip_y,
                'render_depth': True,
                'num_slots':cfg.fine_model_and_render.max_instances,
                'segmentation': False,
            },
        }

    if args.render_train:
           
        testsavedir = os.path.join(cfg.basedir, cfg.expname, f'render_train_{stage}')
        os.makedirs(testsavedir, exist_ok=True)
        rgbs, depths = render_viewpoints(
                render_poses=data_dict['poses'][data_dict['i_train']],
                HW=data_dict['HW'][data_dict['i_train']],
                Ks=data_dict['Ks'][data_dict['i_train']],
                frame_times=data_dict['times'][data_dict['i_train']],
                gt_imgs=[data_dict['images'][i].cpu().numpy() for i in data_dict['i_train']],
                savedir=testsavedir,
                bs=args.bs,
                eval_ssim=args.eval_ssim, eval_lpips_alex=args.eval_lpips_alex, eval_lpips_vgg=args.eval_lpips_vgg,
                **render_viewpoints_kwargs)
        
        rgb_video = 'video.rgb.mp4'
        depth_video = 'video.depth.mp4'
        shape_video = 'video.shape.mp4'
        imageio.mimwrite(os.path.join(testsavedir, rgb_video), utils.to8b(rgbs), fps=10, quality=8)
        imageio.mimwrite(os.path.join(testsavedir, depth_video), utils.to8b(1 - depths / np.max(depths)), fps=10, quality=8)
    

    # render testset and eval
    if args.render_test:        
        testsavedir = os.path.join(cfg.basedir, cfg.expname, f'render_test_{stage}')
        os.makedirs(testsavedir, exist_ok=True)
        rgbs, depths = render_viewpoints(
                render_poses=data_dict['poses'][data_dict['i_test']],
                HW=data_dict['HW'][data_dict['i_test']],
                Ks=data_dict['Ks'][data_dict['i_test']],
                frame_times=data_dict['times'][data_dict['i_test']],
                gt_imgs=[data_dict['images'][i].cpu().numpy() for i in data_dict['i_test']],
                savedir=testsavedir,
                bs=args.bs,
                eval_ssim=args.eval_ssim, eval_lpips_alex=args.eval_lpips_alex, eval_lpips_vgg=args.eval_lpips_vgg,
                writer=writer, gs=int(stage),
                **render_viewpoints_kwargs)

        rgb_video = 'video.rgb.mp4'
        depth_video = 'video.depth.mp4'
        shape_video = 'video.shape.mp4'
        imageio.mimwrite(os.path.join(testsavedir, rgb_video), utils.to8b(rgbs), fps=10, quality=8)
        imageio.mimwrite(os.path.join(testsavedir, depth_video), utils.to8b(1 - depths / np.max(depths)), fps=10, quality=8)
      
    # render video
    if args.render_video:
        testsavedir = os.path.join(cfg.basedir, cfg.expname, f'render_video_{stage}')
        os.makedirs(testsavedir, exist_ok=True)
        rgbs, depths = render_viewpoints(
                render_poses=data_dict['render_poses'],
                HW=data_dict['HW'][data_dict['i_test']][[0]].repeat(len(data_dict['render_poses']), 0),
                Ks=data_dict['Ks'][data_dict['i_test']][[0]].repeat(len(data_dict['render_poses']), 0),
                frame_times=data_dict['render_times'],
                render_factor=args.render_video_factor,
                bs=args.bs,
                savedir=testsavedir,
                **render_viewpoints_kwargs)
        imageio.mimwrite(os.path.join(testsavedir, 'video.rgb.mp4'), utils.to8b(rgbs), fps=30, quality=8)
        imageio.mimwrite(os.path.join(testsavedir, 'video.depth.mp4'), utils.to8b(1 - depths / np.max(depths)), fps=30, quality=8)



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
    # load images / poses / camera settings / data split
    data_dict, data_dict_static = load_everything(args=args, cfg=cfg)

    # export scene bbox and camera poses in 3d for debugging and visualization
    if args.export_bbox_and_cams_only:
        print('Export bbox and cameras...')
        xyz_min, xyz_max = compute_bbox_by_cam_frustrm(args=args, cfg=cfg, **data_dict)
        poses, HW, Ks, i_train = data_dict['poses'], data_dict['HW'], data_dict['Ks'], data_dict['i_train']
        near, far = data_dict['near'], data_dict['far']
        cam_lst = []
        for c2w, (H, W), K in zip(poses[i_train], HW[i_train], Ks[i_train]):
            rays_o, rays_d, viewdirs = VoxelMlp.get_rays_of_a_view(
                    H, W, K, c2w, cfg.data.ndc, inverse_y=cfg.data.inverse_y,
                    flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y,)
            cam_o = rays_o[0,0].cpu().numpy()
            cam_d = rays_d[[0,0,-1,-1],[0,-1,0,-1]].cpu().numpy()
            cam_lst.append(np.array([cam_o, *(cam_o+cam_d*max(near, far*0.05))]))
        np.savez_compressed(args.export_bbox_and_cams_only,
            xyz_min=xyz_min.cpu().numpy(), xyz_max=xyz_max.cpu().numpy(),
            cam_lst=np.array(cam_lst))
        print('done')
        sys.exit()

    

    # train
    if not args.render_only:
        train(args, cfg, data_dict, data_dict_static)

    # load model for rendring
    if args.render_test or args.render_train or args.render_video:
        if args.ft_path:
            ckpt_path = args.ft_path
        else:
            ckpt_path = os.path.join(cfg.basedir, cfg.expname, 'fine_last.tar')
        ckpt_name = ckpt_path.split('/')[-1][:-4]
        if cfg.data.ndc:
            raise NotImplementedError
        else:
            model_class = VoxelMlp.VoxelMlp
        model = utils.load_model_ours(model_class, ckpt_path).to(device)
        stepsize = cfg.fine_model_and_render.stepsize
        print(ckpt_path)
        render_viewpoints_kwargs = {
            'model': model,
            'ndc': cfg.data.ndc,
            'render_kwargs': {
                'near': data_dict['near'],
                'far': data_dict['far'],
                'bg': 1 if cfg.data.white_bkgd else 0,
                'stepsize': stepsize,
                'inverse_y': cfg.data.inverse_y,
                'flip_x': cfg.data.flip_x,
                'flip_y': cfg.data.flip_y,
                'render_depth': True,
                'num_slots':cfg.fine_model_and_render.max_instances,
                'segmentation': args.eval_ari
            },
        }

    
        
    dx = model.get_dynamics(timesteps =cfg.fine_model_and_render.timesteps)  #[T,3,X,Y,Z]
    

    dx[1:] = dx[1:] - dx[:-1]  #velocity
    dx = dx.cpu().numpy()

    
    mean_rgb = model.get_mean_rgb()  #[3,X,Y,Z]


    #remove those space which has little contribution to the final color.
    grad = get_gradient(args, cfg, "fine", model)
    with torch.no_grad():
        for i in range(model.density.shape[0]):
            if grad[i, :, :].max() < 1e-6:
                model.density[0,0,i, :, :] = -100
            else:
                break
        for i in range(model.density.shape[1]):
            if grad[:, i, :].max() < 1e-6:
                model.density[0,0,:, i, :] = -100
            else:
                break
        for i in range(model.density.shape[2]):
            if grad[:, :, i].max() < 1e-6:
                model.density[0,0,:, :, i] = -100
            else:
                break
        for i in range(1,model.density.shape[0]+1):
            if grad[-i, :, :].max() < 1e-6:
                model.density[0,0,-i, :, :] = -100
            else:
                break
        for i in range(1,model.density.shape[1]+1):
            if grad[:, -i, :].max() < 1e-6:
                model.density[0,0,:, -i, :] = -100
            else:
                break
        for i in range(1,model.density.shape[2]+1):
            if grad[:, :, -i].max() < 1e-6:
                model.density[0,0,:, :, -i] = -100
            else:
                break

    
    
    mean_rgb = mean_rgb.cpu().numpy()

    
    if args.per_slot:
        #per slot rendering
        masks = post_process(model.density,model.act_shift,args.num_slots,dx,mean_rgb,args.thresh,method = 'cc').to(model.density.device).float()
        num_slots = masks.shape[1]
        

        path = os.path.join(cfg.basedir, cfg.expname, 'fine_last_n.tar')
        
        kwargs = model.get_kwargs()
        kwargs['max_instances'] = num_slots
        with torch.no_grad():
            masks_ = masks.to(model.density.device)
            density = F.softplus(model.density + model.act_shift,True)  #[1,1,X,Y,Z]
            density = density * masks_  #[1,k,X,Y,Z]
            state_dict = model.state_dict()

            density = torch.log(torch.exp(density) -1 + 1e-10) - model.act_shift
            density = density.to(state_dict['density'])
            state_dict['density'] = density
            
            
            print(density.shape)
            
        torch.save({
            'model_kwargs': kwargs,
            'model_state_dict': state_dict,
        }, path)
    else:
        num_slots = 0
        masks = None
    # render trainset and eval
    if args.render_train:
        for i in range(-1, num_slots):
            if i == -1:
                testsavedir = os.path.join(cfg.basedir, cfg.expname, f'render_train_{ckpt_name}')

            else:
                testsavedir = os.path.join(cfg.basedir, cfg.expname, f'render_train_{ckpt_name}_slot{i}')
            os.makedirs(testsavedir, exist_ok=True)
            rgbs, depths = render_viewpoints(
                    render_poses=data_dict['poses'][data_dict['i_train']][:3],
                    HW=data_dict['HW'][data_dict['i_train']],
                    Ks=data_dict['Ks'][data_dict['i_train']],
                    frame_times=data_dict['times'][data_dict['i_train']],
                    gt_imgs=[data_dict['images'][i].cpu().numpy() for i in data_dict['i_train']],
                    savedir=testsavedir,
                    eval_ssim=args.eval_ssim, eval_lpips_alex=args.eval_lpips_alex, eval_lpips_vgg=args.eval_lpips_vgg,
                    slot_idx = i,
                    mask = masks,
                    bs=args.bs,
                    stc_data=False,
                    **render_viewpoints_kwargs)
        
            rgb_video = 'video.rgb.mp4'
            depth_video = 'video.depth.mp4'
            shape_video = 'video.shape.mp4'
            imageio.mimwrite(os.path.join(testsavedir, rgb_video), utils.to8b(rgbs), fps=10, quality=8)
            imageio.mimwrite(os.path.join(testsavedir, depth_video), utils.to8b(1 - depths / np.max(depths)), fps=10, quality=8)
        


    # render testset and eval
    if args.render_test:
        for i in range(-1, num_slots):
            if i == -1:
                testsavedir = os.path.join(cfg.basedir, cfg.expname, f'render_test_{ckpt_name}')
            else:
                testsavedir = os.path.join(cfg.basedir, cfg.expname, f'render_test_{ckpt_name}_slot{i}')
            os.makedirs(testsavedir, exist_ok=True)
            rgbs, depths = render_viewpoints(
                    render_poses=data_dict['poses'][data_dict['i_test']][:3],
                    HW=data_dict['HW'][data_dict['i_test']],
                    Ks=data_dict['Ks'][data_dict['i_test']],
                    frame_times=data_dict['times'][data_dict['i_test']],
                    gt_imgs=[data_dict['images'][i].cpu().numpy() for i in data_dict['i_test']],
                    savedir=testsavedir,
                    eval_ssim=args.eval_ssim, eval_lpips_alex=args.eval_lpips_alex, eval_lpips_vgg=args.eval_lpips_vgg,
                    slot_idx = i,
                    mask = masks,
                    #timesteps = cfg.fine_model_and_render.timesteps,
                    bs=args.bs,
                    stc_data=False,
                    **render_viewpoints_kwargs)

            rgb_video = 'video.rgb.mp4'
            depth_video = 'video.depth.mp4'
            shape_video = 'video.shape.mp4'
            imageio.mimwrite(os.path.join(testsavedir, rgb_video), utils.to8b(rgbs), fps=10, quality=8)
            imageio.mimwrite(os.path.join(testsavedir, depth_video), utils.to8b(1 - depths / np.max(depths)), fps=10, quality=8)
        

    # render video
    if args.render_video:
        testsavedir = os.path.join(cfg.basedir, cfg.expname, f'render_video_{ckpt_name}')
        os.makedirs(testsavedir, exist_ok=True)
        rgbs, depths = render_viewpoints(
                render_poses=data_dict['render_poses'],
                HW=data_dict['HW'][data_dict['i_test']][[0]].repeat(len(data_dict['render_poses']), 0),
                Ks=data_dict['Ks'][data_dict['i_test']][[0]].repeat(len(data_dict['render_poses']), 0),
                frame_times=data_dict['render_times'],
                render_factor=args.render_video_factor,
                savedir=testsavedir,
                stc_data=False,
                #timesteps = cfg.fine_model_and_render.timesteps,
                bs=args.bs,
                **render_viewpoints_kwargs)
        imageio.mimwrite(os.path.join(testsavedir, 'video.rgb.mp4'), utils.to8b(rgbs), fps=30, quality=8)
        imageio.mimwrite(os.path.join(testsavedir, 'video.depth.mp4'), utils.to8b(1 - depths / np.max(depths)), fps=30, quality=8)
    
    print('Done')

