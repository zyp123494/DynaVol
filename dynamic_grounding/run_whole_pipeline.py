import os, sys, copy, glob, json, time, random, argparse
from shutil import copyfile
from tqdm import tqdm, trange

import mmcv
import imageio
import numpy as np
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import glob
from PIL import Image
from lib import utils
from lib import voxelMlp as VoxelMlp

from lib.load_data import load_data_ours

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
    return parser

def gray2rgb(seg):
    from PIL import Image
    import skimage.io
    PALETTE = [0, 0, 0, 128, 0, 0, 0, 128, 0, 128, 128, 0, 0, 0, 128, 128, 0, 128, 0, 128, 128, 128, 128, 128, 64, 0, 0, 191, 0, 0, 64, 128, 0, 191, 128, 0, 64, 0, 128]
    img =Image.fromarray(seg.astype(np.uint8),mode = 'P')
    img.putpalette(PALETTE)
    img.save('tmp.png')
    img = skimage.io.imread('tmp.png')
    return img[...,:3]

@torch.no_grad()
def render_viewpoints(model, render_poses, HW, Ks, frame_times, ndc, render_kwargs,
                      gt_imgs=None, savedir=None, render_factor=0, batch=None, stc_data=False, bs=4096,
                      eval_ssim=False, eval_lpips_alex=False, eval_lpips_vgg=False, writer=None, gs=-1):
    '''Render images for the given viewpoints; run evaluation if gt given.
    '''
    # init
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    
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
        frame_time = frame_times[i].to(device)
        rays_o, rays_d, viewdirs = VoxelMlp.get_rays_of_a_view(
                H, W, K, c2w, ndc, inverse_y=render_kwargs['inverse_y'],
                flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y)
        keys = ['rgb_marched', 'depth', 'segmentation']
        rays_o = rays_o.flatten(0,-2)
        rays_d = rays_d.flatten(0,-2)
        viewdirs = viewdirs.flatten(0,-2)

        # find the neighbored timesteps motion and put them on gpu memory
        render_result_chunks = [
            {k: v for k, v in model(ro, rd, vd, frame_time, i, start=(frame_time==0), training_flag=False, stc_data=stc_data, first_episode=False, **render_kwargs).items() if k in keys}
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

    f1 = open(os.path.join(savedir, 'result.txt'), 'w')
    if len(psnrs):
        print('Testing psnr', np.mean(psnrs), '(avg)')
        f1.write('Testing psnr:' + str(np.mean(psnrs)) + '(avg)\n')
        print('Testing mse', np.mean(mses), '(avg)')
        f1.write('Testing mses:' + str(np.mean(mses)) + '(avg)\n')
        np.save(os.path.join(savedir, 'psnr.npy'), np.array(psnrs))
        if writer is not None:
            writer.add_scalar('test/psnr', np.mean(psnrs), gs)
        if eval_ssim: 
            print('Testing ssim', np.mean(ssims), '(avg)')
            f1.write('Testing ssim:' + str(np.mean(ssims)) + '(avg)\n')
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
        
        # get ari and ari_fg
        if batch is not None and render_kwargs.get('segmentation', True):
            segmentations = np.stack(segmentations, axis=0)[None, ..., 0]
            ari, ari_fg = ari_metrics(segmentations, gt_segmentations)
            print('Testing ari/ari-fg', ari, ari_fg)
            f1.write('Testing ari/ari-fg:' + str(ari) + str(ari_fg))

            if savedir is not None:
                utils.vis_seg(video[0], segmentations[0,...,None], gt_segmentations[0,...,None], savedir)
    
    f1.close()


    if savedir is not None:
        os.makedirs(os.path.join(savedir, 'depth'), exist_ok=True)
        print(f'Writing images to {savedir}')
        for i in trange(len(rgbs)):
            rgb8 = utils.to8b(rgbs[i])
            depth8 = utils.to8b(1 - depths[i] / np.max(depths))
            filename = os.path.join(savedir, '{:03d}.png'.format(i))
            imageio.imwrite(filename, rgb8)

            filename = os.path.join(savedir, 'depth', '{:03d}.png'.format(i))
            imageio.imwrite(filename, depth8)

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
    data_dict['render_poses'] = torch.Tensor(data_dict['render_poses'])


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


def scene_rep_reconstruction(args, cfg, cfg_model, cfg_train, data_dict, stage, writer, xyz_min=None, xyz_max=None, coarse_ckpt_path=None):

    # init
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    xyz_min, xyz_max = compute_bbox_by_cam_frustrm(args=args, cfg=cfg, **data_dict)
    HW, Ks, near, far, i_train, i_val, i_test, poses, render_poses, images, times, render_times, random_poses, hwf = [
        data_dict[k] for k in [
            'HW', 'Ks', 'near', 'far', 'i_train', 'i_val', 'i_test', 'poses', 'render_poses', 'images', 'times', 'render_times', 'random_poses', 'hwf'#, 'vertices'
        ]
    ]
    frame_times = times[i_train]

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
    print(f'scene_rep_reconstruction ({stage}): reload from {cfg_train.static_model_path}')
    start = 0
    model_kwargs = copy.deepcopy(cfg_model)
    model_kwargs['weight_init'] = {
		'param':model_kwargs[ 'init_weight'],
		'linear_w': model_kwargs['init_weight'],
		'linear_b': model_kwargs['init_bias'],
		'conv_w': model_kwargs['init_weight'],
		'conv_b': model_kwargs['init_bias']}

    if cfg.data.ndc:
        raise NotImplementedError
    else:
        model_class = VoxelMlp.VoxelMlp
    num_voxels_motion = model_kwargs.pop('num_voxels_motion')
    timesteps = model_kwargs.pop('timesteps')   
    warp_ray = model_kwargs.pop('warp_ray')  
    world_motion_bound_scale = model_kwargs.pop('world_motion_bound_scale')  

    if reload_ckpt_path is not None:
        model = utils.load_model_ours(model_class, reload_ckpt_path)
        print("Reload parameters from {}".format(reload_ckpt_path))
    else:
        model = utils.load_pretrained_model_whole(model_class, num_voxels_motion, timesteps, warp_ray, 
                                        cfg_train.static_model_path, world_motion_bound_scale, model_kwargs).to(device)
    optimizer = utils.create_optimizer_or_freeze_model(model, cfg_train, global_step=0)

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

    # init batch rays sampler
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

    if cfg.data.load2gpu_on_the_fly:
        rgb_tr = rgb_tr.to(device)
        rays_o_tr = rays_o_tr.to(device)
        rays_d_tr = rays_d_tr.to(device)
        viewdirs_tr = viewdirs_tr.to(device)
        frame_times = frame_times.to(device)


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
    torch.cuda.empty_cache()
    psnr_lst = []
    time0 = time.time()
    global_step = -1
    
    for global_step in trange(start, cfg_train.N_iters):
        if global_step in cfg_train.pg_motionscale:
            optimizer = utils.create_optimizer_or_freeze_model(model, cfg_train, global_step=0)
         
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
            # pdb.set_trace()
            img_i = torch.tensor(global_step % timesteps, device=device)   

            sel_r = torch.randint(rgb_tr.shape[1], [cfg_train.N_rand])
            sel_c = torch.randint(rgb_tr.shape[2], [cfg_train.N_rand])
            target = rgb_tr[img_i, sel_r, sel_c]
            rays_o = rays_o_tr[img_i, sel_r, sel_c]
            rays_d = rays_d_tr[img_i, sel_r, sel_c]
            viewdirs = viewdirs_tr[img_i, sel_r, sel_c]
            frame_time = frame_times[img_i]
            pose = poses[i_train][img_i]
        else:
            raise NotImplementedError

        # volume rendering
        render_result = model(rays_o, rays_d, viewdirs, frame_time, img_i, global_step=global_step, start=(frame_time==0), first_episode=(int(global_step/timesteps)==0), **render_kwargs)

        # gradient descent step
        optimizer.zero_grad(set_to_none=True)
        loss = cfg_train.weight_main * F.mse_loss(render_result['rgb_marched'], target)
        
        psnr = utils.mse2psnr(loss.detach())


        if cfg_train.weight_entropy_last > 0:
            pout = render_result['alphainv_last'].clamp(1e-6, 1-1e-6)
            entropy_last_loss = -(pout*torch.log(pout) + (1-pout)*torch.log(1-pout)).mean()
            loss += cfg_train.weight_entropy_last * entropy_last_loss
            
            
        if cfg_train.weight_rgbper > 0:
            rgbper = (render_result['raw_rgb'] - target[render_result['ray_id']]).pow(2).sum(-1)
            rgbper_loss = (rgbper * render_result['weights'].detach()).sum() / len(rays_o)
            loss += cfg_train.weight_rgbper * rgbper_loss 
            
        
       
        loss.backward()

        writer.add_scalar('train/loss', loss.item(), global_step) 
        writer.add_scalar('train/psnr', psnr, global_step)

        if frame_time == 0 and global_step!=0:
            S, D = render_result['mean_of_slots'].shape
            for slot_index in range(S):
                writer.add_scalar('train/slots_o_'+str(slot_index), torch.mean(render_result['mean_of_slots'][slot_index]), int(global_step/timesteps)-1)


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
            torch.save({
                'global_step': global_step,
                'model_kwargs': model.get_kwargs(),
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, path)
            print(f'scene_rep_reconstruction ({stage}): saved checkpoints at', path)
            test(args, cfg, str(global_step), model, writer, cfg_train.N_iters)


    if global_step != -1:
        torch.save({
            'global_step': global_step,
            'model_kwargs': model.get_kwargs(),
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, last_ckpt_path)
        print(f'scene_rep_reconstruction ({stage}): saved checkpoints at', last_ckpt_path)


def train(args, cfg, data_dict):

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
            data_dict=data_dict, stage='fine', writer=writer)
    eps_fine = time.time() - eps_fine
    eps_time_str = f'{eps_fine//3600:02.0f}:{eps_fine//60%60:02.0f}:{eps_fine%60:02.0f}'
    print('train: fine detail reconstruction in', eps_time_str)


def test(args, cfg, stage, model, writer, N_iters):
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
                # TODO segmentation -- shape
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
                # batch = batch,
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
    data_dict = load_everything(args=args, cfg=cfg)

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
        train(args, cfg, data_dict)

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

    
    if args.render_train:
        if args.eval_ari: 
            gt_seg = sorted(glob.glob(f'{cfg.data.datadir}/train/segmentation*.png'))
            gt_seg = [np.array(Image.open(x).resize((256,256),Image.NEAREST)) for x in gt_seg]
            gt_seg = np.stack(gt_seg, 0)[None,...]

            video = sorted(glob.glob(f'{cfg.data.datadir}/train/[0-9][0-9][0-9].png'))
            video = [np.array(Image.open(x).convert('RGB').resize((256,256),Image.BICUBIC)) for x in video]
            video = np.stack(video, 0)[None,...].astype(np.uint8)
            batch = (gt_seg,video)
        else:
            batch = None       
        
        testsavedir = os.path.join(cfg.basedir, cfg.expname, f'render_train_{ckpt_name}')
        os.makedirs(testsavedir, exist_ok=True)
        rgbs, depths = render_viewpoints(
                render_poses=data_dict['poses'][data_dict['i_train']],
                HW=data_dict['HW'][data_dict['i_train']],
                Ks=data_dict['Ks'][data_dict['i_train']],
                frame_times=data_dict['times'][data_dict['i_train']],
                gt_imgs=[data_dict['images'][i].cpu().numpy() for i in data_dict['i_train']],
                savedir=testsavedir,
                eval_ssim=args.eval_ssim, eval_lpips_alex=args.eval_lpips_alex, eval_lpips_vgg=args.eval_lpips_vgg,
                batch=batch,
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
        if args.eval_ari:       
            gt_seg = sorted(glob.glob(f'{cfg.data.datadir}/test/segmentation*.png'))
            gt_seg = [np.array(Image.open(x).resize((256,256),Image.NEAREST)) for x in gt_seg]
            gt_seg = np.stack(gt_seg, 0)[None,...]

            video = sorted(glob.glob(f'{cfg.data.datadir}/test/[0-9][0-9][0-9].png'))
            video = [np.array(Image.open(x).convert('RGB').resize((256,256),Image.BICUBIC)) for x in video]
            video = np.stack(video, 0)[None,...].astype(np.uint8)
            batch = (gt_seg,video)
        else:
            batch = None

        testsavedir = os.path.join(cfg.basedir, cfg.expname, f'render_test_{ckpt_name}')
        os.makedirs(testsavedir, exist_ok=True)
        rgbs, depths = render_viewpoints(
                render_poses=data_dict['poses'][data_dict['i_test']],
                HW=data_dict['HW'][data_dict['i_test']],
                Ks=data_dict['Ks'][data_dict['i_test']],
                frame_times=data_dict['times'][data_dict['i_test']],
                gt_imgs=[data_dict['images'][i].cpu().numpy() for i in data_dict['i_test']],
                savedir=testsavedir,
                eval_ssim=args.eval_ssim, eval_lpips_alex=args.eval_lpips_alex, eval_lpips_vgg=args.eval_lpips_vgg,
                batch=batch,
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
                render_poses=data_dict['poses'][data_dict['i_test']][1].repeat(60,1,1),
                HW=data_dict['HW'][data_dict['i_test']][[0]].repeat(len(data_dict['render_poses']), 0),
                Ks=data_dict['Ks'][data_dict['i_test']][[0]].repeat(len(data_dict['render_poses']), 0),
                frame_times=data_dict['render_times'],
                render_factor=args.render_video_factor,
                savedir=testsavedir,
                stc_data=False,
                bs=args.bs,
                **render_viewpoints_kwargs)
        imageio.mimwrite(os.path.join(testsavedir, 'videofixed.rgb.mp4'), utils.to8b(rgbs), fps=30, quality=8)
        imageio.mimwrite(os.path.join(testsavedir, 'videofixed.depth.mp4'), utils.to8b(1 - depths / np.max(depths)), fps=30, quality=8)

    print('Done')

