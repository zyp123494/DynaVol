import os
import time
import functools
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import math
import random
#from lib.tools import compute_rotation_matrix_from_ortho6d
from torch_scatter import segment_coo
from lib.utils import entropy_loss
from timm.models.layers import trunc_normal_

from torch.utils.cpp_extension import load
parent_dir = os.path.dirname(os.path.abspath(__file__))
render_utils_cuda = load(
        name='render_utils_cuda',
        sources=[
            os.path.join(parent_dir, path)
            for path in ['cuda/render_utils.cpp', 'cuda/render_utils_kernel.cu']],
        verbose=True)

total_variation_cuda = load(
        name='total_variation_cuda',
        sources=[
            os.path.join(parent_dir, path)
            for path in ['cuda/total_variation.cpp', 'cuda/total_variation_kernel.cu']],
        verbose=True)

# import lib.networks as networks



import pdb

'''Model'''
class VoxelMlp(torch.nn.Module):
    def __init__(self, xyz_min, xyz_max,
                 num_voxels=0, num_voxels_base=0,add_cam = False,
                 alpha_init=None,
                 mask_cache_path=None, mask_cache_thres=1e-3,
                 fast_color_thres=0,
                 rgbnet_dim=0, rgbnet_direct=False, rgbnet_full_implicit=False,
                 rgbnet_depth=3, rgbnet_width=128,
                 viewbase_pe=4,
                 **kwargs):
        super(VoxelMlp, self).__init__()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.register_buffer('xyz_min', torch.Tensor(xyz_min))
        self.register_buffer('xyz_max', torch.Tensor(xyz_max))
        self.fast_color_thres = fast_color_thres
        self.add_cam = add_cam
        self.skips = kwargs['skips']
        self.n_freq_t = kwargs['n_freq_t']
        self.n_freq_time = kwargs['n_freq_time']
        self.n_freq_feat = kwargs['n_freq_feat']
        self.n_freq_view = kwargs["n_freq_view"]
        print("Add cam:", self.add_cam)

        # determine based grid resolution
        self.num_voxels_base = num_voxels_base
        self.voxel_size_base = ((self.xyz_max - self.xyz_min).prod() / self.num_voxels_base).pow(1/3)
  

        # determine the density bias shift
        self.alpha_init = alpha_init
        self.act_shift = np.log(1/(1-alpha_init) - 1)
        print('voxelMlp: set density bias shift to', self.act_shift)

        # determine init grid resolution
        

        self.dino_channel = 16

        timenet_output = kwargs['voxel_dim']+kwargs['voxel_dim']*2*self.n_freq_feat
        
        self.timenet = nn.Sequential(
            nn.Linear(kwargs['n_freq_time']*2+1, kwargs['net_width']), nn.ReLU(inplace=True),
            nn.Linear(kwargs['net_width'], timenet_output))

        if self.add_cam:
            self.camnet = nn.Sequential(
                nn.Linear(kwargs['n_freq_time']*2+1, kwargs['net_width']), nn.ReLU(inplace=True),
                nn.Linear(kwargs['net_width'], timenet_output))
            print('camnet', self.camnet)

        grid_dim = kwargs['voxel_dim']*3+kwargs['voxel_dim']*2*3*self.n_freq_feat
        input_dim = grid_dim  + 3 + 6 * kwargs['n_freq'] #+ timenet_output

        featurenet_depth = 1
        featurenet_width = kwargs['net_width']

        self.rgb_indepen = nn.Sequential(
            nn.Linear(featurenet_width, featurenet_width),
            nn.ReLU(),
            nn.Linear(featurenet_width, featurenet_width),
            nn.ReLU(),
            nn.Linear(featurenet_width,3)
        )

        
        self.featurenet = nn.Sequential(
            nn.Linear(input_dim, featurenet_width), nn.ReLU(inplace=True),
            *[
                nn.Sequential(nn.Linear(featurenet_width, featurenet_width), nn.ReLU(inplace=True))
                for _ in range(featurenet_depth-1)
            ],
            )

        

        self._set_grid_resolution(num_voxels)

          # local dynamics -- w/ slots
        self._time, self._time_out = self.create_time_net(input_dim=kwargs['n_freq_t']*6+3,
                                                          input_dim_time=timenet_output, D=kwargs['defor_depth'], W=kwargs['net_width'], skips=kwargs['skips'])
        
        
        self._time_inverse, self._time_out_inverse = self.create_time_net(input_dim=kwargs['n_freq_t']*6+3,
                                                          input_dim_time=timenet_output, D=kwargs['defor_depth'], W=kwargs['net_width'], skips=kwargs['skips'])


        

        # init density voxel grid
        self.feature = torch.nn.Parameter(torch.zeros([1, kwargs['voxel_dim'], *self.world_size]))
        self.density = torch.nn.Parameter(torch.zeros([1, 1, *self.world_size]))
        
       

      

        self.dino = torch.nn.Parameter(torch.zeros([1, self.dino_channel, *self.world_size], dtype = torch.float32))
        trunc_normal_(self.dino, std=.02)
        self.dino_mlp = nn.Linear(self.dino_channel, 384)
        
        
    
        self.slots = torch.nn.Parameter(torch.zeros([kwargs['max_instances'], self.dino_channel], dtype = torch.float32))
        trunc_normal_(self.slots, std=.02)

        assert self.slots.shape[0] > 1

        self.temperature = torch.nn.Parameter( 10*torch.ones([1, 1], dtype = torch.float32))  
        
        
        
        
        

        

        self.decoder = Decoder_woslot(n_freq_view=kwargs['n_freq_view'], input_dim=featurenet_width, z_dim=kwargs['net_width'], out_ch=3,cams_dim =timenet_output if self.add_cam else 0)
        

       

       
        self.kwargs = kwargs
      
        print('DynaVol: feature voxel grid', self.feature.shape)
        print('DynaVol: dino voxel grid', self.dino.shape)
        print('DynaVol: timenet mlp', self.timenet)
        print('DynaVol: deformation_net mlp', self._time)
        #print('DynaVol: densitynet mlp', self.densitynet)
        print('DynaVol: featurenet mlp', self.featurenet)
        print('DynaVol: rgbnet mlp', self.decoder)
        print('DynaVol: slots', self.slots.shape)

        self.register_buffer('time_poc', torch.FloatTensor([(2**i) for i in range(self.n_freq_time)]))
        self.register_buffer('grid_poc', torch.FloatTensor([(2**i) for i in range(self.n_freq_feat)]))
        self.register_buffer('pos_poc', torch.FloatTensor([(2**i) for i in range(self.n_freq_t)]))
        self.register_buffer('view_poc', torch.FloatTensor([(2**i) for i in range(self.n_freq_view)]))


        

    def create_time_net(self, input_dim, input_dim_time, D, W, skips, memory=[]):
        layers = [nn.Linear(input_dim + input_dim_time, W)]
        for i in range(D - 1):
            if i in memory:
                raise NotImplementedError
            else:
                layer = nn.Linear

            in_channels = W
            if i in skips:
                in_channels += input_dim

            layers += [layer(in_channels, W)]
        return nn.ModuleList(layers), nn.Linear(W, 3)

    @torch.no_grad()
    def maskout_near_cam_vox(self, cam_o, near):
        self_grid_xyz = torch.stack(torch.meshgrid(
            torch.linspace(self.xyz_min[0], self.xyz_max[0], self.density.shape[2]),
            torch.linspace(self.xyz_min[1], self.xyz_max[1], self.density.shape[3]),
            torch.linspace(self.xyz_min[2], self.xyz_max[2], self.density.shape[4]),
        ), -1)
        nearest_dist = torch.stack([
            (self_grid_xyz.unsqueeze(-2) - co).pow(2).sum(-1).sqrt().amin(-1)
            for co in cam_o.split(100)  # for memory saving
        ]).amin(0)

        nearest_dist = nearest_dist[None, None].expand(-1, self.density.shape[1], -1, -1, -1)
        self.density[nearest_dist <= near] = -100

    def get_mask(self, rays_o, rays_d, near, far, stepsize, **render_kwargs):
        '''Check whether the rays hit the geometry or not'''
        shape = rays_o.shape[:-1]
        rays_o = rays_o.reshape(-1, 3).contiguous()
        rays_d = rays_d.reshape(-1, 3).contiguous()
        stepdist = stepsize * self.voxel_size
        ray_pts, mask_outbbox, ray_id = render_utils_cuda.sample_pts_on_rays(
                rays_o, rays_d, self.xyz_min, self.xyz_max, near, far, stepdist)[:3]
        mask_inbbox = ~mask_outbbox
        hit = torch.zeros([len(rays_o)], dtype=torch.bool)
        hit[ray_id[mask_inbbox]] = 1
        return hit.reshape(shape)

    def get_dynamics(self,frame_times):
        import math
        with torch.no_grad():
            res = []
            ray_pts = self.world_pos[0].flatten(start_dim=1).permute(1,0)
            ray_pts_emb = sin_emb(ray_pts,self.pos_poc)
            for frame_time in frame_times:
                print(frame_time)
                frame_time = torch.tensor(frame_time).to(self.feature.device).reshape([1,1])
                frame_time = sin_emb(frame_time, n_freq = self.time_poc)
                times_feature = self.timenet(frame_time) 
                times_feature = times_feature.expand([ray_pts.shape[0],-1])
                dx = self.query_time(ray_pts_emb, times_feature, self._time_inverse, self._time_out_inverse)
                dx = dx.permute(1,0).reshape([1,3,*self.feature.shape[2:]])
                res.append(dx.cpu())

            res = torch.cat(res,dim = 0)
            return res

    
           
            
 






    def query_time(self, pts_sim, t, net, net_final, pdb_flag=0):
        h = torch.cat([pts_sim, t], dim=-1)
        for i, l in enumerate(net):
            h = net[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([pts_sim, h], -1)

        return net_final(h)

    def _set_grid_resolution(self, num_voxels):#, num_voxels_dino=None):
        self.num_voxels = num_voxels
        self.voxel_size = ((self.xyz_max - self.xyz_min).prod() / num_voxels).pow(1/3)
        self.world_size = ((self.xyz_max - self.xyz_min) / self.voxel_size).long()
        self.voxel_size_ratio = self.voxel_size / self.voxel_size_base
        # define the xyz positions of grid
        self.world_pos = torch.zeros([1, 3, *self.world_size])
        xcoord = torch.linspace(start=self.xyz_min[0], end=self.xyz_max[0], steps=self.world_size[0])
        ycoord = torch.linspace(start=self.xyz_min[1], end=self.xyz_max[1], steps=self.world_size[1])
        zcoord = torch.linspace(start=self.xyz_min[2], end=self.xyz_max[2], steps=self.world_size[2])
        grid = torch.meshgrid(xcoord, ycoord, zcoord)
        for i in range(3):
            self.world_pos[0, i, :, :, :] = grid[i]    
        print('voxelMlp: voxel_size      ', self.voxel_size)
        print('voxelMlp: world_size      ', self.world_size)
        print('voxelMlp: voxel_size_base ', self.voxel_size_base)
        print('voxelMlp: voxel_size_ratio', self.voxel_size_ratio)

    def get_kwargs(self):
        return {
            'xyz_min': self.xyz_min.cpu().numpy(),
            'xyz_max': self.xyz_max.cpu().numpy(),
            'num_voxels': self.num_voxels,
            'num_voxels_base': self.num_voxels_base,
            'alpha_init': self.alpha_init,
            'act_shift': self.act_shift,
            'voxel_size_ratio': self.voxel_size_ratio,
            'fast_color_thres': self.fast_color_thres,
            'n_freq': self.kwargs['n_freq'],
            'n_freq_view': self.kwargs['n_freq_view'],
            "max_instances": self.kwargs['max_instances'],
            "n_freq_t": self.kwargs['n_freq_t'],
            "n_freq_time": self.kwargs['n_freq_time'],
            "n_freq_feat":self.n_freq_feat,
            "defor_depth": self.kwargs['defor_depth'],
            "net_width": self.kwargs['net_width'],
            "skips": self.kwargs['skips'],
            "voxel_dim":self.kwargs['voxel_dim'],
            "add_cam":self.add_cam,
        }

  

    @torch.no_grad()
    def scale_volume_grid(self, num_voxels):
        print('voxelMlp: scale_volume_grid start')
        ori_world_size = self.world_size
        self._set_grid_resolution(num_voxels)
        print('voxelMlp: scale_volume_grid scale world_size from', ori_world_size, 'to', self.world_size)

        self.feature = torch.nn.Parameter(
            F.interpolate(self.feature.data, size=tuple(self.world_size), mode='trilinear', align_corners=True))
        
        self.dino = torch.nn.Parameter(
            F.interpolate(self.dino.data, size=tuple(self.world_size), mode='trilinear', align_corners=True))
        
        self.density = torch.nn.Parameter(
            F.interpolate(self.density.data, size=tuple(self.world_size), mode='trilinear', align_corners=True))


    

        print('voxelMlp: scale_volume_grid finish')


    def feature_total_variation_add_grad(self, weight, dense_mode):
        weight = weight * self.world_size.max() / 128
        total_variation_cuda.total_variation_add_grad(
            self.feature.float(), self.feature.grad.float(), weight, weight, weight, dense_mode)
    
    def dino_total_variation_add_grad(self, weight, dense_mode):
        if self.dino.grad is None:
            return
        weight = weight * self.world_size.max() / 128
        total_variation_cuda.total_variation_add_grad(
            self.dino.float(), self.dino.grad.float(), weight, weight, weight, dense_mode)


    def density_total_variation_add_grad(self, weight, dense_mode):
        weight = weight * self.world_size.max() / 128
        total_variation_cuda.total_variation_add_grad(
            self.density.float(), self.density.grad.float(), weight, weight, weight, dense_mode)

   
    def activate_density_multiple(self, density, interval=None, dens_noise=0):
        interval = interval if interval is not None else self.voxel_size_ratio
        raw_masks = F.softplus(density + self.act_shift, True)

        raw_sigma = raw_masks + dens_noise * torch.randn_like(raw_masks)

        masks = raw_masks / (raw_masks.sum(dim=-1)[:,None] + 1e-5)  # PxK

        sigma = (raw_sigma * masks).sum(dim=-1)
        alpha = 1 - torch.exp(-sigma * interval)
        return alpha

    
    def grid_sampler(self, xyz, *grids, mode=None, align_corners=True):
        '''Wrapper for the interp operation'''
        mode = 'bilinear'
        shape = xyz.shape[:-1]
        xyz = xyz.reshape(1,1,1,-1,3)
        ind_norm = ((xyz - self.xyz_min) / (self.xyz_max - self.xyz_min)).flip((-1,)) * 2 - 1
        ret_lst = [
            # TODO: use `rearrange' to make it readable
            F.grid_sample(grid.float(), ind_norm, mode=mode, align_corners=align_corners).reshape(grid.shape[1],-1).T.reshape(*shape,grid.shape[1])
            for grid in grids
        ]
        for i in range(len(grids)):
            if ret_lst[i].shape[-1] == 1:
                ret_lst[i] = ret_lst[i].squeeze(-1)
        if len(ret_lst) == 1:
            return ret_lst[0]
        return ret_lst

    def grid_sampler_imp(self, xyz, importance=None, vq=None):
        '''
        xyz: global coordinates to query
        '''
        shape = xyz.shape[:-1]
        xyz = xyz.reshape(1,1,1,-1,3)
        ind_norm = ((xyz - self.xyz_min) / (self.xyz_max - self.xyz_min)).flip((-1,)) * 2 - 1
        
        if importance is not None:
            self.channels=1
            sampled_importance = F.grid_sample(importance, ind_norm, mode='bilinear', align_corners=False)
            sampled_importance = sampled_importance.reshape(self.channels,-1).T.reshape(*shape,self.channels)
            if self.channels == 1:
                sampled_importance = sampled_importance.squeeze(-1)

       
       
        if importance is not None:
            return sampled_importance
        else:
            raise NotImplementedError

    def hit_coarse_geo(self, rays_o, rays_d, near, far, stepsize, **render_kwargs):
        '''Check whether the rays hit the solved coarse geometry or not'''
        shape = rays_o.shape[:-1]
        rays_o = rays_o.reshape(-1, 3).contiguous()
        rays_d = rays_d.reshape(-1, 3).contiguous()
        stepdist = stepsize * self.voxel_size
        ray_pts, mask_outbbox, ray_id = render_utils_cuda.sample_pts_on_rays(
                rays_o, rays_d, self.xyz_min, self.xyz_max, near, far, stepdist)[:3]
        mask_inbbox = ~mask_outbbox
        hit = torch.zeros([len(rays_o)], dtype=torch.bool)
        hit[ray_id[mask_inbbox][self.mask_cache(ray_pts[mask_inbbox])]] = 1
        return hit.reshape(shape)
    
    def mult_dist_interp(self, ray_pts_delta):

        x_pad = math.ceil((self.feature.shape[2]-1)/4.0)*4-self.feature.shape[2]+1
        y_pad = math.ceil((self.feature.shape[3]-1)/4.0)*4-self.feature.shape[3]+1
        z_pad = math.ceil((self.feature.shape[4]-1)/4.0)*4-self.feature.shape[4]+1
        grid = F.pad(self.feature.float(),(0,z_pad,0,y_pad,0,x_pad))
        # three 
        vox_l = self.grid_sampler(ray_pts_delta, grid)
        vox_m = self.grid_sampler(ray_pts_delta, grid[:,:,::2,::2,::2])
        vox_s = self.grid_sampler(ray_pts_delta, grid[:,:,::4,::4,::4])
        vox_feature = torch.cat((vox_l,vox_m,vox_s),-1)

        if len(vox_feature.shape)==1:
            vox_feature_flatten = vox_feature.unsqueeze(0)
        else:
            vox_feature_flatten = vox_feature
        
        return vox_feature_flatten

    def sample_ray(self, rays_o, rays_d, near, far, stepsize, is_train=False, **render_kwargs):
        '''Sample query points on rays.
        All the output points are sorted from near to far.
        Input:
            rays_o, rayd_d:   both in [N, 3] indicating ray configurations.
            near, far:        the near and far distance of the rays.
            stepsize:         the number of voxels of each sample step.
        Output:
            ray_pts:          [M, 3] storing all the sampled points.
            ray_id:           [M]    the index of the ray of each point.
            step_id:          [M]    the i'th step on a ray of each point.
        '''
        rays_o = rays_o.contiguous()
        rays_d = rays_d.contiguous()
        stepdist = stepsize * self.voxel_size
        ray_pts, mask_outbbox, ray_id, step_id, N_steps, t_min, t_max = render_utils_cuda.sample_pts_on_rays(
            rays_o, rays_d, self.xyz_min, self.xyz_max, near, far, stepdist)
        mask_inbbox = ~mask_outbbox

        n_max = ray_pts.shape[0] // t_min.shape[0]
        t = torch.linspace(t_min[0], t_max[0], steps = math.ceil(ray_pts.shape[0] / t_min.shape[0])).to(ray_pts.device).unsqueeze(0).expand(t_min.shape[0],-1)
        t = t.reshape([-1])[:ray_pts.shape[0]]
       

        ray_pts = ray_pts[mask_inbbox]
        ray_id = ray_id[mask_inbbox]
        step_id = step_id[mask_inbbox]
        t = t[mask_inbbox]
        
        return ray_pts, ray_id, step_id, t, n_max



    def get_rgb(self):
        with torch.no_grad():        
            rgbs = []
            ray_pts = self.world_pos[0].flatten(start_dim=1).permute(1,0)
            ray_pts_emb = sin_emb(ray_pts, self.pos_poc)
            vox_feature_flatten =self.mult_dist_interp(ray_pts)
            vox_feature_flatten_emb = sin_emb(vox_feature_flatten, n_freq = self.grid_poc)
            h_feature = self.featurenet(torch.cat((vox_feature_flatten_emb, ray_pts_emb), -1))

                
            rgb = self.rgb_indepen(h_feature).sigmoid()

            return rgb.permute(1,0).reshape(self.world_pos.shape)
    


    def forward(self, rays_o, rays_d, viewdirs, frame_time,cam_sel=None,   bg_points_sel=None,global_step=None,mask = None,slot_idx = -1,**render_kwargs):
        '''Volume rendering
        @rays_o:   [N, 3] the starting point of the N shooting rays.
        @rays_d:   [N, 3] the shooting direction of the N rays.
        @viewdirs: [N, 3] viewing direction to compute positional embedding for MLP.
        '''
        assert len(rays_o.shape)==2 and rays_o.shape[-1]==3, 'Only suuport point queries in [N, 3] format'
        
        #save_dict = {}


        ret_dict = {}
        N = len(rays_o)
        frame_time_emb = sin_emb(frame_time, n_freq = self.time_poc)
        times_feature = self.timenet(frame_time_emb)
        viewdirs_emb = sin_emb(viewdirs, self.view_poc)

        if self.add_cam:
            cam_emb= sin_emb(cam_sel, n_freq=self.time_poc)
            cams_feature=self.camnet(cam_emb)

        # sample points on rays
        ray_pts, ray_id, step_id, t,n_max = self.sample_ray(
            rays_o=rays_o, rays_d=rays_d, is_train=global_step is not None, **render_kwargs)
        interval = render_kwargs['stepsize'] * self.voxel_size_ratio

        cycle_loss = 0
        
    
        ray_pts_emb = sin_emb(ray_pts, self.pos_poc)
        
        dx = self.query_time(ray_pts_emb, times_feature[ray_id], self._time, self._time_out)
        ray_pts_ = ray_pts + dx

      
        
        
         
          
        if bg_points_sel is not None:
            bg_points_sel_emb = sin_emb(bg_points_sel, self.pos_poc)
            bg_points_sel_delta = self.query_time(bg_points_sel_emb, times_feature[:(bg_points_sel_emb.shape[0])], self._time, self._time_out)
            ret_dict.update({'bg_points_delta': bg_points_sel + bg_points_sel_delta})
        
        
       
        vox_feature_flatten=self.mult_dist_interp(ray_pts_)
      
        dino_feature = self.grid_sampler(ray_pts_.detach(), self.dino)
        vox_feature_flatten_emb = sin_emb(vox_feature_flatten, n_freq = self.grid_poc)
        
        ray_pts_delta_emb = sin_emb(ray_pts_, n_freq = self.pos_poc)
       
        h_feature = self.featurenet(torch.cat((vox_feature_flatten_emb, ray_pts_delta_emb), -1))
      
        
        dx_inverse = self.query_time(ray_pts_delta_emb.detach(), times_feature[ray_id], self._time_inverse, self._time_out_inverse)
        cycle_loss = F.mse_loss(ray_pts_.detach() + dx_inverse, ray_pts.detach())
    

        density = self.grid_sampler(ray_pts_, self.density).unsqueeze(-1)


        rgb_direct= self.rgb_indepen(h_feature.detach()).sigmoid()






        
        slots_prob = None
        segs = None
        density = F.softplus(density + self.act_shift).squeeze(-1)
        
      
        if mask is not None:
            mask_ori = self.grid_sampler(ray_pts_,mask)  #[P,K]
            slots_prob = mask_ori.clone()
            if slot_idx != -1:
                mask = mask_ori[:,slot_idx]
                density = density * mask

        alpha = 1 - torch.exp(-density * interval)
        if self.fast_color_thres > 0:
            mask = (alpha > self.fast_color_thres)
            h_feature = h_feature[mask]
            ray_id_ = ray_id[mask]
            step_id = step_id[mask]
            alpha = alpha[mask]
            dino_feature = dino_feature[mask]
            ray_pts_ = ray_pts_[mask]
            rgb_direct = rgb_direct[mask]
            t  = t[mask]
            if slots_prob is not None:
                slots_prob = slots_prob[mask]

        # compute accumulated transmittance
        weights, alphainv_last = Alphas2Weights.apply(alpha, ray_id_, N)
        if self.fast_color_thres > 0:
            mask = (weights > self.fast_color_thres)
            weights = weights[mask]
            h_feature = h_feature[mask]
            alpha = alpha[mask]
            ray_id_ = ray_id_[mask]
            step_id = step_id[mask]
            rgb_direct = rgb_direct[mask]
            t = t[mask]
            ray_pts_ = ray_pts_[mask]
            dino_feature = dino_feature[mask]
            if slots_prob is not None:
                slots_prob = slots_prob[mask]
                segs = slots_prob.clone()

       
        viewdirs_emb = viewdirs_emb[ray_id_]
        if self.add_cam:
            rgb = self.decoder(h_feature, viewdirs_emb, ray_id_, cams_feature) 
        else:
            rgb = self.decoder(h_feature, viewdirs_emb, ray_id_) 

       

   

        # Ray marching
        rgb_marched = segment_coo(
                src=(weights.unsqueeze(-1) * rgb),
                index=ray_id_,
                out=torch.zeros([N, 3]),
                reduce='sum')
        rgb_marched += (alphainv_last.unsqueeze(-1) * render_kwargs['bg'])

        rgb_direct = segment_coo(
                src=(weights.unsqueeze(-1).detach() * rgb_direct),
                index=ray_id_,
                out=torch.zeros([N, 3]),
                reduce='sum')
        rgb_direct += (alphainv_last.detach().unsqueeze(-1) * render_kwargs['bg'])



        dino_marched = None
        slots_prob = None
        if global_step is None:
            logits = dino_feature @ self.slots.permute(1,0) #[N,K]
            slots_prob = F.softmax(logits / F.softplus(self.temperature), dim = -1)
            dino_marched = slots_prob @ self.slots  #[N,C]

            dino_marched = segment_coo(
                    src=(weights.unsqueeze(-1).detach() * dino_marched),
                    index=ray_id_,
                    out=torch.zeros([N, dino_feature.shape[-1]]),
                    reduce='sum')

            slots_prob = segment_coo(
                    src=(weights.unsqueeze(-1).detach() * slots_prob),
                    index=ray_id_,
                    out=torch.zeros([N, slots_prob.shape[-1]]),
                    reduce='sum')

            
        
        s = 1 - 1/(1+t)

        
       
        ret_dict.update({
            'alphainv_last': alphainv_last,
            'weights': weights,
            'rgb_marched': rgb_marched,
            'raw_alpha': alpha,
            'raw_rgb': rgb,
            "dino":dino_marched,
            'ray_id': ray_id_,
            "slots_prob":slots_prob,
            'cycle_loss':cycle_loss,
            "rgb_direct":rgb_direct,
            "n_max": n_max,
            's':s,
           
            
        })

       
        if render_kwargs.get('render_depth', True):
            with torch.no_grad():
                depth = segment_coo(
                        src=(weights * step_id),
                        index=ray_id_,
                        out=torch.zeros([N]),
                        reduce='sum')
            ret_dict.update({'depth': depth})

        if render_kwargs.get('segmentation', False):
            
            contribution = segment_coo(
                src=(weights.unsqueeze(-1) * segs),
                index=ray_id_,
                out=torch.zeros([N, segs.shape[1]]),
                reduce='sum') # [M,slots]
            
            # seg_contri = torch.cat([alphainv_last.unsqueeze(-1), contribution], dim=-1) # [N, slots+1]
            seg_contri = torch.cat([torch.zeros_like(contribution)[...,:1], contribution], dim=-1) # [N, slots+1]
            segmentation = seg_contri / (seg_contri.sum(-1,keepdim=True) + 1e-5)
           
            ret_dict.update({'segmentation': segmentation})   #[N]
        
        
        return ret_dict
    def forward_imp(self, rays_o, rays_d, viewdirs, frame_time,cam_sel=None,   bg_points_sel=None,pseudo_grid = None, global_step=None, **render_kwargs):
        '''Volume rendering
        @rays_o:   [N, 3] the starting point of the N shooting rays.
        @rays_d:   [N, 3] the shooting direction of the N rays.
        @viewdirs: [N, 3] viewing direction to compute positional embedding for MLP.
        '''
        assert len(rays_o.shape)==2 and rays_o.shape[-1]==3, 'Only suuport point queries in [N, 3] format'

        ret_dict = {}
        N = len(rays_o)

        # sample points on rays
        ray_pts, ray_id, step_id ,_,_= self.sample_ray(
                rays_o=rays_o, rays_d=rays_d, **render_kwargs)
        interval = render_kwargs['stepsize'] * self.voxel_size_ratio

 

        #print(frame_time,frame_time.shape)
        if self.add_cam:
            cam_emb= sin_emb(cam_sel, n_freq=self.time_poc)
            cams_feature=self.camnet(cam_emb)
        
    
        frame_time = sin_emb(frame_time, n_freq = self.time_poc)
        times_feature = self.timenet(frame_time) 
          
        ray_pts_emb = sin_emb(ray_pts, self.pos_poc)
        
        times_feature = times_feature[ray_id,:]
        dx = self.query_time(ray_pts_emb, times_feature, self._time, self._time_out)
        ray_pts_ = ray_pts + dx

        sampled_pseudo_grid = self.grid_sampler_imp(ray_pts_, importance = pseudo_grid)

        vox_feature_flatten=self.mult_dist_interp(ray_pts_)
        vox_feature_flatten_emb = sin_emb(vox_feature_flatten, n_freq =self.grid_poc)
        
        ray_pts_delta_emb = sin_emb(ray_pts_, n_freq = self.pos_poc)
        
        density = self.grid_sampler(ray_pts_, self.density).unsqueeze(-1)


        
        density = F.softplus(density + self.act_shift, True).squeeze(-1)
        
     
        alpha = 1 - torch.exp(-density * interval)
        if self.fast_color_thres > 0:
            mask = (alpha > self.fast_color_thres)
            ray_pts_ = ray_pts_[mask]
            ray_id_ = ray_id[mask]
            step_id = step_id[mask]
            density = density[mask]
            alpha = alpha[mask]
            sampled_pseudo_grid = sampled_pseudo_grid[mask]
          

        # compute accumulated transmittance
        weights, alphainv_last = Alphas2Weights.apply(alpha, ray_id_, N)
        if self.fast_color_thres > 0:
            mask = (weights > self.fast_color_thres)
            weights = weights[mask]
            alpha = alpha[mask]
            ray_pts_ = ray_pts_[mask]
            ray_id_ = ray_id_[mask]
            step_id = step_id[mask]
            density = density[mask]
            sampled_pseudo_grid = sampled_pseudo_grid[mask]
           
        

        ret_dict.update({
            'sampled_pseudo_grid':sampled_pseudo_grid,
            "weights":weights
        })
        return ret_dict

    
    def forward_dino(self, rays_o, rays_d,  frame_time,  global_step=None,**render_kwargs):
        '''Volume rendering
        @rays_o:   [N, 3] the starting point of the N shooting rays.
        @rays_d:   [N, 3] the shooting direction of the N rays.
        @viewdirs: [N, 3] viewing direction to compute positional embedding for MLP.
        '''
        assert len(rays_o.shape)==2 and rays_o.shape[-1]==3, 'Only suuport point queries in [N, 3] format'

        ret_dict = {}
        N = len(rays_o)

        # sample points on rays
        ray_pts, ray_id, step_id ,_,_= self.sample_ray(
            rays_o=rays_o, rays_d=rays_d, is_train=global_step is not None, **render_kwargs)
        interval = render_kwargs['stepsize'] * self.voxel_size_ratio


        
       
        frame_time = sin_emb(frame_time, n_freq = self.time_poc)
        times_feature = self.timenet(frame_time) 

        ray_pts_emb = sin_emb(ray_pts, self.pos_poc)
       
       
        dx = self.query_time(ray_pts_emb, times_feature[ray_id,:], self._time, self._time_out)
   
        ray_pts_ = ray_pts + dx

       
        density = self.grid_sampler(ray_pts_, self.density).unsqueeze(-1)

        dino_feature = self.grid_sampler(ray_pts_.detach(), self.dino)


        density = F.softplus(density + self.act_shift, True).squeeze(-1)
      
     
        

   
        alpha = 1 - torch.exp(-density * interval)
        if self.fast_color_thres > 0:
            mask = (alpha > self.fast_color_thres)
            ray_pts_ = ray_pts_[mask]
            ray_id_ = ray_id[mask]
            step_id = step_id[mask]
            density = density[mask]
            alpha = alpha[mask]
            dino_feature = dino_feature[mask]
   

        # compute accumulated transmittance
        weights, alphainv_last = Alphas2Weights.apply(alpha, ray_id_, N)
        if self.fast_color_thres > 0:
            mask = (weights > self.fast_color_thres)
            weights = weights[mask]
            alpha = alpha[mask]
            ray_pts_ = ray_pts_[mask]
            ray_id_ = ray_id_[mask]
            step_id = step_id[mask]
            density = density[mask]
            dino_feature = dino_feature[mask]
            
        
        
        logits = dino_feature @ self.slots.permute(1,0) #[N,K]
        slots_prob = F.softmax(logits / F.softplus(self.temperature), dim = -1)
     

        dino_marched = slots_prob @ self.slots  #[N,C]

        dino_marched = segment_coo(
                src=(weights.unsqueeze(-1).detach() * dino_marched),
                index=ray_id_,
                out=torch.zeros([N, dino_feature.shape[-1]]),
                reduce='sum')

        slots_prob = segment_coo(
                src=(weights.unsqueeze(-1).detach() * slots_prob),
                index=ray_id_,
                out=torch.zeros([N, slots_prob.shape[-1]]),
                reduce='sum')
   

        loss_entropy = entropy_loss(slots_prob)

     

       
        ret_dict.update({
        
            "dino":dino_marched,
            "slots_prob":slots_prob,
            "loss_entropy":loss_entropy,
        })

    
        
        
        return ret_dict





class Alphas2Weights(torch.autograd.Function):
    @staticmethod
    def forward(ctx, alpha, ray_id, N):
        weights, T, alphainv_last, i_start, i_end = render_utils_cuda.alpha2weight(alpha, ray_id, N)
        if alpha.requires_grad:
            ctx.save_for_backward(alpha, weights, T, alphainv_last, i_start, i_end)
            ctx.n_rays = N
        return weights, alphainv_last

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_weights, grad_last):
        alpha, weights, T, alphainv_last, i_start, i_end = ctx.saved_tensors
        grad = render_utils_cuda.alpha2weight_backward(
                alpha, weights, T, alphainv_last,
                i_start, i_end, ctx.n_rays, grad_weights, grad_last)
        return grad, None, None


class Decoder_woslot(nn.Module):
    def __init__(self, n_freq_view=3, input_dim=33+64,z_dim=64,  out_ch=3,cams_dim = 128):
        super().__init__()
        self.n_freq_view = n_freq_view
        self.out_ch = out_ch

        self.feature_linears = nn.Linear(input_dim, z_dim)
        self.views_linears = nn.Sequential(nn.Linear(z_dim+6*n_freq_view+3 + cams_dim, z_dim//2),nn.ReLU(),nn.Linear(z_dim//2, out_ch))


    def forward(self, feature, sampling_view,   ray_id,cams_feature = None):
       
        feature = self.feature_linears(feature)
        
      

        if cams_feature is not None:
            feature_views = torch.cat([feature, sampling_view, cams_feature[ray_id,:]],dim=-1)
        else:
            feature_views = torch.cat([feature, sampling_view],dim=-1)

        outputs = self.views_linears(feature_views)

        
        return torch.sigmoid(outputs)





''' Ray and batch
'''
def get_rays(H, W, K, c2w, inverse_y=False, flip_x=False ,flip_y=False, mode='center'):
    i, j = torch.meshgrid(
        torch.linspace(0, W-1, W, device=c2w.device),
        torch.linspace(0, H-1, H, device=c2w.device))  # pytorch's meshgrid has indexing='ij'
    i = i.t().float()
    j = j.t().float()
    if mode == 'lefttop':
        pass
    elif mode == 'center':
        i, j = i+0.5, j+0.5
    elif mode == 'random':
        i = i+torch.rand_like(i)
        j = j+torch.rand_like(j)
    else:
        raise NotImplementedError

    if flip_x:
        i = i.flip((1,))
    if flip_y:
        j = j.flip((0,))
    if inverse_y:
        dirs = torch.stack([(i-K[0][2])/K[0][0], (j-K[1][2])/K[1][1], torch.ones_like(i)], -1)
    else:
        dirs = torch.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -torch.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3,3].expand(rays_d.shape)
    return rays_o, rays_d

def get_rays_np(H, W, K, c2w):
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -np.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = np.broadcast_to(c2w[:3,3], np.shape(rays_d))
    return rays_o, rays_d

def ndc_rays(H, W, focal, near, rays_o, rays_d):
    # Shift ray origins to near plane
    t = -(near + rays_o[...,2]) / rays_d[...,2]
    rays_o = rays_o + t[...,None] * rays_d

    # Projection
    o0 = -1./(W/(2.*focal)) * rays_o[...,0] / rays_o[...,2]
    o1 = -1./(H/(2.*focal)) * rays_o[...,1] / rays_o[...,2]
    o2 = 1. + 2. * near / rays_o[...,2]

    d0 = -1./(W/(2.*focal)) * (rays_d[...,0]/rays_d[...,2] - rays_o[...,0]/rays_o[...,2])
    d1 = -1./(H/(2.*focal)) * (rays_d[...,1]/rays_d[...,2] - rays_o[...,1]/rays_o[...,2])
    d2 = -2. * near / rays_o[...,2]

    rays_o = torch.stack([o0,o1,o2], -1)
    rays_d = torch.stack([d0,d1,d2], -1)

    return rays_o, rays_d

def get_rays_of_a_view(H, W, K, c2w, ndc, mode='center'):
    rays_o, rays_d = get_rays(H, W, K, c2w, mode=mode)
    viewdirs = rays_d / rays_d.norm(dim=-1, keepdim=True)
    if ndc:
        rays_o, rays_d = ndc_rays(H, W, K[0][0], 1., rays_o, rays_d)
    return rays_o, rays_d, viewdirs

@torch.no_grad()
def get_training_rays(rgb_tr, times,train_poses, HW, Ks, ndc):
    print('get_training_rays: start')
    assert len(np.unique(HW, axis=0)) == 1
    assert len(np.unique(Ks.reshape(len(Ks),-1), axis=0)) == 1
    assert len(rgb_tr) == len(train_poses) and len(rgb_tr) == len(Ks) and len(rgb_tr) == len(HW)
    H, W = HW[0]
    K = Ks[0]
    eps_time = time.time()
    rays_o_tr = torch.zeros([len(rgb_tr), H, W, 3], device=rgb_tr.device)
    rays_d_tr = torch.zeros([len(rgb_tr), H, W, 3], device=rgb_tr.device)
    viewdirs_tr = torch.zeros([len(rgb_tr), H, W, 3], device=rgb_tr.device)
    times_tr = torch.ones([len(rgb_tr), H, W, 1], device=rgb_tr.device)

    imsz = [1] * len(rgb_tr)
    for i, c2w in enumerate(train_poses):
        rays_o, rays_d, viewdirs = get_rays_of_a_view(
                H=H, W=W, K=K, c2w=c2w, ndc=ndc,)
        rays_o_tr[i].copy_(rays_o.to(rgb_tr.device))
        rays_d_tr[i].copy_(rays_d.to(rgb_tr.device))
        viewdirs_tr[i].copy_(viewdirs.to(rgb_tr.device))
        times_tr[i] = times_tr[i]*times[i]
        del rays_o, rays_d, viewdirs
    eps_time = time.time() - eps_time
    print('get_training_rays: finish (eps time:', eps_time, 'sec)')
    return rgb_tr, times_tr,rays_o_tr, rays_d_tr, viewdirs_tr, imsz


@torch.no_grad()
def get_training_rays_flatten(rgb_tr_ori, times,train_poses, HW, Ks, ndc):
    print('get_training_rays_flatten: start')
    assert len(rgb_tr_ori) == len(train_poses) and len(rgb_tr_ori) == len(Ks) and len(rgb_tr_ori) == len(HW)
    eps_time = time.time()
    DEVICE = rgb_tr_ori[0].device
    N = sum(im.shape[0] * im.shape[1] for im in rgb_tr_ori)
    rgb_tr = torch.zeros([N,3], device=DEVICE)
    rays_o_tr = torch.zeros_like(rgb_tr)
    rays_d_tr = torch.zeros_like(rgb_tr)
    viewdirs_tr = torch.zeros_like(rgb_tr)
    times_tr=torch.ones([N,1], device=DEVICE)
    times=times.unsqueeze(-1)
    imsz = []
    top = 0
    for c2w, img, (H, W), K ,time_one in zip(train_poses, rgb_tr_ori, HW, Ks,times):
        assert img.shape[:2] == (H, W)
        rays_o, rays_d, viewdirs = get_rays_of_a_view(
                H=H, W=W, K=K, c2w=c2w, ndc=ndc,)
        n = H * W
        times_tr[top:top+n]=times_tr[top:top+n]*time_one
        rgb_tr[top:top+n].copy_(img.flatten(0,1))
        rays_o_tr[top:top+n].copy_(rays_o.flatten(0,1).to(DEVICE))
        rays_d_tr[top:top+n].copy_(rays_d.flatten(0,1).to(DEVICE))
        viewdirs_tr[top:top+n].copy_(viewdirs.flatten(0,1).to(DEVICE))
        imsz.append(n)
        top += n
    assert top == N
    eps_time = time.time() - eps_time
    print('get_training_rays_flatten: finish (eps time:', eps_time, 'sec)')
    return rgb_tr, times_tr,rays_o_tr, rays_d_tr, viewdirs_tr, imsz


@torch.no_grad()
def get_training_rays_in_maskcache_sampling(rgb_tr_ori, times,train_poses, HW, Ks, ndc, model, render_kwargs):
    print('get_training_rays_in_maskcache_sampling: start')
    assert len(rgb_tr_ori) == len(train_poses) and len(rgb_tr_ori) == len(Ks) and len(rgb_tr_ori) == len(HW)
    CHUNK = 64
    DEVICE = rgb_tr_ori[0].device
    eps_time = time.time()
    N = sum(im.shape[0] * im.shape[1] for im in rgb_tr_ori)
    rgb_tr = torch.zeros([N,3], device=DEVICE)
    rays_o_tr = torch.zeros_like(rgb_tr)
    rays_d_tr = torch.zeros_like(rgb_tr)
    viewdirs_tr = torch.zeros_like(rgb_tr)
    times_tr = torch.ones([N,1], device=DEVICE)
    times = times.unsqueeze(-1)
    imsz = []
    top = 0
    for c2w, img, (H, W), K ,time_one in zip(train_poses, rgb_tr_ori, HW, Ks,times):
        assert img.shape[:2] == (H, W)
        rays_o, rays_d, viewdirs = get_rays_of_a_view(
                H=H, W=W, K=K, c2w=c2w, ndc=ndc,)
        mask = torch.empty(img.shape[:2], device=DEVICE, dtype=torch.bool)
        for i in range(0, img.shape[0], CHUNK):
            mask[i:i+CHUNK] = model.get_mask(
                    rays_o=rays_o[i:i+CHUNK], rays_d=rays_d[i:i+CHUNK], **render_kwargs).to(DEVICE)
        n = mask.sum()
        times_tr[top:top+n]=times_tr[top:top+n]*time_one
        rgb_tr[top:top+n].copy_(img[mask])
        rays_o_tr[top:top+n].copy_(rays_o[mask].to(DEVICE))
        rays_d_tr[top:top+n].copy_(rays_d[mask].to(DEVICE))
        viewdirs_tr[top:top+n].copy_(viewdirs[mask].to(DEVICE))
        imsz.append(n)
        top += n

    print('get_training_rays_in_maskcache_sampling: ratio', top / N)
    rgb_tr = rgb_tr[:top]
    rays_o_tr = rays_o_tr[:top]
    rays_d_tr = rays_d_tr[:top]
    viewdirs_tr = viewdirs_tr[:top]
    eps_time = time.time() - eps_time
    print('get_training_rays_in_maskcache_sampling: finish (eps time:', eps_time, 'sec)')
    return rgb_tr, times_tr,rays_o_tr, rays_d_tr, viewdirs_tr, imsz



def sin_emb(input_data,n_freq):
    #n_freq = torch.FloatTensor([(2**i) for i in range(n_freq)]).to(input_data.device)

    input_data_emb = (input_data.unsqueeze(-1) * n_freq).flatten(-2)
    input_data_sin = input_data_emb.sin()
    input_data_cos = input_data_emb.cos()
    input_data_emb = torch.cat([input_data, input_data_sin,input_data_cos], -1)
    return input_data_emb

def batch_indices_generator(N, BS):
    # torch.randperm on cuda produce incorrect results in my machine
    idx, top = torch.LongTensor(np.random.permutation(N)), 0
    while True:
        if top + BS > N:
            idx, top = torch.LongTensor(np.random.permutation(N)), 0
        yield idx[top:top+BS]
        top += BS