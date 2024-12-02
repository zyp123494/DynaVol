_base_ = './default.py'

expname = 'small/8obj'
basedir = './logs'

data = dict(
    datadir='/data/8obj/dynamic',
    dataset_type='dnerf',
    white_bkgd=True,
    half_res = False,
)

fine_train = dict(
    weight_rgbper = 0.01,
    N_iters=20000, 
    lrate_rgb_indepen = 1e-3,
    ray_sampler='random',
     weight_entropy_last=0.01,
     start_ratio = 0.3,  
     increase_until = 2000,
)

fine_model_and_render = dict(
    max_instances=10,
    n_freq=5,      
    n_freq_view=4,                                  # frequency of viewdirs                              # frequency of position of voxel grid
    n_freq_time=5,
    n_freq_t=5,
    n_freq_feat=2,
    add_cam = False,
    alpha_init=5e-4,
    net_width=128,  
    maskout_near_cam_vox = True,
) 

cluster = dict(
    rgb_thresh = 8,#8 by default 15 for metal&3realcmpx
    pos_thresh =2, #2
    traj_thresh = 5,  #5
    use_dbscan =False,  ##True for 3fall3still else False
)

