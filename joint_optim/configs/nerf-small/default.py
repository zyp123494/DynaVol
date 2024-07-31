from copy import deepcopy

expname = None                    # experiment name
basedir = './logs/'               # where to store ckpts and logs

''' Template of data options
'''
data = dict(
    datadir=None,                 # path to dataset root folder
    dataset_type=None,            
    load2gpu_on_the_fly=False,    # do not load all images into gpu (to save gpu memory)
    testskip=1,                   # subsample testset to preview results
    white_bkgd=False,             # use white background (note that some dataset don't provide alpha and with blended bg color)
    half_res=True,              
    factor=4,                     
    ndc=False,                    # use ndc coordinate (only for forward-facing; not support yet)
    spherify=False,               # inward-facing
    llffhold=8,                   # testsplit
    load_depths=False,            # load depth
    use_bg_points=False,
    add_cam=False,
)

''' Template of training options
'''
fine_train = dict(
    N_iters=20000,                # number of optimization steps
    N_rand=4096,                  # batch size (number of random rays per optimization step)
    lrate_feature=0.01,           # lr of density voxel grid
    lrate_featurenet=1e-3,
    lrate__time=1e-4,
    lrate__time_out = 1e-4,
    lrate_density = 0.01,
    lrate_timenet=1e-4,
    lrate_decoder=1e-3,           # lr of the mlp to preduct view-dependent color
    lrate_decay=20,               # lr decay by 0.1 after every lrate_decay*1000 steps
    ray_sampler='in_maskcache',        # ray sampling strategies
    weight_main=1.0,              # weight of photometric loss
    weight_dino = 1.0,
    weight_entropy_last=0.01,
    weight_rgbper=0.01,            # weight of per-point rgb loss
    tv_every=1,                   # count total variation loss every tv_every step
    tv_after=0,                   # count total variation loss from tv_from step
    tv_before=1e9,                   # count total variation before the given number of iterations
    tv_feature_before=10000,            # count total variation densely before the given number of iterations
    weight_tv_feature=0,
    # weight_tv_dino = 1e-5,
    weight_distortion = 0,
    weight_tv_density =0,
    pg_scale=[],
    skip_zero_grad_fields=['feature',"dino","density"],
)

''' Template of model and rendering options
'''

fine_model_and_render = dict(
    num_voxels=110**3,          # expected number of voxel
    num_voxels_base=110**3,      # to rescale delta distance
    voxel_dim=4,                 # feature voxel grid dim
    defor_depth=3,               # depth of the colors MLP (there are rgbnet_depth-1 intermediate features)
    net_width=64,             # width of the colors MLP
    alpha_init=1e-2,              # set the alpha values everywhere at the begin of training
    fast_color_thres=1e-4,           # threshold of alpha value to skip the fine stage sampled point
    stepsize=0.5,                 # sampling stepsize in volume rendering
    world_bound_scale=1.05,
    skips = [],
    maskout_near_cam_vox = True,
)



del deepcopy
