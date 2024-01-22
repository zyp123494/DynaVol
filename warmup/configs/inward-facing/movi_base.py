_base_ = './default_pipeline.py'

expname = '3objs'
basedir = 'exp/'

data = dict(
    datadir='/data/3objsfall/dynamic',
    dataset_type='blender',
    white_bkgd=True,
    half_res = False,
)

data_static = dict(
    datadir='/data/3objsfall/static',
    dataset_type='blender',
    white_bkgd=True,
    num_train = 5,
    half_res = False,
)


fine_train = dict(
    lrate_density=0.1,
    lrate_decoder=1e-3,
    lrate__time = 1e-3,
    lrate__time_out=1e-3,
    lrate__time_inverse = 1e-3,
    lrate__time_out_inverse=1e-3,
    weight_static=1.0,
    N_iters=50000,
    pg_scale = [1000,2000,3000],
    pervoxel_lr = False,
)

fine_model_and_render = dict(
    max_instances=1,
    maskout_near_cam_vox  = True,
    timenet_layers=4,
    timenet_hidden=128, # 64
    skips=[2],
    z_dim=128,                                       # dimension of hidden dimension in nerf decoder
    n_layers=4, 
    timesteps=60, 
)
