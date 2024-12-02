_base_ = './default.py'

expname = 'small/8obj'
basedir = './logs'

data = dict(
    datadir='/data/8obj/dynamic',
    dataset_type='dnerf',
    white_bkgd=True,
    half_res = True,
)

fine_train = dict(
    N_iters=20000, 
    lrate_seg_mask = 1e-3,
    lrate_latent_code = 0.1,
    lrate_feature = 0.01,
    warmup_model_path = "./../warmup/logs/small/8obj"
   
)

fine_model_and_render = dict(
    max_instances=10,
    n_freq=5,      
    n_freq_view=4,                                 
    n_freq_time=5,
    n_freq_t=5,
    n_freq_feat=2,
    add_cam = False,
    alpha_init=5e-3,
    net_width=128,  
    z_dim = 64,
) 

