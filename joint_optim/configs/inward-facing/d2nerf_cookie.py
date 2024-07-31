_base_ = './default_hyper.py'

expname = 'd2nerf_cookie'
basedir = './logs'

data = dict(
    datadir='/data/d2nerf_dataset/vrig_cookie',
    dataset_type='hyper_dataset',
    white_bkgd=False,
    use_bg_points=False,  
    ratio = 0.5,
)


fine_train = dict(
    warmup_model_path = "./../warmup/logs/d2nerf_cookie"
)

fine_model_and_render = dict(
    add_cam = False,
) 
