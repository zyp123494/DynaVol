_base_ = './default_hyper.py'

expname = 'balloon'
basedir = './logs'

data = dict(
    datadir='/data/d2nerf_dataset/vrig_balloon',#vrig-chicken
    dataset_type='hyper_dataset',
    white_bkgd=False,
    use_bg_points=False,  
    ratio = 0.5,
)


fine_train = dict(
    warmup_model_path = "./../warmup/logs/balloon"
)

fine_model_and_render = dict(
    add_cam = False,
) 
