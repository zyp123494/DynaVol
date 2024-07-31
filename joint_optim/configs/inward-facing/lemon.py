_base_ = './default_hyper.py'

expname = 'lemon'
basedir = './logs'

data = dict(
    datadir='/data/hypernerf_dataset/interp_cut-lemon/cut-lemon1',
    dataset_type='hyper_dataset',
    white_bkgd=False,
    use_bg_points=True,   
    ratio = 0.5,
)


fine_train = dict(
    warmup_model_path = "./../warmup/logs/lemon"
)

fine_model_and_render = dict(
    add_cam = False,
) 
