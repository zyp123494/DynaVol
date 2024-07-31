_base_ = './default_hyper.py'

expname = 'cross_hands'
basedir = './logs'

data = dict(
    datadir='/data/hypernerf_dataset/misc_cross-hands/cross-hands1',
    dataset_type='hyper_dataset',
    white_bkgd=False,
    use_bg_points=True, 
    ratio = 0.5,
)


fine_train = dict(
    warmup_model_path = "./../warmup/logs/cross_hands"
)

fine_model_and_render = dict(
    add_cam = False,
) 
