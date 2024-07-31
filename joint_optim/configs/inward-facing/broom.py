_base_ = './default_hyper.py'

expname = 'broom'
basedir = './logs'

data = dict(
    datadir='/data/hypernerf_dataset/vrig_broom/broom2',
    dataset_type='hyper_dataset',
    white_bkgd=False,
    use_bg_points=True,
    ratio = 0.5,
)



fine_train = dict(
    warmup_model_path = "./../warmup/logs/broom"
)

fine_model_and_render = dict(
    add_cam = False,
) 

