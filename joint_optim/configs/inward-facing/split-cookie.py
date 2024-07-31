_base_ = './default_hyper.py'

expname = 'split-cookie'
basedir = './logs'

data = dict(
    datadir='/data/hypernerf_dataset/misc_split-cookie/split-cookie',#vrig-chicken
    dataset_type='hyper_dataset',
    white_bkgd=False,
    use_bg_points=True,
    ratio = 0.5,
)


fine_train = dict(
    warmup_model_path = "./../warmup/logs/split-cookie"
)

fine_model_and_render = dict(
    add_cam = False,
) 
