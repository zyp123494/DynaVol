_base_ = './default_hyper.py'

expname = 'cookie'
basedir = './logs'

data = dict(
    datadir='/data/d2nerf_dataset/vrig_cookie',
    dataset_type='hyper_dataset',
    white_bkgd=False,
    use_bg_points=False, 
    ratio = 0.5,
)

fine_train = dict(
    weight_entropy=0.012,
)

fine_model_and_render = dict(
    add_cam = False,
    max_instances = 15,
)

cluster = dict(
    rgb_thresh = 255,
    pos_thresh =3, #times voxel size
    traj_thresh = 100000,
    use_dbscan = False,
)
