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
    weight_entropy=0.006,
    dino_start_training = 8000,
)


fine_model_and_render = dict(
    add_cam = False,
    max_instances = 15,
)

cluster = dict(
    rgb_thresh = 255,
    pos_thresh =3, #times voxel size
    traj_thresh = 10000000,
    use_dbscan = False,
)
