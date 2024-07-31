_base_ = './default_hyper.py'

expname = 'sieve'
basedir = './logs'

data = dict(
    datadir='/data/NeRF-DS-dataset/sieve_novel_view',
    dataset_type='hyper_dataset',
    white_bkgd=False,
    use_bg_points=False,  
    ratio = 1.0,
)


fine_train = dict(
    dino_start_training = 4000,
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
