_base_ = './default_hyper.py'

expname = 'slice-banana'
basedir = './logs'

data = dict(
    datadir='/data/hypernerf_dataset/interp_slice-banana/slice-banana',
    dataset_type='hyper_dataset',
    white_bkgd=False,
    use_bg_points=True,  
    ratio = 0.5,
)


fine_train = dict(
    weight_entropy=0.01,
    dino_start_training = 4000,
)

fine_model_and_render = dict(
    add_cam = False,
    max_instances = 15,
)

cluster = dict(
    rgb_thresh = 20,
    pos_thresh =3, #times voxel size
    traj_thresh = 3,
    use_dbscan = False,
)