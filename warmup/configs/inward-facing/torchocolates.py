_base_ = './default_hyper.py'

expname = 'torchocolates'
basedir = './logs'

data = dict(
    datadir='/data/ypzhao/hypernerf_dataset/interp_torchocolate/torchocolate',
    dataset_type='hyper_dataset',
    white_bkgd=False,
    use_bg_points=True,   
    ratio = 0.5,
)


fine_train = dict(
    weight_entropy=0.012,
    dino_start_training = 8000,
)


fine_model_and_render = dict(
    add_cam = False,
    max_instances = 15,
)

cluster = dict(
    rgb_thresh = 30,
    pos_thresh =5, #times voxel size
    traj_thresh = 5,
    use_dbscan = False,
)