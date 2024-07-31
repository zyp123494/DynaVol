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
    weight_entropy=0.004,
    dino_start_training = 4000,
)

fine_model_and_render = dict(
    add_cam = False,
    max_instances = 15,
) 

cluster = dict(
    rgb_thresh = 255,
    pos_thresh =10, #times voxel size
    traj_thresh = 100000,
    use_dbscan = True,
)