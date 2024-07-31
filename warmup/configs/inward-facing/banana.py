_base_ = './default_hyper.py'

expname = 'banana'
basedir = './logs'

data = dict(
    datadir='/data/hypernerf_dataset/vrig-peel-banana',
    dataset_type='hyper_dataset',
    white_bkgd=False,
    use_bg_points=True,   
    ratio = 0.5,
)



fine_model_and_render = dict(
    add_cam = False,
    max_instances = 15,
) 

cluster = dict(
    rgb_thresh = 255,
    pos_thresh =5, #times voxel size
    traj_thresh = 5,
    use_dbscan = False,
)
