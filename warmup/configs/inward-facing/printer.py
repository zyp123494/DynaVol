_base_ = './default_hyper.py'

expname = 'printer'
basedir = './logs'

data = dict(
    datadir='/data/hypernerf_dataset/vrig_3dprinter/vrig-3dprinter',
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
    rgb_thresh = 30,
    pos_thresh =3, #times voxel size
    traj_thresh = 3,
    use_dbscan = False,
)
