_base_ = './default_pipeline.py'

expname = 'movi_3objsfall'
basedir = 'exp/'

data = dict(
    datadir='/data/3objsfall/dynamic',
    dataset_type='blender',
    white_bkgd=True,
)


fine_train = dict(
    N_iters=35000,
    static_model_path = "./../warmup/exp/3objs/fine_last_n.tar",
    lrate_density=0.08,
    lrate_decoder=8e-4,
    lrate_slot_attention=8e-4,
    lrate__time=6e-4,
    lrate__time_out=6e-4,
)

fine_model_and_render = dict(
    max_instances=10,
    z_dim=64,
    timenet_layers=4,
    timenet_hidden=128,
    skips=[2],
    timesteps=60,
)
