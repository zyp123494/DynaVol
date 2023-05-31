# DynaVol dataset

This is the implementation of producing the synthetic dataset used in DynaVol. The code uses [Kubric](https://github.com/google-research/kubric) to generate the dataset.

## Preparation

### Installation
Install [Docker](https://www.docker.com/) follow the instruction [here](https://docs.docker.com/engine/install/).

Pull Kubric image follow the instruction [here](https://github.com/google-research/kubric).

## Build DynaVol dataset

### Build a single scene
```bash
$ docker run \
    --rm \
    --interactive \
    --user $(id -u):$(id -g) \
    --volume "$(pwd):/kubric" \
    kubricdockerhub/kubruntu \
    /usr/bin/python3 DynaVol_syn_shape.py 

```

### Build multiple scenes
```bash
$ bash launch.sh

```

The scripts above generate several files and output for Kubric. These files include:

```
    [output]
    ├── static 
    │   └── ├── [train|val|test]
    │       └── transforms_[train|val|test].json
    ├── dynamic 
    │   └── ├── [train|val|test]
    │       └── transforms_[train|val|test].json  
    └── dynamic_4views 
        └── ├── [view0|view1|view2|view3]
            └── transforms_[train].json 
  
```

Post-processing of data, including background removal and merging of 4 fixed views:
```bash
$ python post_process.py

```

## Reproduction: datasets used in DynaVol
[DynaVol_syn_shape.py](DynaVol_syn_shape.py) is used for synthetic objects, and [DynaVol_real_shape.py](DynaVol_real_shape.py) for real-world objects(e.g. 3ObjRealCmpx and 3ObjRealSimp). 

Add "--objects_set kubasic" for more complex shapes (e.g. 3Fall+3Still),

"--num_stc_objects N" to add N static objects,

"--material" to modify the material of objects (e.g. 3ObjMetal), 

"--xy_vel" for more complex motion patterns (e.g. 3ObjRand),

"--real_texture"(only for real-world objects) for real-world texture(e.g. 3ObjRealCmpx).

## Citation

  

If you find our work helps, please cite our paper.

  

```bibtex

@article{gao2023unsupervised,
  title={Unsupervised Object-Centric Voxelization for Dynamic Scene Understanding},
  author={Siyu Gao and Yanpeng Zhao and Yunbo Wang and Xiaokang Yang},
  journal={arXiv preprint arXiv:2305.00393},
  year={2023}
}


```


## Acknowledgements
This codebase is based on [Kubric](https://github.com/google-research/kubric).

