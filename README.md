# DynaVol

[Project page](https://sites.google.com/view/dynavol/) | [arXiv](https://arxiv.org/abs/2305.00393)

Code repository for this paper:  
**Unsupervised Object-Centric Voxelization for Dynamic Scene Understanding.**  
Siyu Gao, Yanpeng Zhao, [Yunbo Wang](https://wyb15.github.io/)<sup>†</sup>, [Xiaokang Yang](https://scholar.google.com/citations?user=yDEavdMAAAAJ&hl=zh-CN)

<img  src="/figure/dynavol.PNG"  alt="dynavol"  style="zoom:67%;"  />

## Preparation

### Installation
```
git clone https://github.com/zyp123494/DynaVol.git
cd DynaVol
pip install -r requirements.txt
```
[Pytorch](https://pytorch.org/), [torch_scatter](https://github.com/rusty1s/pytorch_scatter) and [DGL](https://www.dgl.ai/pages/start.html)(CPU version is sufficient) installation is machine dependent, please install the correct version for your machine.

### DynaVol dataset

DynaVol dataset is available at [GoogleDrive](https://drive.google.com/drive/folders/1rADezOEG3WwMidwQkWQBdGTGiW2Y1Q2K?usp=sharing) or [OneDrive](https://sjtueducn-my.sharepoint.com/:f:/g/personal/zhao-yan-peng_sjtu_edu_cn/ErPjQahfAtFGsj74okb-dKQBcgoVVpdYRr_vG_oC9rXFdQ?e=xkwFdd). For each scene, we release the static data, dynamic data, and dynamic data which is collected by 4 fixed views(can be used to train [DeVRF](https://github.com/showlab/DeVRF/tree/main)). Please refer to the following data structure for an overview of DynaVol dataset.

```
    [3ObjFall|6ObjFall|...]
    ├── static 
    │   └── ├── [train|val|test]
    │       └── transforms_[train|val|test].json
    ├── dynamic 
    │   └── ├── [train|val|test]
    │       └── transforms_[train|val|test].json  
    └── dynamic_4views 
        └── ├── [train]
            └── transforms_[train].json 
  
```

For more details of the DynaVol dataset and the code to generate it, please refer to [DynaVol_dataset](DynaVol_dataset).

## Experiment

### Training
Stage1: Warmup stage 
```bash
$ cd warmup
$ bash run_full.sh

```

Stage2: Dynamic grounding stage, modify the static_model_path in [config](dynamic_grounding/configs/inward-facing/movi_pipeline.py) to the checkpoint of the first stage(e.g. "/DynaVol/warmup/exp/3ObjFall/fine_last_n.tar").
```bash
$ cd ../dynamic_grounding
$ bash run.sh

```

Code for real-world data is coming soon!

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
This codebase is based on [DirectVoxGO](https://github.com/sunset1995/DirectVoxGO) and [DeVRF](https://github.com/showlab/DeVRF/tree/main).

