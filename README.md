
# DynaVol-S

[Project page]( https://zyp123494.github.io/DynaVol-S.github.io/) | [DynaVol](https://arxiv.org/abs/2305.00393) | [DynaVol-S](https://arxiv.org/abs/2407.20908)

Code repository for this paper:  
**DynaVol-S: Dynamic Scene Understanding through Object-Centric Voxelization and Neural Rendering**  
Yanpeng Zhao, Yiwei Hao, Siyu Gao, [Yunbo Wang](https://wyb15.github.io/)<sup>†</sup>, [Xiaokang Yang](https://scholar.google.com/citations?user=yDEavdMAAAAJ&hl=zh-CN)

<img  src="/figure/dynavol-s.png"  alt="dynavol-s"  style="zoom:67%;"  />

## News🎉
-[2024.7.31] DynaVol-S has been integrated into this repo, which significantly improve the model performance in real-world scenes by incorporating DINOv2 features. For the original version aligned with ICLR24 paper, please check the [dynavol](https://github.com/zyp123494/DynaVol/tree/dynavol) branch.

-[2024.1.17] DynaVol got accepted by ICLR2024!

## Preparation

### Installation
```
git clone -b main https://github.com/zyp123494/DynaVol.git
cd DynaVol
conda create -n dynavol python=3.8
conda activate dynavol

#install pytorch
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121

#install Featup
git clone https://github.com/mhamilton723/FeatUp.git
cd FeatUp
pip install -e .

#install requirements
pip install -r requirements.txt
```
Install the correct version of [torch_scatter](https://github.com/rusty1s/pytorch_scatter), for torch=2.1.0+cuda12.1, you can simply download the corresponding version from [here](https://data.pyg.org/whl/) and simply run:
```
pip install torch_scatter-2.1.2+pt21cu121-cp38-cp38-linux_x86_64.whl
```

## Dataset
In our paper, we use:

- synthetic dataset from [DynaVol](https://github.com/zyp123494/DynaVol/tree/dynavol).
- real-world dataset from [Hyper-NeRF](https://hypernerf.github.io/), [NeRF-DS](https://jokeryan.github.io/projects/nerf-ds/), and [D2NeRF](https://d2nerf.github.io/).


## Experiment
Extract DINOv2 features with FeatUp, modify the "img_dir" in extract_dinov2.py then run:
```
python extract_dinov2.py
```

### Training
Stage 1: Warmup

Cofig files are under the config directory

```bash
$ cd warmup
#Synthetic dataset
$ bash run.sh

#Real-world dataset
$ bash run_hyper.sh
```

Stage 2: CRF postprocess

Modify the "base_path" and "data_dir" in crf_postprocess.py, then run:
```
python crf_postprocess.py
```

Stage 3: Joint-optimization
```bash
$ cd ../joint_optim
#Synthetic dataset
$ bash run.sh

#Real-world dataset
$ bash run_hyper.sh
```


## Citation

  

If you find our work helps, please cite our paper.

  

```bibtex


@inproceedings{
zhao2024dynavol,
title={DynaVol: Unsupervised Learning for Dynamic Scenes through Object-Centric Voxelization},
author={Yanpeng Zhao and Siyu Gao and Yunbo Wang and Xiaokang Yang},
booktitle={The Twelfth International Conference on Learning Representations},
year={2024},
url={https://openreview.net/forum?id=koYsgfEwCQ}
}

@misc{zhao2024dynamicsceneunderstandingobjectcentric,
      title={Dynamic Scene Understanding through Object-Centric Voxelization and Neural Rendering}, 
      author={Yanpeng Zhao and Yiwei Hao and Siyu Gao and Yunbo Wang and Xiaokang Yang},
      year={2024},
      eprint={2407.20908},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2407.20908}, 
}


```
