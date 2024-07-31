
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



 code is coming soon!
