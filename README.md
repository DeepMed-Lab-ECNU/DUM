# Debiasing Medical Knowledge for Prompting Universal Model in CT Image Segmentation (TMI)
by Boxiang Yun, Shitian Zhao, Qingli Li, Alex Kot, Yan Wang*
 
## Introduction
A PyTorch implementation of **DUM**, a causal debiasing framework that enhances generalization in universal medical image segmentation by mitigating knowledge bias from text prompts. DUM leverages both organ-level semantic priors and instance-level visual context to improve robustness across diverse clinical scenarios. Based on the paper: [*Debiasing Medical Knowledge for Prompting Universal Model in CT Image Segmentation*](https://ieeexplore.ieee.org/abstract/document/11080474/).

## Requirements
This repository is based on PyTorch 2.2.2, CUDA 11.8, and Python 3.９.７. All experiments in our paper were conducted on NVIDIA GeForce RTX 3090 GPU with an identical experimental setting.

## Data Preparation

- 01 [Multi-Atlas Labeling Beyond the Cranial Vault - Workshop and Challenge (BTCV)](https://www.synapse.org/#!Synapse:syn3193805/wiki/217789)
- 02 [Pancreas-CT TCIA](https://wiki.cancerimagingarchive.net/display/Public/Pancreas-CT)  ## The label we used for Dataset 01 and 02 is in [here](https://zenodo.org/records/1169361).
- 03 [Combined Healthy Abdominal Organ Segmentation (CHAOS)](https://chaos.grand-challenge.org/Combined_Healthy_Abdominal_Organ_Segmentation/)
- 04 [Liver Tumor Segmentation Challenge (LiTS)](https://competitions.codalab.org/competitions/17094#learn_the_details)
- 05 [Kidney and Kidney Tumor Segmentation (KiTS)](https://kits21.kits-challenge.org/participate#download-block)
- 06 [Liver segmentation (3D-IRCADb)](https://www.ircad.fr/research/data-sets/liver-segmentation-3d-ircadb-01/)
- 07 [WORD: A large scale dataset, benchmark and clinical applicable study for abdominal organ segmentation from CT image](https://github.com/HiLab-git/WORD)
- 08 [AbdomenCT-1K](https://github.com/JunMa11/AbdomenCT-1K)
- 09 [Multi-Modality Abdominal Multi-Organ Segmentation Challenge (AMOS)](https://amos22.grand-challenge.org)
- 10 [Decathlon (Liver, Lung, Pancreas, HepaticVessel, Spleen, Colon](https://drive.google.com/drive/folders/1HqEgzS8BV2c7xYNrZdEAnrHk7osJJ--2)
- 11 [CT volumes with multiple organ segmentations (CT-ORG)](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=61080890)
- 12 [AbdomenCT 12organ](https://zenodo.org/records/7860267)

The post_label can be downloaded via [link](https://portland-my.sharepoint.com/:u:/g/personal/jliu288-c_my_cityu_edu_hk/EX04Ilv4zh1Lm_HB0wnpaykB4Slef043RVWhX3lN05gylw?e=qG0DOS).
(see https://github.com/ljwztc/CLIP-Driven-Universal-Model/blob/main/README.md)

- Directory Structure
Datasets/
└── 01_Multi-Atlas_Labeling
    ├── img/
        └──img0001.nii.gz
        ├── ... 
    ├── label/
    └──post_label/
    02_TCIA_Pancreas-CT
    ...
    15_3DIRCADb
    16_TotalsegmentatorV2
    17_AbdomenAtlas_1_0_Mini


## Usage
### Train:
To train a model,
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -W ignore -m torch.distributed.launch --nproc_per_node=4 --master_port=12340 \
train_main_dum.py --dist True --num_workers 8 --num_samples 2 --uniform_sample \
  --log_name train_base_4gpu_1002 --data_root_path ../Datasets/ \
  --num_context 4 --alpha 0.5 --batch_size 1 --max_epoch 1000
```
- Checkpoints and TensorBoard saved in ./out/train_base_4gpu_1002/

To test a model,
```
CUDA_VISIBLE_DEVICES=0 python -W ignore test.py \
--resume1 ./out/train_base_4gpu_1002/epoch_x.pth \
--data_root_path ../Datasets/ \
--dataset_list PAOT_10_inner --num_workers 8
```

## Citation
If you find the project useful, please consider citing:

```bibtex
@article{yun2025debiasing,
  title={Debiasing Medical Knowledge for Prompting Universal Model in CT Image Segmentation},
  author={Yun, Boxiang and Zhao, Shitian and Li, Qingli and Kot, Alex and Wang, Yan},
  journal={IEEE Transactions on Medical Imaging},
  year={2025},
  publisher={IEEE}
}
```

## TODO List
- Release pretrained model weights and provide a simple inference pipeline.


