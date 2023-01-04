
<img src="https://img.shields.io/badge/license-MIT-%23008FC7"> <img src="https://img.shields.io/badge/pytorch-1.8.2-%23EE4C2C"> <img src="https://img.shields.io/badge/python-3.8.11-%233776AB">

# Adversarial Erasing Framework via Triplet with Gated Pyramid Pooling Layer for Weakly Supervised Semantic Segmentation

This repository contains the official PyTorch implementation of the paper "[Adversarial Erasing Framework via Triplet with Gated Pyramid Pooling Layer for Weakly Supervised Semantic Segmentation](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136890323.pdf)" paper (ECCV 2022) by [Sung-Hoon Yoon](https://github.com/sunghoonYoon) and [Hyeokjun Kweon](https://github.com/sangrockEG).


## Introduction
We propose (1) the Gated Pyramid Pooling (GPP) layer to resolve the architectural limitation of classifier (or GAP) and (2) the Adversarial
Erasing Framework via Triplet (AEFT) to effectively prevent the over-expansion via triplet, while preserving the benefits of AE.
With image-level supervision only, we achieved new state-of-the-arts both on PASCAL VOC 2012 and MS-COCO.

<img src= "https://user-images.githubusercontent.com/42232407/179930811-59bc19f8-e0da-44d7-be83-64d9c489b319.png" width="700">


## Prerequisite
* Tested on Ubuntu 18.04, with Python 3.8, PyTorch 1.8.2, CUDA 11.4, both on both single and multi gpu.
* You can create conda environment with the provided yaml file.
```
conda env create -f wsss_new.yaml
```
* [The PASCAL VOC 2012 development kit](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/):
You need to specify place VOC2012 under ./data folder.
* ImageNet-pretrained weights for resnet38d are from [[resnet_38d.params]](https://drive.google.com/drive/folders/1Ak7eAs8Y8ujjv8TKIp-qCW20fgiIWTc2?usp=sharing).
You need to place the weights as ./pretrained/resnet_38d.params.
* Pretrained weight (PASCAL, seed: 56.2% mIoU) can be downloaded [here](https://drive.google.com/drive/folders/1Ak7eAs8Y8ujjv8TKIp-qCW20fgiIWTc2?usp=sharing).

## Usage
> With the following code, you can generate pseudo labels to train the segmentation network. 

> This code includes  [AffinityNet](https://github.com/jiwoon-ahn/psa)

### Training
* Please specify the name of your experiment.
* Training results are saved at ./experiment/[exp_name]
```
python train.py --name [exp_name] --model aeft_gpp
```
### Inference (ex.  alpha=9 (best))
To train AffinityNet, you must extract the followings: 1.seeds, 2.CRF(low):int , 3.CRF(high):int

Option 1: if you need to train AffinityNet, For faster convergence use the res38_cls.pth.
The best setting of AEFT is (low: 2 high: 21 dup:19)
```
python infer.py --name [exp_name] --model aeft_gpp --load_epo [epoch_to_load] --dict --crf --alphas [crf value to extract, e.g. CRF(low), CRF(high)] --infer_list voc12/train_aug.txt
```
Option 2: if you only need to get CRF
```
python infer.py --name [exp_name] --model aeft_gpp --load_epo [epoch_to_load] --dict --crf --alphas [crf value to extract, e.g. 6,7,8] --infer_list voc12/train.txt
```

### Train AffinityNet
The best setting of AEFT is (low: 2 high: 21 dup:19)
```
python train_aff.py --name [exp_name: Use the same name above] --low CRF(low):int --high CRF(high):int --dup [CRF value for affinity:int]
```

### Infer AffinityNet (Pseudo pixel-level labels)
```
python infer_aff.py --name [exp_name: Use the same name above] --low CRF(low):int --high CRF(high):int --dup [CRF value for affinity:int] --train_list voc12/train_aug.txt
```
### Evaluation for CAM
```
python evaluation.py --name [exp_name] --task cam --dict_dir dict
```
### Evaluation for CRF result (ex.  alpha=9 (best))
```
python evaluation.py --name [exp_name] --task crf --dict_dir crf/[xx]
```
## Citation
If our code be useful for you, please consider citing our ECCV paper using the following BibTeX entry.
```
@inproceedings{yoon2022adversarial,
  title={Adversarial Erasing Framework via Triplet with Gated Pyramid Pooling Layer for Weakly Supervised Semantic Segmentation},
  author={Yoon, Sung-Hoon and Kweon, Hyeokjun and Cho, Jegyeong and Kim, Shinjeong and Yoon, Kuk-Jin},
  booktitle={European Conference on Computer Vision},
  pages={326--344},
  year={2022},
  organization={Springer}
}
```

we heavily borrow the work from [AffinityNet](https://github.com/jiwoon-ahn/psa) repository. Thanks for the excellent codes!
```
## Reference
[1] J. Ahn and S. Kwak. Learning pixel-level semantic affinity with image-level supervision for weakly supervised semantic segmentation. In Proc. IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2018.
