
# Adversarial Erasing Framework via Triplet with Gated Pyramid Pooling Layer for Weakly Supervised Semantic Segmentation

This repository contains the official PyTorch implementation of the paper "[Adversarial Erasing Framework via Triplet with Gated Pyramid Pooling Layer for Weakly Supervised Semantic Segmentation](tbd)" paper (ECCV 2022) by [Sung-Hoon Yoon](https://github.com/sunghoonYoon) and [Hyeokjun Kweon](https://github.com/sangrockEG).


## Introduction
We propose (1) the Gated Pyramid Pooling (GPP) layer to resolve the architectural limitation of classifier (or GAP) and (2) the Adversarial
Erasing Framework via Triplet (AEFT) to effectively prevent the over-expansion via triplet, while preserving the benefits of AE.
With image-level supervision only, we achieved new state-of-the-arts both on PASCAL VOC 2012 and MS-COCO.

<img src= "https://user-images.githubusercontent.com/42232407/179930811-59bc19f8-e0da-44d7-be83-64d9c489b319.png" width="700">


## Prerequisite
* Tested on Ubuntu 18.04, with Python 3.6, PyTorch 1.8.2, CUDA 11.4, both on both single and multi gpu.
* You can create conda environment with the provided yaml file.
```
conda env create -f wsss_new.yaml
```
* [The PASCAL VOC 2012 development kit](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/):
You need to specify place VOC2012 under ./data folder.
* ImageNet-pretrained weights for resnet38d are from [[resnet_38d.params]](https://github.com/itijyou/ademxapp).
You need to place the weights as ./pretrained/resnet_38d.params.
* PASCAL-pretrained weights will be uploaded soon.. (TBD)
You need to place the weights as ./pretrained/aeft_gpp.pth.
## Usage
### Training
* Please specify the name of your experiment.
* Training results are saved at ./experiment/[exp_name]
```
python train.py --name [exp_name] --model aeft_gpp
```
### Inference
```
python infer.py --name [exp_name] --model aeft_gpp --load_epo [epoch_to_load] --vis --dict --crf --alphas [crf value to extract, e.g. 6,7,8]
```
### Evaluation for CAM result
```
python evaluation.py --name [exp_name] --task cam --dict_dir dict
```
### Evaluation for CRF result (ex. alpha=6)
```
python evaluation.py --name [exp_name] --task crf --dict_dir crf/[xx]
```
## Citation
If our code be useful for you, please consider citing our ECCV paper using the following BibTeX entry.
```
TBD
```

we heavily borrow the work from [AffinityNet](https://github.com/jiwoon-ahn/psa) repository. Thanks for the excellent codes!
```
## Reference
[1] J. Ahn and S. Kwak. Learning pixel-level semantic affinity with image-level supervision for weakly supervised semantic segmentation. In Proc. IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2018.
