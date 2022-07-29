from __future__ import division
from __future__ import print_function

import os, time
import os.path as osp
import argparse
import glob
import random
import pdb
import importlib
import logging

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import tools.utils as utils

################################################################################
# Infer CAM image, CAM dict and CRF dict from given experiments/checkpoints.
# All of the result files will be saved under given experiment folder.
#
# If you want to get CAM_dict files...
# python infer.py --name [exp_name] --load_epo [epoch] --dict
#
# Or if you want to get CRF dict files with certain alpha (let, a1 and a2)...
# python infer.py --name [exp_name] --load_epo [epoch] --crf --alphas a1 a2
#
# python infer.py --name selfsup5 --model models.model_selfsup5 --dict --crf --load_epo 3 --infer_list voc12/train.txt --alphas
# Of course you can do them at the same time.
# To get CAM image, simply add --vis.
################################################################################

if __name__ == '__main__':

    categories = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 
                  'bus', 'car', 'cat', 'chair', 'cow', 
                  'diningtable', 'dog', 'horse', 'motorbike', 'person', 
                  'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

    parser = argparse.ArgumentParser()

    # Dataset
    parser.add_argument("--infer_list", default="voc12/train.txt", type=str)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--resize", default=[256, 512], nargs='+', type=float)
    parser.add_argument("--crop", default=[320, 320], nargs='+', type=int)
    parser.add_argument("--cj", default=[0.4, 0.4, 0.4, 0.1], nargs='+', type=float)

    # Learning rate
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--wt_dec", default=5e-4, type=float)  
    parser.add_argument("--max_epoches", default=8, type=int)

    # Experiments
    parser.add_argument("--model", default='aeft_gpp', type=str)
    parser.add_argument("--name", required=True, type=str)
    parser.add_argument("--gpu", default=-1, type=int)
    parser.add_argument("--load_epo", required=True, type=int)


    # Hyper-parameters
    parser.add_argument("--D", default=256, type=int)
    parser.add_argument("--M", default=0.996, type=float)
    parser.add_argument("--TH", default=0.3, type=float)
    parser.add_argument("--T", default=2.0, type=float)
    parser.add_argument("--W", default=[1.0, 1.0, 1.0], nargs='+', type=float)
    parser.add_argument("--CRF", default=4, type=int)
    parser.add_argument("--MEM", default=4, type=int)

    # Output
    parser.add_argument("--vis", action='store_true')
    parser.add_argument("--dict", action='store_true')
    parser.add_argument("--crf", action='store_true')
    parser.add_argument("--print_freq", default=100, type=int)
    parser.add_argument("--out_num", default=100, type=int)
    parser.add_argument("--alphas", default=[5,7,9], nargs='+', type=int)

    args = parser.parse_args()
    args.max_step = 1

    logger = logging.getLogger(__name__)

    print('Infer experiment ' + args.name + '!')
    exp_path, ckpt_path, train_path, val_path, infer_path, dict_path, crf_path = utils.make_path(args)

    infer_dataset = utils.build_dataset_moco(args,phase='val', path=args.infer_list)
    infer_data_loader = DataLoader(infer_dataset, shuffle=False, pin_memory=True)

    print('Infer dataset is loaded from ' + args.infer_list)
    
    model = getattr(importlib.import_module('models.model_'+args.model), 'model_WSSS')(args, logger)
    model.load_model(args.load_epo, ckpt_path)
    model.set_phase('eval')
    model.infer_init()

    print('#'*111)
    print(('#'*46)+' Start infer loop '+('#'*47))
    print('#'*111)

    for iter, pack in enumerate(tqdm(infer_data_loader)):
        model.unpack(pack)
        model.infer_multi(42, train_path, dict_path, crf_path, vis=args.vis, dict=args.dict, crf=args.crf)
