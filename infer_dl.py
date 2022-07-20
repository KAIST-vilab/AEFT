from __future__ import division
from __future__ import print_function

import os, time
import os.path as osp
import argparse
import glob
import random
import pdb
import importlib

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
##
import tools.utils as utils

################################################################################
# Infer CAM image, CAM dict and CRF dict from given experiments/checkpoints.
# All of the result files will be saved under given experiment folder.
#
# If you want to get CAM_dict files...
# python infer_dl.py --name [exp_name] --load_epo [epoch] --dict
#
# Or if you want to get CRF dict files with certain alpha (let, a1 and a2)...
# python infer_dl.py --name [exp_name] --crf --alphas a1 a2 --vis
#
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
    parser.add_argument("--infer_list", default="voc12/test.txt", type=str)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--batch_size", default=8, type=int)

    # Learning rate
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--wt_dec", default=5e-4, type=float)  
    parser.add_argument("--max_epoches", default=8, type=int)

    # Experiments
    parser.add_argument("--model", default='models.model_dl', type=str)
    parser.add_argument("--name", required=True, type=str)
    parser.add_argument("--gpu", default=-1, type=int)
    parser.add_argument("--load_epo", default=0, type=int)

    # Model related hyper-parameters
    parser.add_argument("--sharing_position", default=0, type=int)
    parser.add_argument("--loss_weights", default=[1,1,1,1], nargs='+', type=int)
    parser.add_argument("--prob", default=0.50, type=float)
    parser.add_argument("--cl_loop", default=1, type=int)   

    # Output
    parser.add_argument("--vis", action='store_true')
    parser.add_argument("--dict", action='store_true')
    parser.add_argument("--crf", action='store_true')
    parser.add_argument("--print_freq", default=100, type=int)
    parser.add_argument("--out_num", default=100, type=int)
    parser.add_argument("--alphas", default=[10], nargs='+', type=int)
    parser.add_argument("--is_test",default=False,type=bool)

    args = parser.parse_args()
    args.max_step = 1

    if args.is_test:
        phase = "test"
    else:
        phase = "val"

    print("phase: ",phase)

    print('Infer experiment ' + args.name + '!')
    exp_path, ckpt_path, train_path, val_path, infer_path, dict_path, crf_path = utils.make_path(args)

    infer_dataset = utils.build_dataset_dl(phase=phase, path=args.infer_list, gt_path='./data/VOC2012/SegmentationClass')
    infer_data_loader = DataLoader(infer_dataset, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    print('Infer dataset is loaded from ' + args.infer_list)
    
    model = getattr(importlib.import_module(args.model), 'model_WSSS')(args)
    model.load_model(args.load_epo, ckpt_path)
    model.set_phase('eval')

    if phase == 'test':
        test_path = osp.join(exp_path, 'test')
        if not os.path.exists(test_path):
            os.makedirs(test_path)
            print("Making ./test dir")

    print('#'*111)
    print(('#'*46)+' Start infer loop '+('#'*47))
    print('#'*111)
    model.infer_multi_init()
    for iter, pack in enumerate(tqdm(infer_data_loader)):

        model.unpack(pack,test_flag=args.is_test)

        if args.dict:
            if phase =='val':
                model.infer_multi(42, infer_path, dict_path, crf_path, vis=False, dict=True, crf=False) #Val
            elif phase =='test':
                model.infer_multi(42, test_path, dict_path, crf_path, vis=False, dict=True, crf=False)  # test
            # model.visualize_cam(42,42,val_path)
        if args.crf:
            if phase == 'val':
                model.dict2crf(42, infer_path, dict_path, crf_path)
            elif phase == 'test':
                model.dict2crf(42, test_path, dict_path, crf_path)