from __future__ import division
from __future__ import print_function

# Base
import os
import os.path as osp
from tqdm import tqdm
import random
import importlib
import argparse
import logging
import pdb

from matplotlib import pyplot as plt

# DL
import numpy as np
import torch
from torch.utils.data import DataLoader

# Custom
import tools.imutils as imutils
import tools.utils as utils
import tools.pyutils as pyutils
from evaluation import eval_in_script

if __name__ == '__main__':

    categories = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
                  'bus', 'car', 'cat', 'chair', 'cow',
                  'diningtable', 'dog', 'horse', 'motorbike', 'person',
                  'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

    parser = argparse.ArgumentParser()

    # Dataset
    parser.add_argument("--train_list", default="voc12/train_aug.txt", type=str)
    parser.add_argument("--val_list", default="voc12/train.txt", type=str)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--batch_size", default=32, type=int)

    # Augmentation
    parser.add_argument("--resize", default=[256, 512], nargs='+', type=float)
    parser.add_argument("--crop", default=[320, 320], nargs='+', type=int)
    parser.add_argument("--cj", default=[0.4, 0.4, 0.4, 0.1], nargs='+', type=float)


    # Hyper-parameters
    parser.add_argument("--M", default=0.5, type=float)
    parser.add_argument("--TH", default=[0.6,0.2], nargs='+',type=float)
    parser.add_argument("--T", default=18, type= int)
    parser.add_argument("--W", default=[1.0, 0.2, 0.15], nargs='+', type=float)
    parser.add_argument("--CRF", default=4, type=int)

    # Learning rate
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--wt_dec", default=5e-4, type=float)
    parser.add_argument("--max_epochs", default=40, type=int)

    # Experiments
    parser.add_argument("--model", default='aeft_gpp', type=str)
    parser.add_argument("--name", required=True, type=str)
    parser.add_argument("--gpu", default=-1, type=int)
    parser.add_argument("--seed", default=5137, type=int)

    # Output
    parser.add_argument("--vis", action='store_true')
    parser.add_argument("--dict", action='store_false')
    parser.add_argument("--crf", action='store_true')
    parser.add_argument("--print_freq", default=100, type=int)
    parser.add_argument("--out_num", default=100, type=int)
    parser.add_argument("--alphas", default=[6, 10, 24], nargs='+', type=int)

    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    exp_path, ckpt_path, train_path, val_path, infer_path, dict_path, crf_path, log_path = utils.make_path_with_log(args)
    
    # Logger
    if osp.isfile(log_path):
        os.remove(log_path)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    logger.addHandler(console_handler)
    file_handler = logging.FileHandler(log_path)
    logger.addHandler(file_handler)

    logger.info('-'*52 + ' SETUP ' + '-'*52)
    for arg in vars(args):
        logger.info(arg + ' : ' + str(getattr(args, arg)))
    logger.info('-'*111)
    
    logger.info('Start experiment ' + args.name + '!')
    
    train_dataset = utils.build_dataset_moco(args, phase='train', path=args.train_list)
    val_dataset = utils.build_dataset_moco(args, phase='val', path=args.val_list)
    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, drop_last=True, num_workers=4)
    val_data_loader = DataLoader(val_dataset, shuffle=False, pin_memory=True)

    logger.info('Train dataset is loaded from ' + args.train_list)
    logger.info('Validation dataset is loaded from ' + args.val_list)

    train_num_img = len(train_dataset)
    train_num_batch = len(train_data_loader)

    # max_step = (train_num_img // 8 * args.T)+(train_num_img // 32 * (args.max_epochs-args.T))
    max_step = (train_num_img // 8 * args.max_epochs)
    args.max_step = max_step

    model = getattr(importlib.import_module('models.model_'+args.model), 'model_WSSS')(args, logger)
    model.train_setup()

    logger.info('-' * 111)
    logger.info(('-' * 49) + ' Start Train ' + ('-' * 49))

    miou_list = []
    max_miou = 0
    for epo in range(args.max_epochs):
        epo_str = str(epo).zfill(3)

        # Train
        logger.info('-' * 111)
        logger.info('Epoch ' + epo_str + ' train')
        model.set_phase('train')

        print("lr_mult:",model.opt_main.lr_mult)

        if epo< args.T:
            train_data_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, pin_memory=True, drop_last=True, num_workers=4)
            train_num_batch = len(train_data_loader)
        else:
            train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, drop_last=True, num_workers=4)
            train_num_batch = len(train_data_loader)

        for iter, pack in enumerate(tqdm(train_data_loader)):
            model.unpack(pack)
            model.update(epo)
            if iter % args.print_freq == 0 and iter != 0:
                model.print_log(epo + 1, iter / train_num_batch)

        logger.info('Epoch ' + epo_str + ' model is saved!')
        model.save_model(epo, ckpt_path)

        if epo>(args.T-2):
            # # Validation
            logger.info('-' * 111)
            logger.info('Epoch ' + epo_str + ' validation')
            model.set_phase('eval')
            model.infer_init()
            
            for iter, pack in enumerate(tqdm(val_data_loader)):
                model.unpack(pack)
                model.infer_multi(epo + 1, val_path, dict_path, crf_path, vis=(iter<50), dict=args.dict, crf=args.crf)

            # Evaluate mIoU
            miou_temp, th_temp = eval_in_script(logger=logger, eval_list='train', name=args.name, dict_dir='./dict')
            miou_temp_str = str(round(miou_temp,3))
            th_temp_str = str(round(th_temp,3))
            miou_list.append(miou_temp_str)
            logger.info('Epoch ' + epo_str + ' max miou : ' + miou_temp_str + ' at ' + th_temp_str)
            logger.info(miou_list)

            if miou_temp>max_miou:
                max_miou = miou_temp
                logger.info('New record!')