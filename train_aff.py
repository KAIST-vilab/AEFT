import os
import os.path as osp
import numpy as np
import torch
import random
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
import voc12.data
from afftools import pyutils, imutils, torchutils
import argparse
import importlib
from networks import resnet38_aff
from networks.resnet38d import convert_mxnet_to_torch
import imageio
from PIL import Image
import multiprocessing

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

from tqdm import tqdm


###################################################################################################
# This code is based on https://github.com/jiwoon-ahn/psa.
#
# Evaluate the performance of CAM + AFF by computing mIoU.
# It assumes that every CAM and CRF dict file are already infered and saved.
# Training - Infer (random walk) - Evaluation steps are automatically done.
#
# If you want to evaluate CAM + AFF performance of low alpha (let, la) and high alpha (let, ha)...
# python train_aff.py --name [exp_name] --low la --high ha
#
# For CAM or CRF evaluation, go to evaluation.py
####################################################################################################

def infer_aff(args, model, infer_data_loader, rw_path, alpha=19):
    model.eval()

    for iter, (name, img) in enumerate(tqdm(infer_data_loader)):

        name = name[0]

        orig_shape = img.shape
        padded_size = (int(np.ceil(img.shape[2] / 8) * 8), int(np.ceil(img.shape[3] / 8) * 8))

        p2d = (0, padded_size[1] - img.shape[3], 0, padded_size[0] - img.shape[2])
        img = F.pad(img, p2d)

        dheight = int(np.ceil(img.shape[2] / 8))
        dwidth = int(np.ceil(img.shape[3] / 8))

        cam = np.load(osp.join('./experiments', args.name, 'dict', name + '.npy'), allow_pickle=True).item()


        cam_full_arr = np.zeros((21, orig_shape[2], orig_shape[3]), np.float32)
        for k, v in cam.items():
            cam_full_arr[k + 1] = v
        cam_full_arr[0] = (1 - np.max(cam_full_arr[1:], (0), keepdims=False)) ** alpha
        # cam_full_arr[0] = 0.2
        cam_full_arr = np.pad(cam_full_arr, ((0, 0), (0, p2d[3]), (0, p2d[1])), mode='constant')

        with torch.no_grad():
            aff_mat = torch.pow(model.forward(img.cuda(), True), 8)

            trans_mat = aff_mat / torch.sum(aff_mat, dim=0, keepdim=True)
            for _ in range(6):
                trans_mat = torch.matmul(trans_mat, trans_mat)

            cam_full_arr = torch.from_numpy(cam_full_arr)
            cam_full_arr = F.avg_pool2d(cam_full_arr, 8, 8)

            cam_vec = cam_full_arr.view(21, -1)

            cam_rw = torch.matmul(cam_vec.cuda(), trans_mat)
            cam_rw = cam_rw.view(1, 21, dheight, dwidth)

            cam_rw = torch.nn.Upsample((img.shape[2], img.shape[3]), mode='bilinear')(cam_rw)

            if True:
                img_8 = img[0].numpy().transpose(
                    (1, 2, 0))  # F.interpolate(img, (dheight,dwidth), mode='bilinear')[0].numpy().transpose((1,2,0))
                img_8 = np.ascontiguousarray(img_8)
                mean = (0.485, 0.456, 0.406)
                std = (0.229, 0.224, 0.225)
                img_8[:, :, 0] = (img_8[:, :, 0] * std[0] + mean[0]) * 255
                img_8[:, :, 1] = (img_8[:, :, 1] * std[1] + mean[1]) * 255
                img_8[:, :, 2] = (img_8[:, :, 2] * std[2] + mean[2]) * 255
                img_8[img_8 > 255] = 255
                img_8[img_8 < 0] = 0
                img_8 = img_8.astype(np.uint8)
                cam_rw = cam_rw[0].cpu().numpy()
                cam_rw = imutils.crf_inference(img_8, cam_rw, t=1)
                cam_rw = torch.from_numpy(cam_rw).view(1, 21, img.shape[2], img.shape[3]).cuda()

            _, cam_rw_pred = torch.max(cam_rw, 1)

            res = np.uint8(cam_rw_pred.cpu().data[0])[:orig_shape[2], :orig_shape[3]]

            imageio.imwrite(osp.join(rw_path, name + '.png'), res)


categories = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
              'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
              'tvmonitor']


def do_python_eval(predict_folder, gt_folder, name_list, num_cls=21, input_type='png', threshold=1.0, printlog=False):
    TP = []
    P = []
    T = []
    for i in range(num_cls):
        TP.append(multiprocessing.Value('i', 0, lock=True))
        P.append(multiprocessing.Value('i', 0, lock=True))
        T.append(multiprocessing.Value('i', 0, lock=True))

    def compare(start, step, TP, P, T, input_type, threshold):
        for idx in range(start, len(name_list), step):
            name = name_list[idx]
            if input_type == 'png':
                predict_file = os.path.join(predict_folder, '%s.png' % name)
                predict = np.array(Image.open(predict_file))  # cv2.imread(predict_file)
            elif input_type == 'npy':
                predict_file = os.path.join(predict_folder, '%s.npy' % name)
                predict_dict = np.load(predict_file, allow_pickle=True).item()
                h, w = list(predict_dict.values())[0].shape
                tensor = np.zeros((21, h, w), np.float32)
                for key in predict_dict.keys():
                    tensor[key + 1] = predict_dict[key]
                tensor[0, :, :] = threshold
                predict = np.argmax(tensor, axis=0).astype(np.uint8)
            elif input_type == 'npy_crf':
                predict_file = os.path.join(predict_folder, '%s.npy' % name)
                predict_dict = np.load(predict_file, allow_pickle=True).item()
                h, w = list(predict_dict.values())[0].shape
                tensor = np.zeros((21, h, w), np.float32)
                for key in predict_dict.keys():
                    tensor[key] = predict_dict[key]
                predict = np.argmax(tensor, axis=0).astype(np.uint8)

            gt_file = os.path.join(gt_folder, '%s.png' % name)
            gt = np.array(Image.open(gt_file))
            cal = gt < 255
            mask = (predict == gt) * cal

            for i in range(num_cls):
                P[i].acquire()
                P[i].value += np.sum((predict == i) * cal)
                P[i].release()
                T[i].acquire()
                T[i].value += np.sum((gt == i) * cal)
                T[i].release()
                TP[i].acquire()
                TP[i].value += np.sum((gt == i) * mask)
                TP[i].release()

    p_list = []
    for i in range(8):
        p = multiprocessing.Process(target=compare, args=(i, 8, TP, P, T, input_type, threshold))
        p.start()
        p_list.append(p)
    for p in p_list:
        p.join()
    IoU = []
    T_TP = []
    P_TP = []
    FP_ALL = []
    FN_ALL = []
    for i in range(num_cls):
        IoU.append(TP[i].value / (T[i].value + P[i].value - TP[i].value + 1e-10))
        T_TP.append(T[i].value / (TP[i].value + 1e-10))
        P_TP.append(P[i].value / (TP[i].value + 1e-10))
        FP_ALL.append((P[i].value - TP[i].value) / (T[i].value + P[i].value - TP[i].value + 1e-10))
        FN_ALL.append((T[i].value - TP[i].value) / (T[i].value + P[i].value - TP[i].value + 1e-10))
    loglist = {}
    for i in range(num_cls):
        loglist[categories[i]] = IoU[i] * 100

    miou = np.mean(np.array(IoU))
    loglist['mIoU'] = miou * 100
    if printlog:
        for i in range(num_cls):
            if i % 2 != 1:
                print('%11s:%7.3f%%' % (categories[i], IoU[i] * 100), end='\t')
            else:
                print('%11s:%7.3f%%' % (categories[i], IoU[i] * 100))
        print('\n======================================================')
        print('%11s:%7.3f%%' % ('mIoU', miou * 100))
    return miou * 100


def evaluation(rw_path, set='val'):
    df = pd.read_csv('./data/VOC2012/ImageSets/Segmentation/' + set + '.txt', names=['filename'])
    name_list = df['filename'].values
    miou = do_python_eval(rw_path, './data/VOC2012/SegmentationClass', name_list, 21, 'png', None, printlog=True)
    return miou


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--max_epoches", default=8, type=int)
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--wt_dec", default=5e-4, type=float)
    parser.add_argument("--train_list", default="voc12/train_aug.txt", type=str)
    parser.add_argument("--val_list", default="voc12/val.txt", type=str)
    parser.add_argument("--crop_size", default=448, type=int)
    parser.add_argument("--voc12_root", default='./data/VOC2012', type=str)

    parser.add_argument("--name", required=True, type=str)
    parser.add_argument("--low", required=True, type=int)
    parser.add_argument("--high", required=True, type=int)
    parser.add_argument("--dup",required=True,type=int)

    args = parser.parse_args()

    pyutils.Logger(args.name + '_aff' + '.log')

    model = resnet38_aff.Net()

    ckpt_aff_path = osp.join('./experiments', args.name, 'ckpt_aff')

    if not os.path.exists(ckpt_aff_path):
        os.makedirs(ckpt_aff_path)

    ckpt_aff_lh_path = osp.join(ckpt_aff_path, 'aff' + str(args.low).zfill(2) + str(args.high).zfill(2)+str(args.dup).zfill(2))
    if not os.path.exists(ckpt_aff_lh_path):
        os.makedirs(ckpt_aff_lh_path)

    rw_path = osp.join('./experiments', args.name, 'rw', str(args.low).zfill(2) + '_' + str(args.high).zfill(2)+str(args.dup).zfill(2))

    if not os.path.exists(rw_path):
        os.makedirs(rw_path)

    low_crf_dir = osp.join('./experiments', args.name, 'crf', str(args.low).zfill(2))
    high_crf_dir = osp.join('./experiments', args.name, 'crf', str(args.high).zfill(2))

    train_dataset = voc12.data.VOC12AffDataset(args.train_list, label_la_dir=low_crf_dir, label_ha_dir=high_crf_dir,
                                               voc12_root=args.voc12_root, cropsize=args.crop_size, radius=5,
                                               joint_transform_list=[
                                                   None,
                                                   None,
                                                   imutils.RandomCrop(args.crop_size),
                                                   imutils.RandomHorizontalFlip()
                                               ],
                                               img_transform_list=[
                                                   transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3,
                                                                          hue=0.1),
                                                   np.asarray,
                                                   model.normalize,
                                                   imutils.HWC_to_CHW
                                               ],
                                               label_transform_list=[
                                                   None,
                                                   None,
                                                   None,
                                                   imutils.AvgPool2d(8)
                                               ])


    def worker_init_fn(worker_id):
        np.random.seed(1 + worker_id)


    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                   pin_memory=True, drop_last=True, worker_init_fn=worker_init_fn)

    infer_train_dataset = voc12.data.VOC12ImageDataset("voc12/train.txt", 'data/VOC2012',
                                                       transform=torchvision.transforms.Compose(
                                                           [np.asarray,
                                                            model.normalize,
                                                            imutils.HWC_to_CHW]))
    infer_train_data_loader = DataLoader(infer_train_dataset, shuffle=False,pin_memory=True)

    max_step = len(train_dataset) // args.batch_size * args.max_epoches

    param_groups = model.get_parameter_groups()
    optimizer = torchutils.PolyOptimizer([
        {'params': param_groups[0], 'lr': args.lr, 'weight_decay': args.wt_dec},
        {'params': param_groups[1], 'lr': 2 * args.lr, 'weight_decay': 0},
        {'params': param_groups[2], 'lr': 10 * args.lr, 'weight_decay': args.wt_dec},
        {'params': param_groups[3], 'lr': 20 * args.lr, 'weight_decay': 0}
    ], lr=args.lr, weight_decay=args.wt_dec, max_step=max_step)

    # weights_dict = convert_mxnet_to_torch('./pretrained/resnet_38d.params')
    weights_dict = torch.load("./pretrained/res38_cls.pth")
    model.load_state_dict(weights_dict, strict=False)
    model = torch.nn.DataParallel(model).cuda()
    model.train()

    avg_meter = pyutils.AverageMeter('loss', 'bg_loss', 'fg_loss', 'neg_loss', 'bg_cnt', 'fg_cnt', 'neg_cnt')

    timer = pyutils.Timer("Session started: ")

    max_miou = 0

    for ep in range(args.max_epoches):

        for iter, pack in enumerate(train_data_loader):

            aff = model.forward(pack[0])

            bg_label = pack[1][0].cuda(non_blocking=True)
            fg_label = pack[1][1].cuda(non_blocking=True)
            neg_label = pack[1][2].cuda(non_blocking=True)

            bg_count = torch.sum(bg_label) + 1e-5
            fg_count = torch.sum(fg_label) + 1e-5
            neg_count = torch.sum(neg_label) + 1e-5

            bg_loss = torch.sum(- bg_label * torch.log(aff + 1e-5)) / bg_count
            fg_loss = torch.sum(- fg_label * torch.log(aff + 1e-5)) / fg_count
            neg_loss = torch.sum(- neg_label * torch.log(1. + 1e-5 - aff)) / neg_count

            loss = bg_loss / 4 + fg_loss / 4 + neg_loss / 2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            avg_meter.add({
                'loss': loss.item(),
                'bg_loss': bg_loss.item(), 'fg_loss': fg_loss.item(), 'neg_loss': neg_loss.item(),
                'bg_cnt': bg_count.item(), 'fg_cnt': fg_count.item(), 'neg_cnt': neg_count.item()
            })

            if (optimizer.global_step) % 50 == 0:
                timer.update_progress(optimizer.global_step / max_step)

                print('Iter:%5d/%5d' % (optimizer.global_step - 1, max_step),
                      'loss:%.4f %.4f %.4f %.4f' % avg_meter.get('loss', 'bg_loss', 'fg_loss', 'neg_loss'),
                      'cnt:%.0f %.0f %.0f' % avg_meter.get('bg_cnt', 'fg_cnt', 'neg_cnt'),
                      'lr: %.4f' % (optimizer.param_groups[0]['lr']), flush=True)

                avg_meter.pop()

            if (optimizer.global_step) % 200 == 0 and optimizer.global_step>1590:
                print(str(optimizer.global_step) + ' infer')
                ckpt_str = osp.join(ckpt_aff_path,
                                    'aff' + str(args.low).zfill(2) + str(args.high).zfill(2) +str(args.dup).zfill(2) +'/' + str(ep).zfill(
                                        2) + '.pth')
                torch.save(model.module.state_dict(), ckpt_str)

                # Infer and evaluation on train set
                infer_aff(args, model, infer_train_data_loader, rw_path,alpha=args.dup)
                miou = evaluation(rw_path, set='train')

                if miou > max_miou:
                    max_miou = miou
                    torch.save(model.module.state_dict(), osp.join(ckpt_aff_path, 'aff' + str(args.low).zfill(2) + str(
                        args.high).zfill(2) +str(args.dup).zfill(2)+ '/best.pth'))
                    print('New record!')

                print(max_miou, miou)

        else:
            print('')
            timer.reset_stage()