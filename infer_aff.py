import os
import os.path as osp
import numpy as np
import torch
import random
import pdb
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
# python evaluation_aff.py --name [exp_name] --low la --high ha
#
# For CAM or CRF evaluation, go to evaluation.py
####################################################################################################

def infer_aff(args, model, infer_data_loader, rw_path, alpha):

    model.eval()

    for iter, (name, img) in enumerate(tqdm(infer_data_loader)):

        name = name[0]

        orig_shape = img.shape
        padded_size = (int(np.ceil(img.shape[2]/8)*8), int(np.ceil(img.shape[3]/8)*8))

        p2d = (0, padded_size[1] - img.shape[3], 0, padded_size[0] - img.shape[2])
        img = F.pad(img, p2d)

        dheight = int(np.ceil(img.shape[2]/8))
        dwidth = int(np.ceil(img.shape[3]/8))

        cam = np.load(osp.join('./experiments', args.name, 'dict', name +'.npy'), allow_pickle=True).item()

        cam_full_arr = np.zeros((21, orig_shape[2], orig_shape[3]), np.float32)
        for k, v in cam.items():
            cam_full_arr[k+1] = v
        cam_full_arr[0] = (1 - np.max(cam_full_arr[1:], (0), keepdims=False))**alpha
        #cam_full_arr[0] = 0.2
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
                img_8 = img[0].numpy().transpose((1,2,0))#F.interpolate(img, (dheight,dwidth), mode='bilinear')[0].numpy().transpose((1,2,0))
                img_8 = np.ascontiguousarray(img_8)
                mean = (0.485, 0.456, 0.406)
                std = (0.229, 0.224, 0.225)
                img_8[:,:,0] = (img_8[:,:,0]*std[0] + mean[0])*255
                img_8[:,:,1] = (img_8[:,:,1]*std[1] + mean[1])*255
                img_8[:,:,2] = (img_8[:,:,2]*std[2] + mean[2])*255
                img_8[img_8 > 255] = 255
                img_8[img_8 < 0] = 0
                img_8 = img_8.astype(np.uint8)
                cam_rw = cam_rw[0].cpu().numpy()
                cam_rw = imutils.crf_inference(img_8, cam_rw, t=1)
                # cam_rw = imutils.crf_inference_ysh(name,cam_rw,t=10)
                cam_rw = torch.from_numpy(cam_rw).view(1, 21, img.shape[2], img.shape[3]).cuda()


            _, cam_rw_pred = torch.max(cam_rw, 1)

            res = np.uint8(cam_rw_pred.cpu().data[0])[:orig_shape[2], :orig_shape[3]]

            imageio.imwrite(osp.join(rw_path, name + '.png'), res)

categories = ['background','aeroplane','bicycle','bird','boat','bottle','bus','car','cat','chair','cow',
              'diningtable','dog','horse','motorbike','person','pottedplant','sheep','sofa','train','tvmonitor']


def do_python_eval(predict_folder, gt_folder, name_list, num_cls=21, input_type='png', threshold=1.0, printlog=False):
    TP = []
    P = []
    T = []
    for i in range(num_cls):
        TP.append(multiprocessing.Value('i', 0, lock=True))
        P.append(multiprocessing.Value('i', 0, lock=True))
        T.append(multiprocessing.Value('i', 0, lock=True))
    
    def compare(start,step,TP,P,T,input_type,threshold):
        for idx in range(start,len(name_list),step):
            name = name_list[idx]
            if input_type == 'png':
                predict_file = os.path.join(predict_folder,'%s.png'%name)
                predict = np.array(Image.open(predict_file)) #cv2.imread(predict_file)
            elif input_type == 'npy':
                predict_file = os.path.join(predict_folder,'%s.npy'%name)
                predict_dict = np.load(predict_file, allow_pickle=True).item()
                h, w = list(predict_dict.values())[0].shape
                tensor = np.zeros((21,h,w),np.float32)
                for key in predict_dict.keys():
                    tensor[key+1] = predict_dict[key]
                tensor[0,:,:] = threshold 
                predict = np.argmax(tensor, axis=0).astype(np.uint8)
            elif input_type == 'npy_crf':
                predict_file = os.path.join(predict_folder,'%s.npy'%name)
                predict_dict = np.load(predict_file, allow_pickle=True).item()
                h, w = list(predict_dict.values())[0].shape
                tensor = np.zeros((21,h,w),np.float32)
                for key in predict_dict.keys():
                    tensor[key] = predict_dict[key]
                predict = np.argmax(tensor, axis=0).astype(np.uint8)

            gt_file = os.path.join(gt_folder,'%s.png'%name)
            gt = np.array(Image.open(gt_file))
            cal = gt<255
            mask = (predict==gt) * cal
      
            for i in range(num_cls):
                P[i].acquire()
                P[i].value += np.sum((predict==i)*cal)
                P[i].release()
                T[i].acquire()
                T[i].value += np.sum((gt==i)*cal)
                T[i].release()
                TP[i].acquire()
                TP[i].value += np.sum((gt==i)*mask)
                TP[i].release()
    p_list = []
    for i in range(8):
        p = multiprocessing.Process(target=compare, args=(i,8,TP,P,T,input_type,threshold))
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
        IoU.append(TP[i].value/(T[i].value+P[i].value-TP[i].value+1e-10))
        T_TP.append(T[i].value/(TP[i].value+1e-10))
        P_TP.append(P[i].value/(TP[i].value+1e-10))
        FP_ALL.append((P[i].value-TP[i].value)/(T[i].value + P[i].value - TP[i].value + 1e-10))
        FN_ALL.append((T[i].value-TP[i].value)/(T[i].value + P[i].value - TP[i].value + 1e-10))
    loglist = {}
    for i in range(num_cls):
        loglist[categories[i]] = IoU[i] * 100
               
    miou = np.mean(np.array(IoU))
    loglist['mIoU'] = miou * 100
    if printlog:
        for i in range(num_cls):
            if i%2 != 1:
                print('%11s:%7.3f%%'%(categories[i],IoU[i]*100),end='\t')
            else:
                print('%11s:%7.3f%%'%(categories[i],IoU[i]*100))
        print('\n======================================================')
        print('%11s:%7.3f%%'%('mIoU',miou*100))
    return miou*100

def evaluation(rw_path):
    df = pd.read_csv('./data/VOC2012/ImageSets/Segmentation/train.txt', names=['filename'])
    name_list = df['filename'].values
    miou = do_python_eval(rw_path, './data/VOC2012/SegmentationClass', name_list, 21, 'png', None, printlog=True)
    return miou


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--train_list", required=True, type=str)
    parser.add_argument("--val_list", default="voc12/val.txt", type=str)
    parser.add_argument("--crop_size", default=448, type=int)
    parser.add_argument("--voc12_root", default='./data/VOC2012', type=str)
    
    parser.add_argument("--name", required=True, type=str)
    parser.add_argument("--low", required=True, type=int)
    parser.add_argument("--high", required=True, type=int)
    parser.add_argument("--dup",required=True,type=int)

    args = parser.parse_args()

    model = resnet38_aff.Net()

    ckpt_aff_path = osp.join('./experiments', args.name, 'ckpt_aff')
    rw_path = osp.join('./experiments', args.name, 'rw', str(args.low).zfill(2)+'_'+str(args.high).zfill(2)+str(args.dup).zfill(2))


    train_infer_dataset = voc12.data.VOC12ImageDataset(args.train_list, 'data/VOC2012',
                                                 transform=torchvision.transforms.Compose(
                                                                                          [np.asarray,
                                                                                           model.normalize,
                                                                                           imutils.HWC_to_CHW]))
    infer_data_loader = DataLoader(train_infer_dataset, shuffle=False,pin_memory=True)

    # infer_dataset = voc12.data.VOC12ImageDataset("voc12/val.txt", 'data/VOC2012',
    #                                              transform=torchvision.transforms.Compose(
    #                                                                                       [np.asarray,
    #                                                                                        model.normalize,
    #                                                                                        imutils.HWC_to_CHW]))
    # infer_data_loader = DataLoader(infer_dataset, shuffle=False, num_workers=8, pin_memory=True)

    dict_path = os.path.join("./experiments",args.name,'ckpt_aff','aff'+str(args.low).zfill(2)+str(args.high).zfill(2)+str(args.dup).zfill(2),'best.pth')

    model.load_state_dict(torch.load(dict_path))
    model = torch.nn.DataParallel(model).cuda()
    model.eval()

    infer_aff(args, model, infer_data_loader, rw_path, args.dup)
    miou = evaluation(rw_path)
        

    # 11
    # miou_max = -9999
    # crf_max = 0
    # for i in range(5):
    #     infer_aff(args, model, infer_data_loader, rw_path, i+14)
    #     print(i+14)
    #     miou = evaluation(rw_path)
    #     if miou>miou_max:
    #         miou_max = miou
    #         crf_max = i+11
    #
    # print(crf_max, miou_max)
    