import os, time
import os.path as osp
import argparse
import glob
import random
import pdb
from turtle import pos

import numpy as np
from numpy.core.fromnumeric import size
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import itertools

# Image tools
import cv2
import PIL
from PIL import Image
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('Agg') 
from torchvision import transforms

import voc12.data
from tools import utils, pyutils
from tools.imutils import save_img, denorm, _crf_with_alpha

# import resnet38d
from networks import resnet38d, vgg16d
from networks import resnet101


def set_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad

class AttractionLoss(nn.Module):
    def __init__(self):
        super(AttractionLoss, self).__init__()
    def forward(self,anchor,positive):

        return F.l1_loss(positive,anchor)

class RepulsionLoss(nn.Module):

    def __init__(self, margin):
        super(RepulsionLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor_pos,negative_pos,size_average=True):
        
        distance_negative = ((anchor_pos - negative_pos)).pow(2).sum(1) #B C

        loss = (F.relu(- distance_negative + self.margin)).mean()
        return loss


class model_WSSS():

    def __init__(self, args, logger):

        self.args = args
        self.categories = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
                           'bus', 'car', 'cat', 'chair', 'cow',
                           'diningtable', 'dog', 'horse', 'motorbike', 'person',
                           'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

        # Common things
        self.phase = 'train'
        self.dev = 'cuda'
        self.bce = nn.BCEWithLogitsLoss()
        self.bs = args.batch_size
        self.logger = logger

        # Hyper-parameters
        self.T = args.T  # T
        self.M = args.M  # Margin 
        self.TH = args.TH  # Threshold 
        self.W = args.W  # Weight for each term in loss 

        # Model attributes
        self.net_names = ['net_main']
        self.base_names = ['cls','attract','repulse']
        self.loss_names = ['loss_' + bn for bn in self.base_names]
        self.acc_names = ['acc_' + bn for bn in self.base_names]

        self.nets = []
        self.opts = []

        # Evaluation-related
        self.running_loss = [0] * len(self.loss_names)
        self.right_count = [0] * len(self.acc_names)
        self.wrong_count = [0] * len(self.acc_names)
        self.accs = [0] * len(self.acc_names)
        self.count = 0
        self.num_count = 0

        self.val_wrong = 0
        self.val_right = 0

        # Define networks
        self.net_main = resnet38d.Net_gpp()

        self.attract_loss = AttractionLoss()
        self.repulse_loss = RepulsionLoss(self.M)
        self.L2 = nn.MSELoss()
        self.L1 = nn.L1Loss()

        # Initialize networks with ImageNet pretrained weight
        self.net_main.load_state_dict(resnet38d.convert_mxnet_to_torch('./pretrained/resnet_38d.params'), strict=False)
       

    # Save networks
    def save_model(self, epo, ckpt_path):
        epo_str = str(epo).zfill(3)
        torch.save(self.net_main.module.state_dict(), ckpt_path + '/' + epo_str + 'net_main.pth')

    # Load networks
    def load_model(self, epo, ckpt_path):
        epo_str = str(epo).zfill(3)
        self.net_main.load_state_dict(torch.load(ckpt_path + '/' + epo_str + 'net_main.pth'), strict=True)

        self.net_main = torch.nn.DataParallel(self.net_main.to(self.dev))

    # Set networks' phase (train/eval)
    def set_phase(self, phase):

        if phase == 'train':
            self.phase = 'train'
            for name in self.net_names:
                getattr(self, name).train()
            self.logger.info('Phase : train')

        else:
            self.phase = 'eval'
            for name in self.net_names:
                getattr(self, name).eval()
            self.logger.info('Phase : eval')

    # Set optimizers and upload networks on multi-gpu
    def train_setup(self):

        args = self.args

        param_main = self.net_main.get_parameter_groups()


        self.opt_main = utils.PolyOptimizer([
            {'params': param_main[0], 'lr': 1 * args.lr, 'weight_decay': args.wt_dec},
            {'params': param_main[1], 'lr': 2 * args.lr, 'weight_decay': 0},  # non-scratch bias
            {'params': param_main[2], 'lr': 10 * args.lr, 'weight_decay': args.wt_dec},  # scratch weight
            {'params': param_main[3], 'lr': 20 * args.lr, 'weight_decay': 0}  # scratch bias
        ],
            lr=args.lr, weight_decay=args.wt_dec, max_step=args.max_step)

     

        self.logger.info('Poly-optimizer for net_main is defined.')
        self.logger.info('* Base learning rate : ' + str(args.lr))
        self.logger.info('* non-scratch layer weight lr : ' + str(args.lr))
        self.logger.info('* non-scratch layer bias lr : ' + str(2 * args.lr))
        self.logger.info('* scratch layer weight lr : ' + str(10 * args.lr))
        self.logger.info('* scratch layer bias lr : ' + str(20 * args.lr))
        self.logger.info('* Weight decaying : ' + str(args.wt_dec) + ', max step : ' + str(args.max_step))

        # self.logger.info('Net_sup will be updated by MVW only.')

        self.net_main = torch.nn.DataParallel(self.net_main.to(self.dev))
        self.logger.info('Networks are uploaded on multi-gpu.')

        self.nets.append(self.net_main)

    # Unpack data pack from data_loader
    def unpack(self, pack):

        if self.phase == 'train':
            # self.img_o = pack['img_ori'].to(self.dev)  # B x 3 x H x W
            self.img_a = pack['img_a'].to(self.dev)  # B x 3 x H x W
            self.img_p = pack['img_p'].to(self.dev)  # B x 3 x H x W
            self.label = pack['label'].to(self.dev)  # B x 20
            self.name = pack['name']  # list of image names

        if self.phase == 'eval':
            self.img = pack[1]
            # To handle MSF dataset
            for i in range(8):
                self.img[i] = self.img[i].to(self.dev)
            self.label = pack[2].to(self.dev)
            self.name = pack[0][0]

        self.split_label()

    # Do forward/backward propagation and call optimizer to update the networks
    def update(self, epo):

        # Tensor dimensions
        B = self.img_a.shape[0]
        H = self.img_a.shape[2]
        W = self.img_a.shape[3]
        C = 20  # Number of cls

        self.B = B
        self.C = C

        p_idx = []
        n_idx = []

        j_list = [x for x in range(B)]

        for i in range(B):
            anchor = torch.nonzero(self.label[i]).detach().cpu().numpy()
            end = j_list[-1]
            for j in j_list:
                left = torch.nonzero(self.label[j]).detach().cpu().numpy()
                intersection = np.intersect1d(anchor, left)
                if j == end and len(p_idx) == i:
                    p_idx.append(i)

                if len(intersection) > 0 and len(p_idx) < (i + 1):  
                    p_idx.append(j)
                if len(intersection) == 0 and len(n_idx) < (i + 1):
                    n_idx.append(j)

                if len(p_idx) == (i + 1) and len(n_idx) == (i + 1):
                    random.shuffle(j_list)
      
        #
        ################################################### Update network ###################################################
        #
        self.img = self.img_a
        self.opt_main.zero_grad()

        self.bs = 8 if epo<self.T else 32

        #Attract
        cam, self.out, gpp = self.net_main(self.img)

        cam_large = F.interpolate(F.relu(cam),size=(H,W),mode='bilinear',align_corners=False)

        cam_norm = self.max_norm(cam_large)
        cam_fg = torch.max(cam_norm*self.label.view(B,C,1,1),dim=1,keepdim=True)[0] # B 1 H W
        
        mask_pos = (cam_fg<self.TH[0])*cam_fg
      
        self.loss_cls = self.W[0] * self.bce(self.out[:8], self.label[:8])
                
        loss = self.loss_cls

        ################################################### Attraction ###################################################

        if self.W[1] > 0 and epo>= self.T:
            cam_mask, self.out_mask, gpp_masked = self.net_main(self.img*mask_pos) #original

            anchor = F.adaptive_avg_pool2d(F.relu(gpp),(1,1)).view(B,C)*self.label
            positive = F.adaptive_avg_pool2d(F.relu(gpp_masked),(1,1)).view(B,C)*self.label
            
            self.loss_attract = self.W[1]*self.attract_loss(anchor,positive)
            
            loss += self.loss_attract

            cam_n = self.max_norm(cam)*self.label.view(B,C,1,1)
            cam_fn = F.interpolate(cam_n,size=(H,W),mode='bilinear',align_corners=False)*self.label.view(B,C,1,1)
            cam_fg = torch.max(cam_fn,dim=1,keepdim=True)[0] # B 1 H W
        else:
            self.loss_attract = torch.Tensor([0])

        ################################################### Repulusion ###################################################

        if self.W[2] > 0 and epo>= self.T:

            mask_anchor_bg = (cam_n<self.TH[1]).float()
            mask_negative = (cam_n<1.0).float()

            feat_norm = gpp/(torch.max(torch.abs(gpp).view(B,C,-1),dim=-1)[0].view(B,C,1,1)+1e-5) # 

            anchor_pos = (torch.sum((mask_anchor_bg*(feat_norm)).view(B,C,-1),dim=-1,keepdim=True)/(torch.sum(mask_anchor_bg.view(B,C,-1),dim=-1,keepdim=True)+1e-5)).view(B,C)*self.label
            negative_pos =  (torch.sum((mask_negative*(feat_norm)).view(B,C,-1),dim=-1,keepdim=True)/(torch.sum(mask_negative.view(B,C,-1),dim=-1,keepdim=True)+1e-5)).view(B,C)*self.label
            
            negative_pos = negative_pos[n_idx]

            distance_negative_pos = ((anchor_pos - negative_pos)).pow(2).sum(1)
            all = (distance_negative_pos<self.M).cpu().numpy().flatten()
            hard_triplets_pos = np.where(all == 1)
        

            anchor_pos = anchor_pos[hard_triplets_pos]
            negative_pos = negative_pos[hard_triplets_pos]
            
            if len(hard_triplets_pos[0])>4:
                self.loss_repulse = self.W[2]*self.repulse_loss(anchor_pos,negative_pos)
                loss += self.loss_repulse
            else:
                self.loss_repulse = torch.Tensor([0])
        else:
            self.loss_repulse = torch.Tensor([0])

        loss.backward()
        self.opt_main.step()
              
       
        ################################################### Export ###################################################


        for i in range(len(self.loss_names)):
            self.running_loss[i] += getattr(self, self.loss_names[i]).item()
        self.count += 1

        self.count_rw(self.label, self.out, 0)
    
       
    # Initialization for msf-infer
    def infer_init(self):
        n_gpus = torch.cuda.device_count()
        self.net_main_replicas = torch.nn.parallel.replicate(self.net_main.module, list(range(n_gpus)))

    # (Multi-Thread) Infer MSF-CAM and save image/cam_dict/crf_dict
    def infer_multi(self, epo, val_path, dict_path, crf_path, vis=False, dict=False, crf=False):

        if self.phase != 'eval':
            self.set_phase('eval')

        epo_str = str(epo).zfill(3)
        gt = self.label[0].cpu().detach().numpy()
        self.gt_cls = np.nonzero(gt)[0]

        _, _, H, W = self.img[2].shape
        n_gpus = torch.cuda.device_count()

        def _work(i, img):
            with torch.no_grad():
                with torch.cuda.device(i % n_gpus):
                    cam, _ , _ = self.net_main_replicas[i % n_gpus](img.cuda())
                    cam = F.upsample(cam, (H, W), mode='bilinear', align_corners=False)[0]
                    cam = F.relu(cam)
                    cam = cam.cpu().numpy() * self.label.clone().cpu().view(20, 1, 1).numpy()

                    if i % 2 == 1:
                        cam = np.flip(cam, axis=-1)
                    return cam

        thread_pool = pyutils.BatchThreader(_work, list(enumerate(self.img)), batch_size=8, prefetch_size=0,
                                            processes=8)
        cam_list = thread_pool.pop_results()
        cam = np.sum(cam_list, axis=0)
        cam_max = np.max(cam, (1, 2), keepdims=True)
        norm_cam = cam / (cam_max + 1e-5)

        self.cam_dict = {}
        for i in range(20):
            if self.label[0, i] > 1e-5:
                self.cam_dict[i] = norm_cam[i]

                

        if vis:
            img_np = denorm(self.img[2][0]).cpu().detach().data.permute(1, 2, 0).numpy()
            for c in self.gt_cls:
                save_img(osp.join(val_path, epo_str + '_' + self.name + '_cam_' + self.categories[c] + '.png'), img_np,
                         norm_cam[c])

        if dict:
            np.save(osp.join(dict_path, self.name + '.npy'), self.cam_dict)

        if crf:
            for a in self.args.alphas:
                crf_dict = _crf_with_alpha(self.cam_dict, self.name, alpha=a)
                np.save(osp.join(crf_path, str(a).zfill(2), self.name + '.npy'), crf_dict)

    # Print loss/accuracy (and re-initialize them)
    def print_log(self, epo, iter):

        loss_str = ''
        acc_str = ''

        for i in range(len(self.loss_names)):
            loss_str += self.loss_names[i] + ' : ' + str(round(self.running_loss[i] / self.count, 5)) + ', '

        for i in range(len(self.acc_names)):
            if self.right_count[i] != 0:
                acc = 100 * self.right_count[i] / (self.right_count[i] + self.wrong_count[i])
                acc_str += self.acc_names[i] + ' : ' + str(round(acc, 2)) + ', '
                self.accs[i] = acc

        self.logger.info(loss_str[:-2])
        self.logger.info(acc_str[:-2])

        self.running_loss = [0] * len(self.loss_names)
        self.right_count = [0] * len(self.acc_names)
        self.wrong_count = [0] * len(self.acc_names)
        self.count = 0

    def count_rw(self, label, out, idx):
        for b in range(self.bs):  # 8
            gt = label[b].cpu().detach().numpy()
            gt_cls = np.nonzero(gt)[0]
            num = len(np.nonzero(gt)[0])
            pred = out[b].cpu().detach().numpy()
            pred_cls = pred.argsort()[-num:][::-1]

            for c in gt_cls:
                if c in pred_cls:
                    self.right_count[idx] += 1
                else:
                    self.wrong_count[idx] += 1

 

    # Max_norm
    def max_norm(self, cam_cp):
        N, C, H, W = cam_cp.size()
        cam_cp = F.relu(cam_cp)
        max_v = torch.max(cam_cp.view(N, C, -1), dim=-1)[0].view(N, C, 1, 1)
        min_v = torch.min(cam_cp.view(N, C, -1), dim=-1)[0].view(N, C, 1, 1)
        cam_cp = F.relu(cam_cp - min_v - 1e-5) / (max_v - min_v + 1e-5)
        return cam_cp

    def cam_l1(self, cam1, cam2):
        return torch.mean(torch.abs(cam2.detach() - cam1))

    def split_label(self):

        bs = self.label.shape[0] if self.phase == 'train' else 1  # self.label.shape[0]
        self.label_exist = torch.zeros(bs, 20).cuda()
        # self.label_remain = self.label.clone()
        for i in range(bs):
            label_idx = torch.nonzero(self.label[i], as_tuple=False)
            rand_idx = torch.randint(0, len(label_idx), (1,))
            target = label_idx[rand_idx][0]
            # self.label_remain[i, target] = 0
            self.label_exist[i, target] = 1
        self.label_remain = self.label - self.label_exist

        self.label_all = self.label  # [:16]
