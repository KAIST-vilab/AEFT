import os
import os.path as osp
import pandas as pd
import numpy as np
from PIL import Image
import multiprocessing
import argparse
import pdb


################################################################################
# Evaluate the performance by computing mIoU.
# It assumes that every CAM or CRF dict file is already infered and saved.  
# For CAM, threshold will be searched in range [0.01, 0.80].
#
# If you want to evaluate CAM performance...
# python evaluation.py --name [exp_name] --task cam --dict_dir dict
#
# Or if you want to evaluate CRF performance of certain alpha (let, a1)...
# python evaluation.py --name [exp_name] --task crf --dict_dir crf/a1
#
# For AFF evaluation, go to evaluation_aff.py
################################################################################


categories = ['background','aeroplane','bicycle','bird','boat','bottle','bus','car','cat','chair','cow',
              'diningtable','dog','horse','motorbike','person','pottedplant','sheep','sofa','train','tvmonitor']

def do_python_eval(predict_folder, gt_folder, name_list, num_cls, task, threshold, printlog=False):
    TP = []
    P = []
    T = []
    for i in range(num_cls):
        TP.append(multiprocessing.Value('i', 0, lock=True))
        P.append(multiprocessing.Value('i', 0, lock=True))
        T.append(multiprocessing.Value('i', 0, lock=True))
    
    def compare(start,step,TP,P,T,task,threshold):
        for idx in range(start,len(name_list),step):
            name = name_list[idx]

            if task=='cam':
                predict_file = os.path.join(predict_folder,'%s.npy'%name)
                predict_dict = np.load(predict_file, allow_pickle=True).item()
                h, w = list(predict_dict.values())[0].shape
                tensor = np.zeros((21,h,w),np.float32)
                for key in predict_dict.keys():
                    tensor[key+1] = predict_dict[key]
                tensor[0,:,:] = threshold 
                predict = np.argmax(tensor, axis=0).astype(np.uint8)

            if task=='crf':
                predict_file = os.path.join(predict_folder,'%s.npy'%name)
                predict_dict = np.load(predict_file, allow_pickle=True).item()
                h, w = list(predict_dict.values())[0].shape
                tensor = np.zeros((21,h,w),np.float32)
                for key in predict_dict.keys():
                    tensor[key] = predict_dict[key]
                predict = np.argmax(tensor, axis=0).astype(np.uint8)

            if task=='dl':
                predict_file = os.path.join(predict_folder,'%s.npy'%name)
                predict_dict = np.load(predict_file, allow_pickle=True).item()
                h, w = list(predict_dict.values())[0].shape
                tensor = np.zeros((21,h,w),np.float32)
                for key in predict_dict.keys():
                    tensor[key] = predict_dict[key]
                predict = np.argmax(tensor, axis=0).astype(np.uint8)

            if task=='png':
                predict_file = os.path.join(predict_folder, '%s.png' % name)
                predict = np.array(Image.open(predict_file))

            gt_file = os.path.join(gt_folder,'%s.png'%name)
            gt = np.array(Image.open(gt_file))
            cal = gt<255 # Reject object boundary
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
        p = multiprocessing.Process(target=compare, args=(i,8,TP,P,T,task,threshold))
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
    return loglist

def writedict(file, dictionary):
    s = ''
    for key in dictionary.keys():
        sub = '%s:%s  '%(key, dictionary[key])
        s += sub
    s += '\n'
    file.write(s)

def writelog(filepath, metric, comment):
    filepath = filepath
    logfile = open(filepath,'a')
    import time
    logfile.write(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    logfile.write('\t%s\n'%comment)
    writedict(logfile, metric)
    logfile.write('=====================================\n')
    logfile.close()


def eval_in_script(logger=None, eval_list='train', task='cam', name=None, dict_dir=None, gt_dir='./data/VOC2012/SegmentationClass'):
    
    eval_list = './data/VOC2012/ImageSets/Segmentation/' + eval_list + '.txt'
    df = pd.read_csv(eval_list, names=['filename'])
    name_list = df['filename'].values
    pred_dir = osp.join('./experiments', name, dict_dir)  

    max_miou = 0
    max_th = 0
    for i in range(10,30):
        t = i/100.#+0.05
        loglist = do_python_eval(pred_dir, gt_dir, name_list, 21, task, t, printlog=False)
        logger.info('%d/60 threshold: %.3f\tmIoU: %.3f%%'%(i, t, loglist['mIoU']))
        
        miou_temp = loglist['mIoU']
        if miou_temp>max_miou:
            max_miou = miou_temp
            max_th = t

    return max_miou, max_th


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--list", default="train", type=str)
    parser.add_argument("--task", required=True, type=str)
    parser.add_argument("--name", required=True, type=str)
    parser.add_argument("--dict_dir", required=True, type=str)
    parser.add_argument("--gt_dir", default='./data/VOC2012/SegmentationClass', type=str)
    
    args = parser.parse_args()

    eval_list = './data/VOC2012/ImageSets/Segmentation/' + args.list + '.txt'
    df = pd.read_csv(eval_list, names=['filename'])
    name_list = df['filename'].values

    pred_dir = osp.join('./experiments', args.name, args.dict_dir)  
    print('Evaluate ' + pred_dir + ' with ' + eval_list)

    if args.task=='cam':
        for i in range(10,35):
            t = i/100.#+0.05
            loglist = do_python_eval(pred_dir, args.gt_dir, name_list, 21, args.task, t, printlog=False)
            print('%d/60 threshold: %.3f\tmIoU: %.3f%%'%(i, t, loglist['mIoU']))
    
    elif args.task=='crf':
        loglist = do_python_eval(pred_dir, args.gt_dir, name_list, 21, args.task, 0, printlog=True)

    elif args.task=='dl'or args.task=='png':
        loglist = do_python_eval(pred_dir, args.gt_dir, name_list, 21, args.task, 0, printlog=True)
