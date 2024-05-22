import time
import os
import math
import argparse
from glob import glob
from pathlib import Path
from collections import OrderedDict
import random
import warnings
from datetime import datetime

import numpy as np
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from skimage.io import imread, imsave

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import datasets, models, transforms

from dataset import Dataset

import nets.unet as unet
from metrics import dice_coef, batch_iou, mean_iou, iou_score ,ppv,sensitivity,specificity,boundary_iou
import losses
from utils import str2bool, count_params
import joblib
from hausdorff import hausdorff_distance
import imageio
import pandas as pd

IMG_PATH = glob(r"/home/cavin/Experiment/ZR/EXP/Med3D/data/processed/2D/testImage/*")
MASK_PATH = glob(r"/home/cavin/Experiment/ZR/EXP/Med3D/data/processed/2D/testMask/*")

# 先执行GetPicture 再执行 Calculate
# MODE = 'GetPicture' #'Calculate'

MODE = 'Calculate'

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default='ZR_UDMamba_woDS',
                        help='model name')
    parser.add_argument('--mode', default=MODE,
                        help='GetPicture or Calculate')
    parser.add_argument('--outpath', default='/home/cavin/Experiment/ZR/EXP/Med3D/output/cbam_output', type=str,
                        help='model output path')
    parser.add_argument('--modelpath', default='/home/cavin/Experiment/ZR/EXP/Med3D/models/cbam_models', type=str,
                    help='model output path')

    args = parser.parse_args()

    return args


def main():
    val_args = parse_args()
    
    pkl_val = os.path.join(val_args.modelpath, val_args.name, 'args.pkl')
    
    args = joblib.load(pkl_val)
    
    name = os.path.join(val_args.outpath, args.name)
    if not os.path.exists(name):
        os.makedirs(name)

    print('Config -----')
    for arg in vars(args):
        print('%s: %s' %(arg, getattr(args, arg)))
    print('------------')

    pkl_arg = os.path.join(val_args.modelpath, args.name, 'args.pkl')
    joblib.dump(args, pkl_arg)

    # create model
    print("=> creating model %s" %args.arch)
    model = unet.__dict__[args.arch](args)

    model = model.cuda()

    # Data loading code
    img_paths = IMG_PATH
    mask_paths = MASK_PATH

    val_img_paths = img_paths
    val_mask_paths = mask_paths

    mode_path = os.path.join(val_args.modelpath, args.name, 'model.pth')
    
    model.load_state_dict(torch.load(mode_path))
    model.eval()

    val_dataset = Dataset(args, val_img_paths, val_mask_paths)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        drop_last=False)

    if val_args.mode == "GetPicture":

        """
        获取并保存模型生成的标签图
        """
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')

            with torch.no_grad():
                for i, (input, target) in tqdm(enumerate(val_loader), total=len(val_loader)):
                    input = input.cuda()
                    #target = target.cuda()

                    # compute output
                    if args.deepsupervision:
                        output = model(input)[-1]
                    else:
                        output = model(input)
                    #print("img_paths[i]:%s" % img_paths[i])
                    output = torch.sigmoid(output).data.cpu().numpy()
                    img_paths = val_img_paths[args.batch_size*i:args.batch_size*(i+1)]
                    #print("output_shape:%s"%str(output.shape))

                    for i in range(output.shape[0]):
                        npName = os.path.basename(img_paths[i])
                        overNum = npName.find(".npy")
                        rgbName = npName[0:overNum]
                        rgbName = rgbName  + ".png"
                        rgbPic = np.zeros([160, 160, 3], dtype=np.uint8)
                        for idx in range(output.shape[2]):
                            for idy in range(output.shape[3]):
                                if output[i,0,idx,idy] > 0.5:
                                    rgbPic[idx, idy, 0] = 0
                                    rgbPic[idx, idy, 1] = 128
                                    rgbPic[idx, idy, 2] = 0
                                if output[i,1,idx,idy] > 0.5:
                                    rgbPic[idx, idy, 0] = 255
                                    rgbPic[idx, idy, 1] = 0
                                    rgbPic[idx, idy, 2] = 0
                                if output[i,2,idx,idy] > 0.5:
                                    rgbPic[idx, idy, 0] = 255
                                    rgbPic[idx, idy, 1] = 255
                                    rgbPic[idx, idy, 2] = 0
                            
                        imname = args.name + '/'+ rgbName
                        imsavepath = os.path.join(val_args.outpath, imname)
                        imsave(imsavepath, rgbPic)

            torch.cuda.empty_cache()
        """
        将验证集中的GT numpy格式转换成图片格式并保存
        """
        print("Saving GT,numpy to picture")
        
        val_gt_path = os.path.join(val_args.outpath, args.name, "GT/")

        if not os.path.exists(val_gt_path):
            os.mkdir(val_gt_path)
        for idx in tqdm(range(len(val_mask_paths))):
            mask_path = val_mask_paths[idx]
            name = os.path.basename(mask_path)
            overNum = name.find(".npy")
            name = name[0:overNum]
            rgbName = name + ".png"

            npmask = np.load(mask_path)

            GtColor = np.zeros([npmask.shape[0],npmask.shape[1],3], dtype=np.uint8)
            for idx in range(npmask.shape[0]):
                for idy in range(npmask.shape[1]):
                    #坏疽(NET,non-enhancing tumor)(标签1) 红色
                    if npmask[idx, idy] == 1:
                        GtColor[idx, idy, 0] = 255
                        GtColor[idx, idy, 1] = 0
                        GtColor[idx, idy, 2] = 0
                    #浮肿区域(ED,peritumoral edema) (标签2) 绿色
                    elif npmask[idx, idy] == 2:
                        GtColor[idx, idy, 0] = 0
                        GtColor[idx, idy, 1] = 128
                        GtColor[idx, idy, 2] = 0
                    #增强肿瘤区域(ET,enhancing tumor)(标签4) 黄色
                    elif npmask[idx, idy] == 4:
                        GtColor[idx, idy, 0] = 255
                        GtColor[idx, idy, 1] = 255
                        GtColor[idx, idy, 2] = 0

            #imsave(val_gt_path + rgbName, GtColor)
            imageio.imwrite(val_gt_path + rgbName, GtColor)

        print("Done!")



    if val_args.mode == "Calculate":
        """
        计算各种指标:Dice、Sensitivity、PPV
        """
        wt_dices = []
        tc_dices = []
        et_dices = []
        wt_sensitivities = []
        tc_sensitivities = []
        et_sensitivities = []
        wt_ppvs = []
        tc_ppvs = []
        et_ppvs = []
        wt_Hausdorf = []
        tc_Hausdorf = []
        et_Hausdorf = []
        wt_specificity= []
        tc_specificity= []
        et_specificity = []
        wt_biou = []
        tc_biou = []
        et_biou = []
        
        wtMaskList = []
        tcMaskList = []
        etMaskList = []
        wtPbList = []
        tcPbList = []
        etPbList = []
        
        outpath = val_args.outpath + '/' + args.name

        gtpath = os.path.join(outpath, 'GT')

        maskls = list(Path(gtpath).glob("*.png"))

        maskPath = [str(path) for path in maskls]
        

        
        pbls = list(Path(outpath).glob("*.png"))
        pbPath = [str(path) for path in pbls]

        
        if len(maskPath) == 0:
            print("请先生成图片!")
            return

        for myi in tqdm(range(len(maskPath))):
            mask = imread(maskPath[myi])
            pb = imread(pbPath[myi])

            wtmaskregion = np.zeros([mask.shape[0], mask.shape[1]], dtype=np.float32)
            wtpbregion = np.zeros([mask.shape[0], mask.shape[1]], dtype=np.float32)

            tcmaskregion = np.zeros([mask.shape[0], mask.shape[1]], dtype=np.float32)
            tcpbregion = np.zeros([mask.shape[0], mask.shape[1]], dtype=np.float32)

            etmaskregion = np.zeros([mask.shape[0], mask.shape[1]], dtype=np.float32)
            etpbregion = np.zeros([mask.shape[0], mask.shape[1]], dtype=np.float32)

            for idx in range(mask.shape[0]):
                for idy in range(mask.shape[1]):
                    # 只要这个像素的任何一个通道有值,就代表这个像素不属于前景,即属于WT区域
                    if mask[idx, idy, :].any() != 0:
                        wtmaskregion[idx, idy] = 1
                    if pb[idx, idy, :].any() != 0:
                        wtpbregion[idx, idy] = 1
                    # 只要第一个通道是255,即可判断是TC区域,因为红色和黄色的第一个通道都是255,区别于绿色
                    if mask[idx, idy, 0] == 255:
                        tcmaskregion[idx, idy] = 1
                    if pb[idx, idy, 0] == 255:
                        tcpbregion[idx, idy] = 1
                    # 只要第二个通道是128,即可判断是ET区域
                    if mask[idx, idy, 1] == 128:
                        etmaskregion[idx, idy] = 1
                    if pb[idx, idy, 1] == 128:
                        etpbregion[idx, idy] = 1
            #开始计算WT
            # print(wtmaskregion)
            # print(wtpbregion)
            dice = dice_coef(wtpbregion,wtmaskregion)
            wt_dices.append(dice)
            ppv_n = ppv(wtpbregion, wtmaskregion)
            wt_ppvs.append(ppv_n)
            Hausdorff = hausdorff_distance(wtmaskregion, wtpbregion)
            wt_Hausdorf.append(Hausdorff)
            sensitivity_n = sensitivity(wtpbregion, wtmaskregion)
            wt_sensitivities.append(sensitivity_n)
            specificity_n = specificity(wtpbregion, wtmaskregion)
            wt_specificity.append(specificity_n)
            wt_biou.append(boundary_iou(wtmaskregion, wtpbregion))
            
            # 开始计算TC
            dice = dice_coef(tcpbregion, tcmaskregion)
            tc_dices.append(dice)
            ppv_n = ppv(tcpbregion, tcmaskregion)
            tc_ppvs.append(ppv_n)
            Hausdorff = hausdorff_distance(tcmaskregion, tcpbregion)
            tc_Hausdorf.append(Hausdorff)
            sensitivity_n = sensitivity(tcpbregion, tcmaskregion)
            tc_sensitivities.append(sensitivity_n)
            specificity_n = specificity(tcpbregion, tcmaskregion)
            tc_specificity.append(specificity_n)
            tc_biou.append(boundary_iou(tcmaskregion, tcpbregion))
            
            # 开始计算ET
            dice = dice_coef(etpbregion, etmaskregion)
            et_dices.append(dice)
            ppv_n = ppv(etpbregion, etmaskregion)
            et_ppvs.append(ppv_n)    
            Hausdorff = hausdorff_distance(etmaskregion, etpbregion)
            et_Hausdorf.append(Hausdorff)
            sensitivity_n = sensitivity(etpbregion, etmaskregion)
            et_sensitivities.append(sensitivity_n)
            specificity_n = specificity(etpbregion, etmaskregion)
            et_specificity.append(specificity_n)
            et_biou.append(boundary_iou(etmaskregion, etpbregion))
            
        print('WT Dice: %.4f' % np.mean(wt_dices))
        print('TC Dice: %.4f' % np.mean(tc_dices))
        print('ET Dice: %.4f' % np.mean(et_dices))
        print("=============")
        print('WT PPV: %.4f' % np.mean(wt_ppvs))
        print('TC PPV: %.4f' % np.mean(tc_ppvs))
        print('ET PPV: %.4f' % np.mean(et_ppvs))
        print("=============")
        print('WT sensitivity: %.4f' % np.mean(wt_sensitivities))
        print('TC sensitivity: %.4f' % np.mean(tc_sensitivities))
        print('ET sensitivity: %.4f' % np.mean(et_sensitivities))
        print("=============")
        print('WT specificity: %.4f' % np.mean(wt_specificity))
        print('TC specificity: %.4f' % np.mean(tc_specificity))
        print('ET specificity: %.4f' % np.mean(et_specificity))
        print("=============")
        print('WT Hausdorff: %.4f' % np.mean(wt_Hausdorf))
        print('TC Hausdorff: %.4f' % np.mean(tc_Hausdorf))
        print('ET Hausdorff: %.4f' % np.mean(et_Hausdorf))
        print("=============")
        print('WT Boundary IoU: %.4f' % np.mean(wt_biou))
        print('TC Boundary IoU: %.4f' % np.mean(tc_biou))
        print('ET Boundary IoU: %.4f' % np.mean(et_biou))


if __name__ == '__main__':
    main( )
