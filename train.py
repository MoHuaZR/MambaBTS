# -*- coding: utf-8 -*-

import time
import os
import math
import argparse
from glob import glob
from collections import OrderedDict
import random
import warnings
from datetime import datetime

import numpy as np
from tqdm import tqdm

from sklearn.model_selection import train_test_split
import joblib
from skimage.io import imread

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

from metrics import dice_coef, batch_iou, mean_iou, iou_score
import losses 
from utils import str2bool, count_params
import pandas as pd
from nets import unet

arch_names = list(unet.__dict__.keys())
loss_names = list(losses.__dict__.keys())
print(loss_names)
loss_names.append('BCEWithLogitsLoss')

IMG_PATH = glob(r"/home/cavin/Experiment/ZR/EXP/Med3D/data/processed/2D/trainImage/*")
MASK_PATH = glob(r"/home/cavin/Experiment/ZR/EXP/Med3D/data/processed/2D/trainMask/*")

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--ppath', default='/home/cavin/Experiment/ZR/EXP/Med3D/models/mamba_models', type=str,
                        help='model raw path')
    parser.add_argument('--name', default=None,
                        help='model name: (default: arch+timestamp)')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='UDMamba',
                        choices=arch_names,
                        help='model architecture: ' +
                            ' | '.join(arch_names) +
                            ' (default: Unet)')
    parser.add_argument('--deepsupervision', default=False, type=str2bool)
    parser.add_argument('--dataset', default="ZR",
                        help='dataset name')
    parser.add_argument('--input-channels', default=4, type=int,
                        help='input channels')
    parser.add_argument('--resume_path', default= '', type=str,
                        help= 'Path for resume model.')
    parser.add_argument('--image-ext', default='png',
                        help='image file extension')
    parser.add_argument('--mask-ext', default='png',
                        help='mask file extension')
    parser.add_argument('--aug', default=False, type=str2bool)
    parser.add_argument('--loss', default='BCEDiceLoss',
                        choices=loss_names,
                        help='loss: ' +
                            ' | '.join(loss_names) +
                            ' (default: BCEDiceLoss)')
    parser.add_argument('--epochs', default=10000, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--early_stop', default=20, type=int,
                        metavar='N', help='early stopping (default: 20)')
    parser.add_argument('--batch_size', default=48, type=int,
                        metavar='N', help='mini-batch size (default: 16)')
    parser.add_argument('--optimizer', default='Adam',
                        choices=['Adam', 'SGD'],
                        help='loss: ' +
                            ' | '.join(['Adam', 'SGD']) +
                            ' (default: Adam)')
    parser.add_argument('--scheduler', default='ExponentialLR',
                        choices=['ExponentialLR', 'CosineAnnealingLR','StepLR','CyclicLR'],
                        help='loss: ' +
                            ' | '.join(['ExponentialLR', 'CosineAnnealingLR','StepLR','CyclicLR']) +
                            ' (default: ExponentialLR)')
    parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight-decay', default=1e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', default=False, type=str2bool,
                        help='nesterov')

    args = parser.parse_args()

    return args

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train(args, train_loader, model, criterion, optimizer, epoch, scheduler=None):
    losses = AverageMeter()
    ious = AverageMeter()

    model.train()

    for i, (input, target) in tqdm(enumerate(train_loader), total=len(train_loader)):
        input = input.cuda()
        target = target.cuda()
        optimizer.zero_grad()
        # compute output
        if args.deepsupervision:
            outputs = model(input)
            loss = 0
            for output in outputs:
                loss += criterion(output, target)
            loss /= len(outputs)
            iou = iou_score(outputs[-1], target)
        else:
            output = model(input)
            loss = criterion(output, target)
            iou = iou_score(output, target)

        losses.update(loss.item(), input.size(0))
        ious.update(iou, input.size(0))

        # compute gradient and do optimizing step
        
        loss.backward()
        optimizer.step()
        
    scheduler.step  ()
    log = OrderedDict([
        ('loss', losses.avg),
        ('iou', ious.avg),
        ('lr', scheduler.get_last_lr()[0])
    ])

    return log


def validate(args, val_loader, model, criterion):
    losses = AverageMeter()
    ious = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, (input, target) in tqdm(enumerate(val_loader), total=len(val_loader)):
            input = input.cuda()
            target = target.cuda()

            # compute output
            if args.deepsupervision:
                outputs = model(input)
                loss = 0
                for output in outputs:
                    loss += criterion(output, target)
                loss /= len(outputs)
                iou = iou_score(outputs[-1], target)
            else:
                output = model(input)
                loss = criterion(output, target)
                iou = iou_score(output, target)

            losses.update(loss.item(), input.size(0))
            ious.update(iou, input.size(0))

    log = OrderedDict([
        ('loss', losses.avg),
        ('iou', ious.avg),
    ])

    return log


def main():
    args = parse_args()
    #args.dataset = "datasets"
    
    if args.name is None:
        if args.deepsupervision:
            args.name = '%s_%s_wDS' %(args.dataset, args.arch)
        else:
            args.name = '%s_%s_woDS' %(args.dataset, args.arch)
    mopath = os.path.join(args.ppath, args.name)
    if not os.path.exists(mopath):
        os.makedirs(mopath)
    print('Config -----')
    for arg in vars(args):
        print('%s: %s' % (arg, getattr(args, arg)))
    print('------------')

    with open('%s/args.txt' % mopath, 'w') as f:
        for arg in vars(args):
            print('%s: %s' %(arg, getattr(args, arg)), file=f)

    joblib.dump(args, '%s/args.pkl' % mopath)

    # define loss function (criterion)
    if args.loss == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss().cuda()
    elif args.loss == 'BceDiceLoss':
        criterion = BceDiceLoss(wb=1, wd=1).cuda()
    else:
        criterion = losses.__dict__[args.loss]().cuda()

    cudnn.benchmark = True

    # Data loading code
    img_paths = IMG_PATH
    mask_paths = MASK_PATH

    train_img_paths, val_img_paths, train_mask_paths, val_mask_paths = \
        train_test_split(img_paths, mask_paths, test_size=0.2, random_state=41)
    print("train_num:%s"%str(len(train_img_paths)))
    print("val_num:%s"%str(len(val_img_paths)))


    # create model
    print("=> creating model %s" %args.arch)
    model = unet.__dict__[args.arch](args)
    
    if args.resume_path:
        if os.path.isfile(args.resume_path):
            print("=> loading checkpoint '{}'".format(args.resume_path))
            checkpoint = torch.load(args.resume_path)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
              .format(args.resume_path, checkpoint['epoch']))
    model = model.cuda()

    total_params = count_params(model)
    params_in_millions = round(total_params / 1e6, 2)
    print("Parameters num: {} M".format(params_in_millions))

    if args.optimizer == 'Adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    elif args.optimizer == 'SGD':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,
            momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov)
    if args.scheduler == 'ExponentialLR':
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    elif args.scheduler == 'CosineAnnealingLR':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.early_stop)
    elif args.scheduler == 'StepLR':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
    elif args.scheduler == 'CyclicLR':
        scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-4, max_lr=5e-4, step_size_up=500, step_size_down=500, mode='triangular', gamma=1.0, scale_fn=None, scale_mode='cycle', cycle_momentum=True, base_momentum=0.8, max_momentum=0.9, last_epoch=-1)

    train_dataset = Dataset(args, train_img_paths, train_mask_paths, args.aug)
    val_dataset = Dataset(args, val_img_paths, val_mask_paths)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        drop_last=False)

    log = pd.DataFrame(index=[], columns=[
        'epoch', 'lr', 'loss', 'iou', 'val_loss', 'val_iou'
    ])

    best_iou = 0
    trigger = 0
    for epoch in range(args.epochs):
        print('Epoch [%d/%d]' %(epoch, args.epochs))

        # train for one epoch
        train_log = train(args, train_loader, model, criterion, optimizer, epoch, scheduler)
        # evaluate on validation set
        val_log = validate(args, val_loader, model, criterion)

        print('loss %.4f - iou %.4f - val_loss %.4f - val_iou %.4f'
            %(train_log['loss'], train_log['iou'], val_log['loss'], val_log['iou']))

        tmp = pd.Series([
            epoch,
            train_log['lr'],
            train_log['loss'],
            train_log['iou'],
            val_log['loss'],
            val_log['iou'],
        ], index=['epoch', 'lr', 'loss', 'iou', 'val_loss', 'val_iou'])

        log = log._append(tmp, ignore_index=True)
        log.to_csv('%s/log.csv' % mopath, index=False)

        trigger += 1
        
        if val_log['iou'] > best_iou:
            torch.save(model.state_dict(), '%s/model.pth' % mopath)
            best_iou = val_log['iou']
            print("=> saved best model")
            trigger = 0

        # early stopping
        if not args.early_stop is None:
            if trigger >= args.early_stop:
                print("=> early stopping")
                break

        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()