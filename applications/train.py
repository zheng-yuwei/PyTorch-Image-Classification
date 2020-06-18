# -*- coding: utf-8 -*-
"""
File train.py

模型训练脚本
"""
import time
import shutil
import logging

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.optimizer import Optimizer
import apex
from apex import amp

from utils import AverageMeter, ProgressMeter
from applications.test import test
from dataloader import MixUp
from dataloader import MultiScale
from dataloader.my_dataloader import DataLoaderX


mix_up = None
multi_scale = None
bn_gammas = None
net_weights = None  # 排除bias项的weight decay


def train(train_loader: DataLoader, val_loader: DataLoader, model: nn.Module,
          criterion: nn.Module, optimizer: Optimizer,
          scheduler: torch.optim.lr_scheduler._LRScheduler, args):
    """
    训练模型
    :param train_loader: 训练集
    :param val_loader: 验证集
    :param model: 模型
    :param criterion: 损失函数
    :param optimizer: 优化器
    :param args: 训练超参
    """
    writer = SummaryWriter(args.logdir)
    # writer.add_graph(model, (torch.rand(1, 3, args.image_size[0], args.image_size[1]),))
    global mix_up, multi_scale, bn_gammas, net_weights
    if mix_up is None:
        mix_up = MixUp(args)
    if args.multi_scale and multi_scale is None:
        multi_scale = MultiScale(args.image_size)
    if bn_gammas is None:
        bn_gammas = [m.weight for m in model.modules()
                     if isinstance(m, nn.BatchNorm2d) or
                     isinstance(m, nn.SyncBatchNorm) or
                     isinstance(m, apex.parallel.SyncBatchNorm)]

    if net_weights is None:
        net_weights = [param for name, param in model.named_parameters() if name[-4:] != 'bias']

    best_val_acc1 = 0
    learning_rate = 0
    for epoch in range(args.epochs):
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
        learning_rate = scheduler.get_last_lr()
        if isinstance(learning_rate, list):
            learning_rate = learning_rate[0]
        # 训练一个epoch，并在验证集上评估
        train_loss, train_acc1 = train_epoch(train_loader, model, criterion, optimizer, epoch, args)
        val_acc1, val_loss, _ = test(val_loader, model, criterion, args, is_confuse_matrix=False)
        scheduler.step()
        # 保存当前及最好的acc@1的checkpoint
        is_best = val_acc1 >= best_val_acc1
        best_val_acc1 = max(val_acc1, best_val_acc1)
        save_checkpoint({
            # 'epoch': epoch + 1,
            # 'arch': args.arch,
            'state_dict': model.module.state_dict(),
            # 'best_acc1': best_val_acc1,
            # 'optimizer': optimizer.state_dict(),
        }, is_best, args)

        all_bn_weight = []
        for gamma in bn_gammas:
            all_bn_weight.append(gamma.cpu().data.numpy())
        writer.add_histogram('BN gamma', np.concatenate(all_bn_weight, axis=0), epoch)
        # writer.add_scalars('Loss', {'Train': train_loss, 'Val': val_loss}, epoch)
        # writer.add_scalars('Accuracy', {'Train': train_acc1, 'Val': val_acc1}, epoch)
        writer.add_scalar('Train/Loss', train_loss, epoch)
        writer.add_scalar('Train/Accuracy', train_acc1, epoch)
        writer.add_scalar('Val/Loss', val_loss, epoch)
        writer.add_scalar('Val/Accuracy', val_acc1, epoch)
        writer.add_scalar('learning rate', learning_rate, epoch)
        writer.flush()
    writer.close()
    logging.info(f'Training Over with lr={learning_rate}~~')


def train_epoch(train_loader, model, criterion, optimizer, epoch, args):
    """
    训练模型一个epoch的数据
    :param train_loader: 训练集
    :param model: 模型
    :param criterion: 损失函数
    :param optimizer: 优化器
    :param epoch: 当前迭代次数
    :param args: 训练超参
    """
    global mix_up, multi_scale, bn_gammas, net_weights
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(len(train_loader), batch_time, data_time,
                             losses, top1, prefix=f"Epoch: [{epoch}]")

    # 训练模式
    model.train()
    end_time = time.time()
    for i, (images, targets, _, weights) in enumerate(train_loader):
        # 更新数据加载时间度量
        data_time.update(time.time() - end_time)
        # 只有训练集，才可能进行mixup和multi-scale数据增强
        images, targets1, targets2, mix_rate = mix_up(images, targets)
        if args.multi_scale:
            images = multi_scale(images)

        images = DataLoaderX.normalize(images, args)
        if args.cuda:
            images = images.cuda(args.gpu, non_blocking=True)
            targets1 = targets1.cuda(args.gpu, non_blocking=True)
            weights = weights.cuda(args.gpu, non_blocking=True)
            if targets2 is not None:
                targets2 = targets2.cuda(args.gpu, non_blocking=True)
                mix_rate = mix_rate.cuda(args.gpu, non_blocking=True)

        output = model(images)
        loss = criterion(output, targets1, weights)
        if targets2 is not None:
            loss = mix_rate * loss + (1.0 - mix_rate) * criterion(output, targets2, weights)
            if mix_rate < 0.5:
                targets1 = targets2

        optimizer.zero_grad()
        if args.cuda:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        # network slimming
        if args.sparsity:
            for gamma in bn_gammas:
                gamma.data.add_(-torch.sign(gamma.data),
                                alpha=args.slim * optimizer.param_groups[0]['lr'])
        # weight decay
        for param in net_weights:
            param.data.add_(-param.data,
                            alpha=args.weight_decay * optimizer.param_groups[0]['lr'])

        optimizer.step()

        # 更新度量
        acc1 = accuracy(output, targets1)
        losses.update(loss.detach().cpu().item(), images.size(0))
        top1.update(acc1.item(), images.size(0))
        # 更新一个batch训练时间度量
        batch_time.update(time.time() - end_time)
        end_time = time.time()

        if i % args.print_freq == 0:
            progress.print(i)
    return losses.avg, top1.avg


def accuracy(output, target):
    """
    计算准确率和预测结果
    :param output: 分类预测
    :param target: 分类标签
    """
    with torch.no_grad():
        _, pred = output.max(dim=1)
        if pred.dim() != target.dim():
            _, target = target.max(dim=1)
        acc = pred.eq(target).float().mean() * 100
        return acc


def save_checkpoint(state, is_best, args, filename='checkpoints/checkpoint_{}.pth'):
    """
    保存模型
    :param state: 模型状态
    :param is_best: 模型是否当前测试集准确率最高
    :param args: 训练超参
    :param filename: 保存的文件名
    """
    filename = filename.format(args.arch)
    if (args.gpus > 1) and (args.gpu != args.gpus - 1):
        # 同一台服务器上多卡训练时，只有最后一张卡保存模型（多卡同时保存到同一位置会混乱）
        return
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, f'checkpoints/model_best_{args.arch}.pth')
