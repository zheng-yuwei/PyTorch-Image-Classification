# -*- coding: utf-8 -*-
"""
File test.py

模型测试脚本
"""
import time
import typing
import logging
import argparse

import numpy as np
import torch
from torch import nn
import torch.distributed as dist
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report

from utils import meters, confusion_matrix
from dataloader.my_dataloader import DataLoaderX


def test(test_loader: DataLoader, model: nn.Module, criterion: nn.Module,
         args: argparse.Namespace, is_confuse_matrix: bool = True) -> \
        (float, float, typing.List[typing.Tuple[(str, int, int, str)]]):
    """
    验证集、测试集 模型评估
    :param test_loader: 测试集DataLoader对象
    :param model: 待测试模型
    :param criterion: 损失函数
    :param args: 测试参数
    :param is_confuse_matrix: 是否输出混淆矩阵
    """
    batch_time = meters.AverageMeter('Time', ':6.3f')
    losses = meters.AverageMeter('Loss', ':.4e')
    top1 = meters.AverageMeter('Acc@1', ':6.2f')
    progress = meters.ProgressMeter(
        len(test_loader), batch_time, losses, top1, prefix='Test: ')

    model.eval()
    # 图像路径，label，预测类别，概率向量 统计量
    all_paths, all_targets, all_preds, all_probs = list(), list(), list(), list()
    # 将路径映射为index，这样可以转为tensor，在分布式训练的时候才能使用gather
    path_2_index = {path: i for i, (path, _, _) in enumerate(test_loader.dataset.samples)}
    index_2_path = {i: path for path, i in path_2_index.items()}
    with torch.no_grad():
        end_time = time.time()
        for i, (images, targets, paths, weights) in enumerate(test_loader):
            images = DataLoaderX.normalize(images, args)
            if args.cuda:
                images = images.cuda(args.gpu, non_blocking=True)
                targets = targets.cuda(args.gpu, non_blocking=True)
                weights = weights.cuda(args.gpu, non_blocking=True)

            # 模型预测
            outputs = model(images)
            loss = criterion(outputs, targets, weights)

            # 统计准确率和损失函数
            acc1, pred, target = accuracy(outputs, targets)
            # 统计量
            losses.update(loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))
            batch_time.update(time.time() - end_time)
            end_time = time.time()

            # 收集结果
            all_paths.append(np.array([path_2_index[p] for p in paths], dtype=np.float32))
            all_targets.append(targets.cpu().numpy())
            all_preds.append(pred.cpu().numpy())
            all_probs.append(outputs.sigmoid().cpu().numpy())

            if i % args.print_freq == 0:
                progress.print(i)
    all_paths = np.concatenate(all_paths, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)
    all_probs = np.concatenate(all_probs, axis=0)

    if args.distributed:
        all_paths = gather_tensors_from_gpus(all_paths, args)
        all_targets = gather_tensors_from_gpus(all_targets, args)
        all_preds = gather_tensors_from_gpus(all_preds, args)
        all_probs = gather_tensors_from_gpus(all_probs, args)
        top1 = gather_meters_from_gpus(top1, args)
        losses = gather_meters_from_gpus(losses, args)
        batch_time = gather_meters_from_gpus(batch_time, args)

    all_paths = [index_2_path[i] for i in all_paths]
    logging.info(f'* Acc@1 {top1.avg:.3f} and loss {losses.avg:.3f} with time {batch_time.avg:.3f}')

    if is_confuse_matrix and ((args.gpus <= 1) or (args.gpu == args.gpus - 1)):
        # 同一台服务器上多卡训练时，只有最后一张卡保存混淆图
        report = classification_report(all_targets, all_preds, target_names=test_loader.dataset.classes)
        logging.info(f'分类结果报告:\n{report}',)

        confusion_matrix.plot_confusion_matrix(all_targets, all_preds, test_loader.dataset.classes,
                                               title=f'ConfusionMatrix', is_save=True)
    return top1.avg, losses.avg, list(zip(all_paths, all_targets, all_preds, all_probs))


def gather_tensors_from_gpus(array: np.ndarray, args) -> np.ndarray:
    """
    从多个GPU中汇总变量，并拼成新的tensor
    :param array: 待汇总变量
    :param args: 测试超参
    :return 汇总并拼接后的数组
    """
    with torch.no_grad():
        tensor = torch.from_numpy(array).cuda(args.gpu)
        multi_gpu_tensors = [torch.empty_like(tensor) for _ in range(args.world_size)]
        dist.all_gather(multi_gpu_tensors, tensor)
    return torch.cat(multi_gpu_tensors, dim=0).detach().cpu().numpy()


def gather_meters_from_gpus(meter: meters.AverageMeter, args) -> meters.AverageMeter:
    """
    从多个GPU中汇总统计指标，并返回更新后的指标
    :param meter: 待汇总统计指标
    :param args: 测试超参
    :return 汇总并更新后的指标
    """
    meter_array = np.array([[meter.avg, meter.count]], dtype=np.float32)
    all_meter = gather_tensors_from_gpus(meter_array, args)
    meter.reset()
    for item, count in all_meter:
        meter.update(item, count)
    return meter


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
        correct = pred.eq(target)
        acc = correct.float().mean() * 100
        return acc, pred, target
