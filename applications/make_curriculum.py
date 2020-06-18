# -*- coding: utf-8 -*-
"""
File make_curriculum.py

使用模型进行推理，获取课程文件，用于后续的课程学习
"""
import os
import logging
import argparse

from torch import nn
from torch.utils.data import DataLoader

from .test import test


def make_curriculum(curriculum_loader: DataLoader, model: nn.Module, criterion: nn.Module,
                    args: argparse.Namespace, is_confuse_matrix: bool = True):
    """
    获取课程学习时，不同难易样本具有不同损失权重的课程文件
    :param curriculum_loader: 评估的数据集
    :param model: 制作课程的模型
    :param criterion: 评估的指标/损失函数
    :param args: 超参
    :param is_confuse_matrix: 是否对评估的数据集输出混淆矩阵
    """
    _, _, paths_targets_preds_probs = test(curriculum_loader, model, criterion, args, is_confuse_matrix)

    if len(args.curriculum_thresholds) != len(args.curriculum_weights):
        raise ValueError(f'课程数不确定：课程阈值({args.curriculum_thresholds})与'
                         f'课程权重({args.curriculum_weights})的长度必须相等！')

    if args.curriculum_thresholds[-1] > 0.0:
        logging.warning(f'课程阈值({args.curriculum_thresholds})中没有指定最小阈值（0.0），'
                        f'可能部分样本不会被添加到该学习的课程之中')
    curriculums = []
    # 逐样本划分课程
    for path, target, _, prob in paths_targets_preds_probs:
        for i, threshold in enumerate(args.curriculum_thresholds):
            if prob[target] > threshold:
                curriculums.append(f'{path},{args.curriculum_weights[i]}\n')
                break

    distill_file_path = os.path.join(args.data, 'curriculum.txt')
    with open(distill_file_path, 'w+') as curriculum_file:
        curriculum_file.writelines(curriculums)

    logging.info('Finish generating curriculum file for curriculum learning!')
