# -*- coding: utf-8 -*-
"""
File distillation.py

使用模型进行推理，获取概率文件，用于后续的模型蒸馏
"""
import os
import logging
import argparse

import numpy as np
from torch import nn
from torch.utils.data import DataLoader

from .test import test


def distill(distill_loader: DataLoader, model: nn.Module, criterion: nn.Module,
            args: argparse.Namespace, is_confuse_matrix: bool = True):
    """
    获取模型蒸馏时，教师模型的评估的概率文件
    :param distill_loader: 评估的数据集
    :param model: 教师模型
    :param criterion: 评估的指标/损失函数
    :param args: 超参
    :param is_confuse_matrix: 是否对评估的数据集输出混淆矩阵
    """
    _, _, paths_targets_preds_probs = test(distill_loader, model, criterion, args, is_confuse_matrix)

    knowledges = []
    original_labels = []
    for path, target, _, prob in paths_targets_preds_probs:
        prob = ','.join([f'{num:.2f}' for num in prob])
        knowledges.append(f'{path},{prob}\n')

        label = np.eye(args.num_classes, dtype=np.float32)[target]
        label = ','.join([str(num) for num in label])
        original_labels.append(f"{path},{label}\n")

    distill_file_path = os.path.join(args.data, 'distill.txt')
    with open(distill_file_path, 'w+') as knowledge_file:
        knowledge_file.writelines(knowledges)

    distill_file_path = os.path.join(args.data, 'label.txt')
    with open(distill_file_path, 'w+') as label_file:
        label_file.writelines(original_labels)

    logging.info('Finish generating knowledge file for model distillation!')
