# -*- coding: utf-8 -*-
"""
File utils.py

损失函数相关工具
decode_to_onehot: 将用index表示的多标签二分类label，转为onehot表示的概率矩阵
"""
import torch


def decode_to_onehot(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    将用index表示的多标签二分类label，转为onehot表示的概率矩阵
    :param labels: index表示的多标签二分类label，如 [0, 3, 4] 表示连续三个样本的标签为类别 0, 3, 4
    :param num_classes: 类别数，或者标签数
    :return: 一批 每个样本的标签都是onehot表示 组成的 概率矩阵
    """
    batch_size = labels.size(0)
    onehot_labels = labels.new_full((batch_size, num_classes), 0)
    onehot_labels = onehot_labels.scatter(1, labels.unsqueeze(1), 1).float()
    return onehot_labels
