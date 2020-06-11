# -*- coding: utf-8 -*-
"""
File basic_bce.py

多标签二分类交叉熵损失函数
"""
import torch
from torch import nn
from torch.nn import functional as F

from criterions.utils import decode_to_onehot


class MultiLabelBCELoss(nn.Module):
    """多标签二分类交叉熵损失函数"""

    def __init__(self):
        super(MultiLabelBCELoss, self).__init__()

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """
        多标签二分类交叉熵损失函数
        :param predictions: 网络预测logit输出, (N, num_class)
        :param targets: 多标签二分类label，index形式(N,) 或 one-hot形式(N, num_class)
        :param weights: 每个样本的权重 (N,)
        :return: 损失值
        """
        if targets.dim() != predictions.dim():
            targets = decode_to_onehot(targets, predictions.size(-1))
        if weights is None:
            weights = self.identity_weights(predictions, targets)
        if weights.dim() != predictions.dim():
            weights = weights.repeat(predictions.size(-1), 1).T

        weights *= self.get_weights(predictions.sigmoid().detach(), targets)
        loss = F.binary_cross_entropy_with_logits(predictions, targets, weights, reduction='sum')
        return loss

    def get_weights(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        每一项多标签二分类交叉熵损失的权重
        :param predictions: 预测的概率矩阵（非logit形式，且detach了）， (N, num_class)
        :param targets: 解码为one-hot形式的多标签二分类label，(N, num_class)
        :return: 每一项损失的权重，(N, num_class)
        """
        return self.identity_weights(predictions, targets)

    @staticmethod
    def identity_weights(predictions: torch.Tensor, _: torch.Tensor) -> torch.Tensor:
        """
        返回与输入矩阵一样大小的、都为1的矩阵，作为单位损失权重
        :param predictions: 预测的概率矩阵， (N, num_class)
        :param _: 解码为one-hot形式的多标签二分类label，(N, num_class)
        :return: 单位损失权重，(N, num_class)
        """
        return torch.ones_like(predictions)
