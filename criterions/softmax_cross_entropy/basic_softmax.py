# -*- coding: utf-8 -*-
"""
File basic_softmax.py

softmax交叉熵损失函数，分类index label和one hot label形式都支持 
"""
import torch
from torch import nn

from criterions.utils import decode_to_onehot


class CrossEntropyLoss(nn.Module):
    """ 多分类交叉熵损失 """
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """
        多分类交叉熵损失
        :param predictions: 预测的logit矩阵，(batch_size, label_num)
        :param targets: 多分类label，如果是非one-hot形状(batch_size,)，则需要先解码
        :param weights: 每个样本的权重
        :return: 损失值
        """
        if predictions.dim() != targets.dim():
            targets = decode_to_onehot(targets, predictions.size(-1))
        if weights is None:
            weights = self.identity_weights(predictions, targets)
        if weights.dim() != predictions.dim():
            weights = weights.repeat(predictions.size(-1), 1).T

        logged_x_pred = predictions.log_softmax(dim=1)
        weights *= self.get_weights(predictions.softmax(dim=1).detach(), targets)
        return -torch.sum(targets * logged_x_pred * weights, dim=1).sum()

    def get_weights(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        多标签二分类交叉熵损失的权重
        :param predictions: 预测的概率矩阵，(batch_size, label_num)
        :param targets: 解码后的多标签二分类label概率矩阵，(batch_size, label_num)
        :return: 与predictions同维度的权重矩阵，(batch_size, label_num)
        """
        return self.identity_weights(predictions, targets)

    @staticmethod
    def identity_weights(predictions: torch.Tensor, _: torch.Tensor) -> torch.Tensor:
        """
        多分类交叉熵损失的单位权重的dummy函数
        :param predictions: 预测的概率矩阵，(batch_size, label_num)
        :param _: 解码后的多标签二分类label概率矩阵，(batch_size, label_num)
        :return: 与predictions同维度的单位矩阵
        """
        return torch.ones_like(predictions)
