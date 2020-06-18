# -*- coding: utf-8 -*-
"""
File ohm_softmax.py

加入OHM的softmax交叉熵损失函数，分类index label和one hot label形式都支持 
"""
import argparse

import torch

from .basic_softmax import CrossEntropyLoss


class OHMCELoss(CrossEntropyLoss):
    """ 困难样本学习 """
    def __init__(self, args: argparse.Namespace):
        super(OHMCELoss, self).__init__()
        self.hard_ratio = args.hard_ratio

    def get_weights(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        多分类交叉熵损失
        :param predictions: 预测的概率矩阵，(batch_size, label_num)
        :param targets: 解码后的多分类label概率矩阵，(batch_size, label_num)
        :return: 与predictions同维度的权重矩阵，(batch_size, label_num)
        """
        gradients, _ = torch.abs(predictions - targets).max(dim=1)
        ohm_positions = gradients.sort()[1][:int(gradients.numel() * self.hard_ratio)]
        weights = torch.zeros_like(predictions)
        weights[ohm_positions] = 1
        return weights
