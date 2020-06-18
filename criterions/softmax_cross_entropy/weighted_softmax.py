# -*- coding: utf-8 -*-
"""
File weighted_softmax.py

加权多分类交叉熵损失 WeightedCELoss，不同的误分类情况具有不同的损失权重
"""
import argparse

import torch

from .basic_softmax import CrossEntropyLoss
from config import CriterionConstant


class WeightedCELoss(CrossEntropyLoss):
    """ 借用偏序loss的思想 """
    def __init__(self, args: argparse.Namespace):
        super(WeightedCELoss, self).__init__()
        self.label_weights = torch.from_numpy(CriterionConstant.weights_for_ce)
        if args.cuda:
            self.label_weights = self.label_weights.cuda(args.gpu)

    def get_weights(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        多标签二分类交叉熵损失
        :param predictions: 预测的概率矩阵，(batch_size, label_num)
        :param targets: 解码后的多标签二分类label概率矩阵，(batch_size, label_num)
        :return: 与predictions同维度的权重矩阵，(batch_size, label_num)
        """
        return targets @ self.label_weights
