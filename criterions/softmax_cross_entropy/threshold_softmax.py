# -*- coding: utf-8 -*-
"""
File threshold_softmax.py

带阈值控制的多分类交叉熵损失 ThresholdCELoss
非目标类需要大于指定阈值才计算loss，目标类需要小于指定阈值才计算loss，
与制作软标签、hard-example mining有类似思想
"""
import argparse

import torch

from .basic_softmax import CrossEntropyLoss
from config import CriterionConstant


class ThresholdCELoss(CrossEntropyLoss):
    """ 借用软标签和困难样本的思想 """
    def __init__(self, args: argparse.Namespace):
        super(ThresholdCELoss, self).__init__()
        self.low_threshold = torch.from_numpy(CriterionConstant.low_threshold_for_ce)
        self.up_threshold = torch.from_numpy(CriterionConstant.up_threshold_for_ce)
        if args.cuda:
            self.low_threshold = self.low_threshold.cuda(args.gpu)
            self.up_threshold = self.up_threshold.cuda(args.gpu)

    def get_weights(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        多分类交叉熵损失
        :param predictions: 预测的概率矩阵，(batch_size, label_num)
        :param targets: 解码后的多分类label概率矩阵，(batch_size, label_num)
        :return: 与predictions同维度的权重矩阵，(batch_size, label_num)
        """
        low_threshold = targets @ self.low_threshold
        up_threshold = targets @ self.up_threshold
        weights = ((predictions > low_threshold) & (predictions < up_threshold)).float()
        return weights
