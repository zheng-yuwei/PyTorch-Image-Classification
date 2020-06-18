# -*- coding: utf-8 -*-
"""
File threshold_bce.py

带阈值控制的多标签二分类交叉熵损失 ThresholdCELoss
非目标类需要大于指定阈值才计算loss，目标类需要小于指定阈值才计算loss，
与制作软标签、hard-example mining有类似思想
"""
import argparse

import torch

from .basic_bce import MultiLabelBCELoss
from config import CriterionConstant


class ThresholdBCELoss(MultiLabelBCELoss):
    """ 借用软标签和困难样本的思想 """
    def __init__(self, args: argparse.Namespace):
        super(ThresholdBCELoss, self).__init__()
        self.low_threshold = torch.from_numpy(CriterionConstant.low_threshold_for_bce)
        self.up_threshold = torch.from_numpy(CriterionConstant.up_threshold_for_bce)
        if args.cuda:
            self.low_threshold = self.low_threshold.cuda(args.gpu)
            self.up_threshold = self.up_threshold.cuda(args.gpu)

    def get_weights(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        多标签二分类交叉熵损失
        :param predictions: 预测的概率矩阵，(batch_size, label_num)
        :param targets: 解码后的多标签二分类label概率矩阵，(batch_size, label_num)
        :return: 每一项损失的权重，(N, num_class)
        """
        low_threshold = targets @ self.low_threshold
        up_threshold = targets @ self.up_threshold
        weights = ((predictions > low_threshold) & (predictions < up_threshold)).float()
        return weights
