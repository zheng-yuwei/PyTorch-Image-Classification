# -*- coding: utf-8 -*-
"""
File hybrid_softmax.py

混合策略的多分类交叉熵损失 
"""
import argparse

import torch

from .basic_softmax import CrossEntropyLoss
from .threshold_softmax import ThresholdCELoss
from .weighted_softmax import WeightedCELoss
from .ohm_softmax import OHMCELoss
from .ghm_softmax import GHMCELoss


class HybridCELoss(CrossEntropyLoss):
    """ 自定义的多标签二分类损失函数，包含weighted loss，threshold loss，ghm loss,ohm loss等功能 """

    def __init__(self, args: argparse.Namespace):
        super(HybridCELoss, self).__init__()
        self.threshold_func = self.identity_weights
        self.weighted_func = self.identity_weights
        self.ohm_func = self.identity_weights
        self.ghm_func = self.identity_weights

        if args.threshold_loss:
            self.threshold_func = ThresholdCELoss(args).get_weights
        if args.weighted_loss:
            self.weighted_func = WeightedCELoss(args).get_weights
        if args.ohm_loss:
            self.ohm_func = OHMCELoss(args).get_weights
        if args.ghm_loss:
            self.ghm_func = GHMCELoss(args).get_weights

    def get_weights(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        多标签二分类交叉熵损失
        :param predictions: 预测的概率矩阵，(batch_size, label_num)
        :param targets: 解码后的多分类label概率矩阵，(batch_size, label_num)
        :return: 与predictions同维度的权重矩阵，(batch_size, label_num)
        """
        weights = self.threshold_func(predictions, targets)
        weights *= self.weighted_func(predictions, targets)
        weights *= self.ohm_func(predictions, targets)
        weights *= self.ghm_func(predictions, targets)
        return weights
