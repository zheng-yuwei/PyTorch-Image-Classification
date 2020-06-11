# -*- coding: utf-8 -*-
"""
File ohm_bce.py
@author: ZhengYuwei
基于在线困难样本挖掘的多标签二分类交叉熵损失 OHMBCELoss 
"""
import torch

from .basic_bce import MultiLabelBCELoss


class OHMBCELoss(MultiLabelBCELoss):
    """ 困难样本学习 """
    def __init__(self, args):
        super(OHMBCELoss, self).__init__()
        self.hard_ratio = args.hard_ratio

    def get_weights(self, predictions: torch.FloatTensor, targets: torch.FloatTensor):
        """ 多标签二分类交叉熵损失
        :param predictions: 预测的概率矩阵，(batch_size, label_num)
        :param targets: 解码后的多标签二分类label概率矩阵，(batch_size, label_num)
        """
        gradients = torch.abs(predictions - targets)
        threshold = gradients.flatten().sort()[0][int(targets.numel() * self.hard_ratio)]
        weights = (gradients > threshold).float()
        return weights
