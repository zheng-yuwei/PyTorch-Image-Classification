# -*- coding: utf-8 -*-
"""
File weight_bce.py
@author: ZhengYuwei
加权多标签二分类交叉熵损失 WeightedCELoss，不同的误分类情况具有不同的损失权重 
"""
import torch

from .basic_bce import MultiLabelBCELoss
from config import CriterionConstant


class WeightedBCELoss(MultiLabelBCELoss):
    """ 借用偏序loss的思想 """
    def __init__(self, args):
        super(WeightedBCELoss, self).__init__()
        self.label_weights = torch.from_numpy(CriterionConstant.weights_for_bce)
        if args.cuda:
            self.label_weights = self.label_weights.cuda(args.gpu)

    def get_weights(self, predictions: torch.FloatTensor, targets: torch.FloatTensor):
        """ 多标签二分类交叉熵损失
        :param predictions: 预测的概率矩阵，(batch_size, label_num)
        :param targets: 解码后的多标签二分类label概率矩阵，(batch_size, label_num)
        :return 每个label的损失的权重，(batch_size, label_num)
        """
        return targets @ self.label_weights
