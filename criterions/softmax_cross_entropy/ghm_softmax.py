# -*- coding: utf-8 -*-
"""
File ghm_softmax.py

GHM加权的多分类交叉熵损失 GHMCELoss
基于梯度密度的倒数对损失进行加权（密度越大，损失越小）的BCE损失
"""
import argparse

import torch

from criterions.softmax_cross_entropy.basic_softmax import CrossEntropyLoss


class GHMCELoss(CrossEntropyLoss):
    """ GHM Classification Loss.
    "Gradient Harmonized Single-stage Detector".
    https://arxiv.org/abs/1811.05181
    """
    def __init__(self, args: argparse.Namespace, bins: int = 30, momentum: float = 0.75):
        """
        GHM多分类损失函数
        :param args: 训练超参
        :param bins: Number of the unit regions for distribution calculation.
        :param momentum: The parameter for moving average.
        """
        super(GHMCELoss, self).__init__()
        self.bins = bins
        self.momentum = momentum
        self.edges = torch.arange(bins + 1).float()
        if args.cuda:
            self.edges = self.edges.cuda(args.gpu)
        self.edges /= bins
        self.edges[0] -= 1e-6
        self.edges[-1] += 1e-6

        if momentum > 0:
            self.acc_sum = torch.zeros(bins)
            if args.cuda:
                self.acc_sum = self.acc_sum.cuda(args.gpu)

    def get_weights(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        多分类交叉熵损失
        :param predictions: 预测的概率矩阵，(batch_size, label_num)
        :param targets: 解码后的多分类label概率矩阵，(batch_size, label_num)
        :return: 与predictions同维度的权重矩阵，(batch_size, label_num)
        """
        edges = self.edges
        mmt = self.momentum
        weights = torch.zeros_like(targets)

        # 计算梯度
        g = torch.abs(predictions - targets)
        total_items = targets.shape[0] * targets.shape[1]
        n = 0  # n valid bins
        for i in range(self.bins):
            inds = (g >= edges[i]) & (g < edges[i+1])
            num_in_bin = inds.sum().item()
            if num_in_bin > 0:
                if mmt > 0:
                    self.acc_sum[i] = mmt * self.acc_sum[i] + (1 - mmt) * num_in_bin
                    weights[inds] = total_items / self.acc_sum[i]
                else:
                    weights[inds] = total_items / num_in_bin
                n += 1

        if n > 0:
            weights = weights / n

        return weights


if __name__ == '__main__':
    import argparse
    inputs = torch.tensor([[-3.1781, -2.9444, -3.8918, -4.5951, 4.5951],
                           [-3.1781, -2.9444, 4.5951, -3.8918, -4.5951]])

    real_targets = torch.tensor([[0., 0., 0., 0., 1.],
                                 [0., 0., 1., 0., 0.]])
    temp_args = argparse.Namespace()
    temp_args.cuda = False
    ghm_loss = GHMCELoss(temp_args)

    loss_value = ghm_loss(inputs.sigmoid().detach(), real_targets, None)
    print(f'first loss: {loss_value}')
    loss_value = ghm_loss(inputs.sigmoid().detach(), real_targets, None)
    print(f'first loss: {loss_value}')
