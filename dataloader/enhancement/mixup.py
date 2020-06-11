# -*- coding: utf-8 -*-
"""
File mixup.py

mixup数据增强，应用在整个批次上
"""
import typing
import argparse

import numpy as np
import torch


class MixUp:

    def __init__(self, args: argparse.Namespace):
        """
        数据增强 mixup，ref: https://github.com/facebookresearch/mixup-cifar10
        :param args: 超参
        """
        self.is_mixup = args.mixup
        self.mixup_ratio = args.mixup_ratio
        self.mixup_alpha = args.mixup_alpha
        self.num_classes = args.num_classes

    def __call__(self, inputs: torch.FloatTensor, targets: typing.Union[torch.IntTensor, torch.FloatTensor]) \
            -> (torch.FloatTensor, typing.Union[torch.IntTensor, torch.FloatTensor],
                typing.Optional[torch.FloatTensor], float):
        """
        对图像和标签进行线性插值混合
        :param inputs: 图像pytorch数组，NCHW
        :param targets: 标签pytorch数组，(N,) 或 (N, num_classes)
        :return: with or w/o mixup的 图像，标签1，标签2（没有mixup就为None），混合比例
        """
        if not self.is_mixup or np.random.rand() > self.mixup_ratio:
            return inputs, targets, None, 1.0

        # 确保标签不是类别index，否则转为one-hot编码， (N, num_classes)
        batch_size, num_channel, image_height, image_width = inputs.shape
        if targets.dim() == 1:
            targets = targets.unsqueeze(1)
            y_onehot = torch.zeros((batch_size, self.num_classes), dtype=torch.float32)
            targets = y_onehot.scatter_(1, targets, 1)
        # 制作mixup的数据
        rp2 = torch.randperm(batch_size)
        inputs1 = inputs
        targets1 = targets
        inputs2 = inputs[rp2]
        targets2 = targets[rp2]
        # mix images
        mix_rate = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        mix_rate = torch.tensor(mix_rate).float()
        inputs_shuffle = mix_rate * inputs1 + (1 - mix_rate) * inputs2

        return inputs_shuffle, targets1, targets2, mix_rate
