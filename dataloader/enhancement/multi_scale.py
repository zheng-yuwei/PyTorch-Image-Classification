# -*- coding: utf-8 -*-
"""
File multi_scale.py

将图像重新等比例伸缩到指定尺寸随机产生的随机尺寸，用于后续多尺度训练，应用在整个批次上
"""
import typing
import logging

import numpy as np
import torch
from torch import nn
from scipy.stats import truncnorm

from dataloader.utils import get_rescale_size


class MultiScale:

    def __init__(self, output_size: typing.Union[int, tuple, list], h_w_ratio: float = 1.8):
        """
        将图像伸缩到某一尺寸，尺寸是根据指定尺寸的正态分布随机生成的
        :param output_size: 指定的基线尺寸， int 或 tuple，int则表示宽，长则根据h_w_ratio而计算
                            如果是tuple，则代表 (H, W)，h_w_ratio依此计算
        :param h_w_ratio: 长:宽
        """
        assert isinstance(output_size, (int, tuple, list))
        if isinstance(output_size, (tuple, list)):
            h_w_ratio = 1.0 * output_size[0] / output_size[1]
            output_size = output_size[1]
        logging.info(f'>>>>>>>>>>>> MultiScale Mode, Use H:W={h_w_ratio}, base width={output_size}')
        self.h_w_ratio = h_w_ratio

        # 2倍标准差 = -/+15% 截断范围 的正态分布采用图像宽度
        range_ratio = 0.15
        norm_delta = 2.0
        mean = output_size
        sigma = range_ratio * mean / norm_delta
        self.generator = truncnorm(-norm_delta, norm_delta, loc=mean, scale=sigma)
        # plt.hist(self.generator.rvs(1000))

    def __call__(self, images: torch.FloatTensor) -> torch.Tensor:
        """
        对一批图像进行等比例随机缩放
        :param images: 一批图像, N3HW
        :return: 等比例随机缩放后的一批图像
        """
        h, w = images.shape[2:4]
        # 随机生成rescale的目标尺寸
        target_w = int(np.round(self.generator.rvs()))
        target_h = int(np.round(target_w * self.h_w_ratio))

        # resize & padding
        (new_h, new_w), (left, right, top, bottom) = get_rescale_size(h, w, target_h, target_w)
        images = nn.functional.interpolate(images, size=(new_h, new_w),
                                           mode='bilinear', align_corners=False)
        images = nn.functional.pad(images, [left, right, top, bottom])

        return images
