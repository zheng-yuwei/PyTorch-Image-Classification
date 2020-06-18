# -*- coding: utf-8 -*-
"""
File grad_cam_plus.py

Grad-CAM++ 可视化
"""
import numpy as np

from .grad_cam import GradCAM


class GradCamPlusPlus(GradCAM):
    def __init__(self, net, layer_name):
        super(GradCamPlusPlus, self).__init__(net, layer_name)

    def _calc_grad(self, output):
        """
        :param output: 模型输出，[N, num_classes]
        :return 梯度图均值
        """
        weights = []  # [N, num_classes]
        for index in range(output.size(1)):
            self.net.zero_grad()
            targets = output[:, index].sum()
            targets.backward(retain_graph=True)

            gradients = np.maximum(self.gradients, 0.)  # ReLU, [N,C,H,W]
            # indicates = np.where(gradients > 0, 1., 0.)  # 示性函数
            norm_factor = np.sum(gradients, axis=(2, 3), keepdims=True)  # 归一化
            # alpha = indicates / (norm_factor + 1e-7)
            weight = np.sum(gradients / (norm_factor + 1e-7), axis=(2, 3))  # [N, C]
            weights.append(weight)
        weights = np.stack(weights, axis=1)  # [N, num_classes, C]
        return weights
