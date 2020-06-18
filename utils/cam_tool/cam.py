# -*- coding: utf-8 -*-
"""
File cam.py

CAM可视化 
"""
import numpy as np
import torch

from dataloader.my_dataloader import DataLoaderX


class CAM:
    """
    Class Activation Mapping可视化
    """

    def __init__(self, model, module_name):
        """
        在模型的指定模块上注册 forward hook
        :param model: 模型
        :param module_name: 网络最后的卷积层模块的名称
        """
        self.model = model
        self.model.eval()
        self.features = None
        # 最后一层全连接层的权重矩阵，(num_classes, num_conv_channel)
        self.fc_weights = np.squeeze(list(model.parameters())[-2].data.cpu().numpy())
        # 获取最后的卷积层的前向输出
        self.hook = getattr(model, module_name).register_forward_hook(self._hook_fn)

    def _hook_fn(self, module, input, output):
        """
        hook函数，进行自定义操作
        :param module: 整个模块，若需要可对内部层进行细致的操控
        :param input: 模块输入
        :param output: 模块输出
        """
        self.features = output.detach().cpu().numpy()
        print("features shape:{}".format(output.size()))

    def remove(self):
        """ 移除hook """
        self.hook.remove()

    def __call__(self, images, args) -> (torch.tensor, np.ndarray):
        """
        计算CAM图
        :param images: pytorch的tensor，[N,3,H,W]
        :param args: 超参，主要用cuda选项和预处理选项
        :return: 模型预测输出 (N, num_classes)，cam图 (N, num_classes, H_conv, W_conv)
        """
        with torch.no_grad():
            images = DataLoaderX.normalize(images, args)  # 图像标准化
            if args.cuda:
                images = images.cuda(args.gpu, non_blocking=True)
            outputs = self.model(images)
            cams = self._calc_cam()
        return outputs, cams

    def _calc_cam(self) -> np.ndarray:
        """
        计算CAM，cam = sum_nc(w * features)
        :return batch中每个image的每个CAM，(N, num_classes, H, W)
        """
        bz, nc, h, w = self.features.shape  # batch的最后卷积层，(N, num_conv_channel, H, W)
        batch_cams = self.fc_weights @ self.features.transpose(1, 0, 2, 3).reshape(nc, -1)  # C(NHW)
        batch_cams = batch_cams.reshape(-1, bz, h, w).transpose(1, 0, 2, 3)  # NCHW
        # 各自image的标准化
        batch_cams -= np.min(batch_cams, axis=(1, 2, 3), keepdims=True)
        batch_cams /= np.max(batch_cams, axis=(1, 2, 3), keepdims=True)
        batch_cam_imgs = np.uint8(255 * batch_cams)
        return batch_cam_imgs
