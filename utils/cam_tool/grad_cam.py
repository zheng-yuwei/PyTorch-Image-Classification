# -*- coding: utf-8 -*-
"""
File grad_cam.py

Grad-CAM 可视化
"""
import numpy as np

from dataloader.my_dataloader import DataLoaderX


class GradCAM(object):
    """
    1: 网络不更新梯度，输入需要梯度更新
    2: 使用目标类别的得分做反向传播
    """

    def __init__(self, net, layer_name):
        self.net = net
        self.net.eval()
        self.layer_name = layer_name
        self.features = None
        self.gradients = None
        self.handlers = []
        self._register_hook()

    def _register_hook(self):
        handler = getattr(self.net, self.layer_name).register_forward_hook(self._get_features_hook)
        self.handlers.append(handler)
        handler = getattr(self.net, self.layer_name).register_backward_hook(self._get_grads_hook)
        self.handlers.append(handler)

    def _get_features_hook(self, module, input, output):
        self.features = output.cpu().data.numpy()  # [N,C,H,W]
        print("features shape:{}".format(self.features.shape))

    def _get_grads_hook(self, module, input_grad, output_grad):
        """
        :param input_grad: tuple, input_grad[0]: None
                                   input_grad[1]: weight
                                   input_grad[2]: bias
        :param output_grad:tuple,长度为1
        :return:
        """
        self.gradients = output_grad[0].cpu().data.numpy()  # [N,C,H,W]
        print("gradients shape:{}".format(self.gradients.shape))

    def remove_handlers(self):
        for handle in self.handlers:
            handle.remove()

    def __call__(self, images, args):
        """
        计算Grad-CAM图
        :param images: pytorch的tensor，[N,3,H,W]
        :param args: 超参，主要用cuda选项和预处理选项
        :return:
        """
        images = DataLoaderX.normalize(images, args)  # 图像标准化
        if args.cuda:
            images = images.cuda(args.gpu, non_blocking=True)
        # 前向得到特征图
        outputs = self.net(images)  # [N, num_classes]
        # 后向，得到梯度，计算每个特征图的加权
        weights = self._calc_grad(outputs)  # [N, num_classes, C]
        # grad-cam
        batch_cam_imgs = self._calc_grad_cam(weights)
        return outputs, batch_cam_imgs

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
            weight = np.mean(self.gradients, axis=(2, 3))  # [N, C]
            weights.append(weight)
        weights = np.stack(weights, axis=1)  # [N, num_classes, C]
        return weights

    def _calc_grad_cam(self, weights):
        """
        :param weights: 每个类别的梯度对每张特征图的加权权重，[N, num_classes, C]
        :return
        """
        bz, nc, h, w = self.features.shape  # [N,C,H,W]
        batch_cams = []  # [N, num_classes, H, W]
        for i in range(bz):
            cams = weights[i] @ self.features[i].reshape(nc, -1)
            cams = cams.reshape(-1, h, w)
            batch_cams.append(cams)
        batch_cams = np.array(batch_cams, dtype=np.float32)  # [N, num_classes, H, W]
        # batch_cams = np.maximum(batch_cams, 0)  # ReLU
        # 数值归一化
        batch_cams -= np.min(batch_cams, axis=(1, 2, 3), keepdims=True)
        batch_cams /= np.max(batch_cams, axis=(1, 2, 3), keepdims=True)
        batch_cam_imgs = np.uint8(255 * batch_cams)
        return batch_cam_imgs
