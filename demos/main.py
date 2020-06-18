# -*- coding: utf-8 -*-
"""
File main.py

图像识别分类器demo 
"""
import typing
from functools import partial

import cv2
import numpy as np
import torch
from torchvision import transforms


class Classifier:
    """ 部署的分类器 """
    MODEL_WEIGHT_PATH = '../checkpoints/jit_efficientnet_b0.pt'
    IMAGE_SIZE = (400, 224)

    def __init__(self):
        """ 初始化 模型、预处理器 """
        torch.set_num_threads(1)
        torch.set_flush_denormal(True)
        self.model = torch.jit.load(self.MODEL_WEIGHT_PATH)
        self.model.eval()
        self.preprocess = transforms.Compose([
            Rescale(self.IMAGE_SIZE),
            partial(cv2.cvtColor, code=cv2.COLOR_BGR2RGB),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def recognize(self, image: np.ndarray) -> np.ndarray:
        """
        图像识别
        :param image: opencv bgr格式的numpy数组
        :return 概率最大的类别结果，每一类的概率
        """
        image = self.preprocess(image)

        self.model.eval()
        with torch.no_grad():
            output = self.model(image.unsqueeze(0))[0]
        probabilities = output.sigmoid().detach().numpy()
        return probabilities


class Rescale:
    """ 将样本中图片按指定尺寸等比例缩放 """

    def __init__(self, output_size: typing.Union[int, tuple, list]):
        """
        将图像等比例伸缩到指定尺寸，空余部分pad 0
        :param output_size: 指定的等比例伸缩后的尺寸
        """
        assert isinstance(output_size, (int, tuple, list))
        if isinstance(output_size, int):
            output_size = (output_size, output_size)
        self.output_size = output_size

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        对cv2读取的单张BGR图像进行图像等比例伸缩，空余部分pad 0
        :param image: cv2读取的bgr格式图像， (h, w, 3)
        :return: 等比例伸缩后的图像， (h, w, 3)
        """
        h, w = image.shape[:2]
        target_h, target_w = self.output_size[0], self.output_size[1]
        (new_h, new_w), (left, right, top, bottom) = self.get_rescale_size(h, w, target_h, target_w)

        # 等比例缩放
        image = cv2.resize(image, (new_w, new_h))
        # padding
        image = cv2.copyMakeBorder(image, top, bottom, left, right,
                                   cv2.BORDER_CONSTANT, value=[0, 0, 0])
        return image

    @staticmethod
    def get_rescale_size(src_h: int, src_w: int, target_h: int, target_w: int) -> \
            ((int, int), (int, int, int, int)):
        """
        按长边等比例缩放，短边pad 0
        :param src_h: 源尺寸高
        :param src_w: 源尺寸宽
        :param target_h: 目标尺寸高
        :param target_w: 目标尺寸宽
        :return: （缩放后高，缩放后宽），（左边需要pad的宽度，右边需要pad的宽度，上边需要pad的宽度，下边需要pad的宽度）
        """
        # 等比例缩放
        scale = max(src_h / target_h, src_w / target_w)
        new_h, new_w = int(src_h / scale), int(src_w / scale)
        # padding
        left_more_pad, top_more_pad = 0, 0
        if new_w % 2 != 0:
            left_more_pad = 1
        if new_h % 2 != 0:
            top_more_pad = 1
        left = right = (target_w - new_w) // 2
        top = bottom = (target_h - new_h) // 2
        left += left_more_pad
        top += top_more_pad
        return (new_h, new_w), (left, right, top, bottom)


if __name__ == '__main__':
    import os
    recognizer = Classifier()
    root_dir = 'test_images'
    for image_name in os.listdir(root_dir):
        image_path = os.path.join(root_dir, image_name)
        image = cv2.imread(image_path)
        result = recognizer.recognize(image)
        print(f'{image_name}: {result}')
