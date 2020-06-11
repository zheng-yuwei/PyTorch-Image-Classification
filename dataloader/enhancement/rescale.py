# -*- coding: utf-8 -*-
"""
File rescale.py

将图像等比例伸缩到指定尺寸，空余部分pad 0，应用在单张图像上
"""
import typing

import cv2
import numpy as np

from dataloader.utils import get_rescale_size


class Rescale:

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
        (new_h, new_w), (left, right, top, bottom) = get_rescale_size(h, w, target_h, target_w)

        # 等比例缩放
        image = cv2.resize(image, (new_w, new_h))
        # padding
        image = cv2.copyMakeBorder(image, top, bottom, left, right,
                                   cv2.BORDER_CONSTANT, value=[0, 0, 0])
        return image
