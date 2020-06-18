# -*- coding: utf-8 -*-
"""
File heatmap.py

绘制热力图 
"""
import cv2
import numpy as np


class HeatMapTool:

    @staticmethod
    def add_heat(image: np.ndarray, cams: np.ndarray) -> np.ndarray:
        """
        给图像加heatmap
        :param image: 单张图像，bgr uint8格式
        :param cams: 类激活图，灰度图，uint8格式
        :return: 原图 和 叠加了热力图图像 按宽方向拼接起来的图
        """
        h, w, nc = image.shape
        cam_images = [image]
        for cam in cams:
            heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
            heatmap = cv2.resize(heatmap, (w, h))
            cam_image = heatmap * 0.5 + image * 0.5
            cam_images.append(cam_image)
        return np.concatenate(cam_images, axis=1).astype(np.uint8)
