# -*- coding: utf-8 -*-
"""
File visualize.py

使用CAM（class activaion mapping）进行可视化
Learning Deep Features for Discriminative Localization
ref: https://github.com/zhoubolei/CAM/blob/master/pytorch_CAM.py
"""
import os
import logging
import argparse

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from utils import HeatMapTool, CAM, GradCAM, GradCamPlusPlus


class Visualize:

    @staticmethod
    def visualize(data_loader: DataLoader, model: torch.nn.Module, args: argparse.Namespace):
        """
        获取最后一层卷积层激活函数后的激活输出，根据选取的可视化方法，可视化所有类别的激活图
        efficientnet的最后一层卷积层激活函数输出模块名为 _swish_fc，如果要可视化其他模型，这里需要做对应修改
        :param data_loader: 待可视化的数据集
        :param model: 待可视化模型
        :param args: 可视化超参
        """
        model.eval()
        cam_inferences = []
        if args.visual_method in ('cam', 'all'):
            cam_inferences.append(CAM(model, '_swish_fc'))
        if args.visual_method in ('grad-cam', 'all'):
            cam_inferences.append(GradCAM(model, '_swish_fc'))
        if args.visual_method in ('grad-cam++', 'all'):
            cam_inferences.append(GradCamPlusPlus(model, '_swish_fc'))
        if len(cam_inferences) == 0:
            raise NotImplementedError('--visual_method must be in (cam, grad-cam, grad-cam++)')
        for i, (images, labels, paths, _) in enumerate(data_loader):
            batch_images, batch_cams = [], []
            uint8_images = np.uint8(images.numpy().transpose((0, 2, 3, 1))[..., ::-1] * 255)
            for cam_inference in cam_inferences:
                outputs, cams = cam_inference(images, args)
                batch_cams.append(cams)
                batch_images.append(uint8_images)
            batch_cams = np.concatenate(batch_cams, axis=2)
            batch_images = np.concatenate(batch_images, axis=1)
            if args.cuda:
                labels = labels.cuda(args.gpu, non_blocking=True)

            match_results, labels, predictions = Visualize._assess(outputs, labels)
            for path, image, cams, is_match, label, pred in \
                    zip(paths, batch_images, batch_cams, match_results, labels, predictions):
                # 预测错误，则需要画出错误的热图和正确的热图
                logging.info(f'{is_match} path: {path},{data_loader.dataset.classes[label]},'
                             f'{data_loader.dataset.classes[pred]}')
                cv2.imwrite(f'visual_images/{os.path.basename(path)}',
                            HeatMapTool.add_heat(image, cams))
                # cv2.imshow(f'{path[10:-4]}-truth', HeatMapTool.add_heat(image, cams))
                # cv2.waitKey(0)

    @staticmethod
    def _assess(outputs: torch.Tensor, labels: torch.Tensor) -> (np.ndarray, np.ndarray, np.ndarray):
        """
        评估输出是否预测对标签
        :param outputs: 模型输出
        :param labels: 标签
        :return 是否预测准确，标签，预测结果
        """
        with torch.no_grad():
            _, preds = outputs.max(dim=1)
            if preds.dim() != labels.dim():
                _, labels = labels.max(dim=1)
            result = preds.eq(labels).detach().cpu().numpy()
        return result, labels.detach().cpu().numpy(), preds.detach().cpu().numpy()
