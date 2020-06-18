# -*- coding: utf-8 -*-
"""
File my_augment.py

自定义数据增强，应用在单张图像上
"""
import logging

import numpy as np
import matplotlib.pyplot as plt
import imgaug as ia
from imgaug import augmenters as iaa


class MyAugment:

    def __init__(self):
        self.seq = iaa.Sequential(
            [
                iaa.Fliplr(0.5),
                iaa.Sometimes(0.5, iaa.Crop(percent=(0, 0.1))),

                iaa.Sometimes(0.5, iaa.Affine(
                    rotate=(-20, 20),  # 旋转±20度
                    # shear=(-16, 16),   # 剪切变换±16度，矩形变平行四边形
                    # order=[0, 1],  # 使用最近邻插值 或 双线性插值
                    cval=0,  # 填充值
                    mode=ia.ALL  # 定义填充图像外区域的方法
                )),

                # 使用0~3个方法进行图像增强
                iaa.SomeOf((0, 3),
                           [
                               iaa.Sometimes(0.8, iaa.OneOf([
                                   iaa.GaussianBlur((0, 2.0)),  # 高斯模糊
                                   iaa.AverageBlur(k=(1, 5)),  # 平均模糊，磨砂
                               ])),

                               # 要么运动，要么美颜
                               iaa.Sometimes(0.8, iaa.OneOf([
                                   iaa.MotionBlur(k=(3, 11)),  # 运动模糊
                                   iaa.BilateralBlur(d=(1, 5),
                                                     sigma_color=(10, 250),
                                                     sigma_space=(10, 250)),  # 双边滤波，美颜
                               ])),

                               # 模仿雪花
                               iaa.Sometimes(0.8, iaa.OneOf([
                                   iaa.SaltAndPepper(p=(0., 0.03)),
                                   iaa.AdditiveGaussianNoise(loc=0, scale=(0., 0.05 * 255), per_channel=False)
                               ])),

                               # 对比度
                               iaa.Sometimes(0.8, iaa.LinearContrast((0.6, 1.4), per_channel=0.5)),

                               # 锐化
                               iaa.Sometimes(0.8, iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5))),

                               # 整体亮度
                               iaa.Sometimes(0.8, iaa.OneOf([
                                   # 加性调整
                                   iaa.AddToBrightness((-30, 30)),
                                   # 线性调整
                                   iaa.MultiplyBrightness((0.5, 1.5)),
                                   # 加性 & 线性
                                   iaa.MultiplyAndAddToBrightness(mul=(0.5, 1.5), add=(-30, 30)),
                                ])),

                               # 饱和度
                               iaa.Sometimes(0.8, iaa.OneOf([
                                   iaa.AddToSaturation((-75, 75)),
                                   iaa.MultiplySaturation((0., 3.)),
                               ])),

                               # 色相
                               iaa.Sometimes(0.8, iaa.OneOf([
                                   iaa.AddToHue((-255, 255)),
                                   iaa.MultiplyHue((-3.0, 3.0)),
                               ])),

                               # 云雾
                               # iaa.Sometimes(0.3, iaa.Clouds()),

                               # 卡通化
                               # iaa.Sometimes(0.01, iaa.Cartoon()),
                           ],
                           random_order=True
                           )
            ],
            random_order=True
        )

    def __call__(self, img: np.ndarray) -> np.ndarray:
        """
        对cv2读取的单张BGR图像进行图像增强
        :param img: cv2读取的bgr格式图像， (h, w, 3)
        :return: 增强后的图像， (h, w, 3)
        """
        image_aug = self.seq.augment_image(img)
        return image_aug

    def __repr__(self):
        return 'Self-defined Augment Policy'


if __name__ == '__main__':
    import argparse

    import cv2
    import torchvision
    from torchvision import transforms
    from torch.utils.data import DataLoader

    from dataloader.enhancement.multi_scale import MultiScale
    from dataloader.enhancement.rescale import Rescale
    from dataloader.enhancement.mixup import MixUp

    image_size = (400, 224)
    scale = MultiScale(image_size)
    mixup = MixUp(argparse.Namespace(mixup=True, mixup_ratio=1.0, mixup_alpha=1.0, num_classes=3))

    def show_images(dataset, is_multi_scale=True, is_mixup=True, col=4):
        loader = DataLoader(dataset, batch_size=col, shuffle=False,
                            num_workers=0, pin_memory=False)
        for images, labels in loader:
            if is_multi_scale:
                images = scale(images)
            if is_mixup:
                images, _, _, _ = mixup(images, labels)
            images = images.permute(0, 2, 3, 1).numpy()[..., ::-1]
            logging.info(f'size: {images[0].shape}')
            images = np.hstack(images)

            plt.imshow(images)
            plt.axis('off')
            plt.show()

    data_set = torchvision.datasets.ImageFolder('data/train', loader=cv2.imread,
                                                transform=transforms.Compose([
                                                    MyAugment(),
                                                    Rescale(image_size),
                                                    transforms.ToTensor(),
                                                ]))
    show_images(data_set)
