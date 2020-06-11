# -*- coding: utf-8 -*-
"""
File __init__.py.py

数据增强
Rescale：等比例缩放，应用在单张图像上
MyAugment：自定义的数据增强，应用在单张图像上
autoaugment：autoaugment，应用在单张图像上
MixUp：图像mixup增强，应用在整个批次上
MultiScale：多尺度训练，应用在整个批次上
"""
from .rescale import Rescale
from .my_augment import MyAugment
from .autoaugment import ImageNetPolicy, CIFAR10Policy, SVHNPolicy
from .mixup import MixUp
from .multi_scale import MultiScale
