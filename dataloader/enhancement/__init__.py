# -*- coding: utf-8 -*-
"""
File __init__.py.py
@author: ZhengYuwei
数据增强
"""
from .multi_scale import MultiScale
from .rescale import Rescale
from .autoaugment import ImageNetPolicy, CIFAR10Policy, SVHNPolicy
from .my_augment import MyAugment
from .mixup import MixUp
