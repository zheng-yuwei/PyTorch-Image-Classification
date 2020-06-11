# -*- coding: utf-8 -*-
"""
File __init__.py.py

图像预处理，数据加载等
"""
from .enhancement import (
    MyAugment,
    ImageNetPolicy, CIFAR10Policy, SVHNPolicy,
    Rescale,
    MultiScale,
    MixUp
)
from .my_dataloader import load, DataLoaderX
