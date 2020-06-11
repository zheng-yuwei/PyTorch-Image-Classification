# -*- coding: utf-8 -*-
"""
File __init__.py.py

损失函数
softmax_cross_entropy: 多分类交叉熵损失函数及其变种
binary_cross_entropy： 多标签二分类损失函数及其变种
"""
from .softmax_cross_entropy import (
    CrossEntropyLoss,
    ThresholdCELoss,
    WeightedCELoss,
    GHMCELoss,
    OHMCELoss,
    HybridCELoss
)
from .binary_cross_entropy import (
    MultiLabelBCELoss,
    ThresholdBCELoss,
    WeightedBCELoss,
    GHMBCELoss,
    OHMBCELoss,
    HybridBCELoss,
)
