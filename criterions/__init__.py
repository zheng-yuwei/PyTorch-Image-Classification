# -*- coding: utf-8 -*-
"""
File __init__.py.py
@author: ZhengYuwei
损失函数 
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
