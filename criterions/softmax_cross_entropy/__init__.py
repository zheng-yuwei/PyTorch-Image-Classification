# -*- coding: utf-8 -*-
"""
File __init__.py.py

softmax交叉熵损失函数及其变体 
"""
from .basic_softmax import CrossEntropyLoss
from .ghm_softmax import GHMCELoss
from .hybrid_softmax import HybridCELoss
from .ohm_softmax import OHMCELoss
from .threshold_softmax import ThresholdCELoss
from .weighted_softmax import WeightedCELoss
