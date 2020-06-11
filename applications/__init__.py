# -*- coding: utf-8 -*-
"""
File __init__.py.py

应用：
train：训练
test：测试
Visualize：可视化
make_curriculum：课程学习时制作课程权重文件
distill：蒸馏时制作教师模型概率文件
convert_to_jit：模型转JIT
"""
from .train import train
from .test import test
from .visualize import Visualize
from .make_curriculum import make_curriculum
from .distillation import distill
from .convert import convert_to_jit
