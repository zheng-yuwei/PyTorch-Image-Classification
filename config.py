# -*- coding: utf-8 -*-
"""
File config.py
@author: ZhengYuwei
配置文件 
"""
import argparse
import numpy as np


parser = argparse.ArgumentParser(description='PyTorch Classification Model Training')

parser.add_argument('--data', default='data/', metavar='DIR', help='数据集路径')
parser.add_argument('-a', '--arch', metavar='ARCH', default='efficientnet-b0',
                    help='模型结构，默认：efficientnet-b0')
parser.add_argument('--image_size', default=[400, 224], type=int, nargs='+', dest='image_size',
                    help='模型输入尺寸[H, W]，默认：[400, 224]')
parser.add_argument('--num_classes', default=6, type=int, help='类别数，默认：6')
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='数据加载进程数，默认：16')
parser.add_argument('-b', '--batch_size', default=256, type=int, metavar='N',
                    help='训练batch size大小，默认：256')

# 分布式训练相关
parser.add_argument('--seed', default=1234, type=int,
                    help='训练或测试时，使用随机种子保证结果的可复现，默认不使用')
parser.add_argument('--sync_bn', default=False, action='store_true',
                    help='BN同步，默认使用')
parser.add_argument('--cuda', default=True, dest='cuda', action='store_true',
                    help='是否使用cuda进行模型推理，默认 True，会根据实际机器情况调整')
parser.add_argument('-n', '--nodes', default=1, type=int, help='分布式训练的节点数')
parser.add_argument('-g', '--gpus', default=2, type=int,
                    help='每个节点使用的GPU数量，可通过设置环境变量（CUDA_VISIBLE_DEVICES=1）限制使用哪些/单个GPU')
parser.add_argument('--rank', default=-1, type=int, help='分布式训练的当前节点的序号')
parser.add_argument('--init_method', default='tcp://11.6.127.208:10006', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--logdir', default='logs', type=str, metavar='PATH',
                    help='Tensorboard日志目录，默认 logs')

# 训练过程参数设置
parser.add_argument('--train', default=False, dest='train', action='store_true',
                    help='是否训练，默认：False')
parser.add_argument('--epochs', default=85, type=int, metavar='N',
                    help='训练epoch数，默认：85')
parser.add_argument('--opt', default='adam', type=str, help='优化器，默认：adam')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float, metavar='LR',
                    help='初始学习率，默认：1e-4', dest='lr')
parser.add_argument('--lr_ratios', '--lr_ratios', default=None, type=float, nargs='+',
                    help='初始学习率每step的变化率，如 [1., 0.1] 则是一开始使用初始学习率，后续衰减 * 0.1，'
                         '默认：[0.1, 1., 0.1, 0.01, 0.001]', dest='lr_ratios')
parser.add_argument('--lr_steps', '--lr_steps', default=None, type=float, nargs='+',
                    help='初始学习率每次衰减的epoch数，如[10, 20]表示在10 epoch衰减，20应该是结束epoch，'
                         '默认：将总epoch分为5段', dest='lr_steps')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='学习率动量')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float, metavar='W',
                    help='网络权重衰减正则项，默认: 1e-4', dest='weight_decay')
parser.add_argument('--warmup', default=5, type=int, metavar='W', help='warm-up迭代数')
parser.add_argument('-p', '--print-freq', default=50, type=int, metavar='N',
                    help='训练过程中的信息打印，每隔多少个batch打印一次，默认: 50')
parser.add_argument('--pretrained', default=False, dest='pretrained', action='store_true',
                    help='是否使用预训练模型，默认不使用')
parser.add_argument('--advprop', default=False, action='store_true',
                    help='使用advprop的预训练模型，默认否，主要针对EfficientNet系列')

# 网络参数设置
parser.add_argument('--criterion', default='softmax', type=str,
                    help='使用的损失函数，默认 softmax，可选 bce')
parser.add_argument('--weighted_loss', default=False, dest='weighted_loss', action='store_true',
                    help='损失函数是否使用加权策略，默认否')
parser.add_argument('--threshold_loss', default=False, dest='threshold_loss', action='store_true',
                    help='损失函数是否使用阈值策略，默认否')
parser.add_argument('--ghm_loss', default=False, dest='ghm_loss', action='store_true',
                    help='损失函数是否使用GHM策略，默认否')
parser.add_argument('--ohm_loss', default=False, dest='ohm_loss', action='store_true',
                    help='损失函数是否使用OHM策略，默认否')
parser.add_argument('--hard_ratio', default=0.7, dest='hard_ratio', type=float,
                    help='OHM损失函数中困难样本的比例，默认 0.7')

# 额外的训练技巧参数设置
parser.add_argument('--mixup', default=False, dest='mixup', action='store_true',
                    help='使用mix-up对训练数据进行数据增强，默认 False')
parser.add_argument('--mixup_ratio', default=0.5, dest='mixup_ratio', type=float,
                    help='开启mix-up对训练数据进行数据增强时，使用增强的概率，默认 0.5')
parser.add_argument('--mixup_alpha', default=1.1, dest='mixup_alpha', type=float,
                    help='mix-up时两张图像混合的beta分布的参数，默认 1.1')
parser.add_argument('--aug', default=False, dest='aug', action='store_true',
                    help='进行数据增强，默认 False')
parser.add_argument('--multi_scale', default=False, dest='multi_scale', action='store_true',
                    help='多尺度训练，默认 False')

parser.add_argument('--sparsity', default=False, dest='sparsity', action='store_true',
                    help='是否使用network slimming训练稀疏网络，默认 False')
parser.add_argument('--slim', default=1e-4, type=float, dest='slim',
                    help='network slimming中BN gamma的权重衰减系数，默认 1e-4)')

# 其他策略的参数设置
parser.add_argument('-e', '--evaluate', dest='evaluate', default=False, action='store_true',
                    help='在测试集上评估模型')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='重加载已训练好的模型 (默认: none)')
parser.add_argument('--jit', dest='jit', default=False, action='store_true',
                    help='将模型转为jit格式！')

parser.add_argument('--knowledge', dest='knowledge', default=None, type=str,
                    choices=[None, 'train', 'test', 'val'],
                    help='指定数据集，使用教师模型（配合resume选型指定）对该数据集进行预测，获取概率文件（知识)')
parser.add_argument('--distill', default=False, dest='distill', action='store_true',
                    help='模型蒸馏（需要教师模型输出的概率文件)，默认 False')

parser.add_argument('--visual_data', dest='visual_data', default=None, type=str,
                    choices=[None, 'train', 'test', 'val'],
                    help='指定数据集，对模型进行可视化 TIPs:也可查看数据增强方式后的可视化效果，但仅限于train集')
parser.add_argument('--visual_method', dest='visual_method', default='cam', type=str,
                    choices=['all', 'cam', 'grad-cam', 'grad-cam++'], help='模型进行可视化的方法')

parser.add_argument('--make_curriculum', dest='make_curriculum', default=None, type=str,
                    choices=[None, 'train', 'test', 'val'],
                    help='指定数据集，制作课程学习中，不同样本损失权重的课程文件')
parser.add_argument('--curriculum_thresholds', dest='curriculum_thresholds', default=None,
                    type=float, nargs='+',
                    help='样本大于阈值列表（由大到小排序）中相应阈值，会被分配到同一课程中，制作在同一课程文件中')
parser.add_argument('--curriculum_weights', dest='curriculum_weights', default=None,
                    type=float, nargs='+',
                    help='与curriculum_thresholds对应，指定对应课程中样本的权重')
parser.add_argument('--curriculum_learning', dest='curriculum_learning',
                    default=False, action='store_true',
                    help='是否进行课程学习，默认 False')


class CriterionConstant:
    # weighted_bce中损失函数不同label间误分的loss权重，每一行表示该label误分为其他label的损失权重
    weights_for_bce = np.array([
        [1., 1., 1., 1., 1., 0.5],
        [1., 1., 0.5, 2., 2., 1.],  # 第二类与第三类相近，误分可以接受，但与第4、5类误分很严重
        [1., 0.5, 1., 2., 2., 1.],
        [1., 2., 2., 5., 1., 1.],  # 5 表示自身损失的加权，因为这一类比较重要
        [1., 2., 2., 1., 1., 1.],
        [0.5, 1., 1., 1., 1., 1.]
    ], dtype=np.float32)
    weights_for_ce = weights_for_bce
    # threshold_bce中损失函数不同label的概率阈值控制
    # 每一行表示该label的阈值下界，低于该阈值则不计算loss
    low_threshold_for_bce = np.array([
        [-0.1, 0.05, 0.05, 0.05, 0.05, 0.05],
        [0.05, -0.1, 0.05, 0.05, 0.05, 0.05],
        [0.05, 0.05, -0.1, 0.05, 0.05, 0.05],
        [0.05, 0.05, 0.05, -0.1, 0.05, 0.05],
        [0.05, 0.05, 0.05, 0.05, -0.1, 0.05],
        [0.05, 0.05, 0.05, 0.05, 0.05, -0.1]
    ], dtype=np.float32)
    low_threshold_for_ce = low_threshold_for_bce
    # 每一行表示该label的阈值下界，高该阈值则不计算loss
    up_threshold_for_bce = np.array([
        [0.95, 1.1, 1.1, 1.1, 1.1, 1.1],
        [1.1, 0.95, 1.1, 1.1, 1.1, 1.1],
        [1.1, 1.1, 0.95, 1.1, 1.1, 1.1],
        [1.1, 1.1, 1.1, 0.95, 1.1, 1.1],
        [1.1, 1.1, 1.1, 1.1, 0.95, 1.1],
        [1.1, 1.1, 1.1, 1.1, 1.1, 0.95],
    ], dtype=np.float32)
    up_threshold_for_ce = up_threshold_for_bce
