# -*- coding: utf-8 -*-
"""
File my_dataloader.py

数据集加载
"""
import os
import glob
import typing
import logging
import argparse
from functools import partial

import cv2
import torch
import numpy as np
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator

from dataloader.enhancement import Rescale, MyAugment


def load(args: argparse.Namespace, name='train') -> DataLoader:
    """
    加载数据集
    :param args: 训练参数
    :param name: 加载的数据集类型，(train, test, val)
    :return: 返回DataLoader对象
    """
    names = ('train', 'test', 'val')
    assert name in names, f'Name of Data Set must be in {names}!'
    data_dir = os.path.join(args.data, name)

    data_set = MyDatasetFolder(data_dir, args)
    logging.info(f'{name} labels: {data_set.class_to_idx}.')

    train_sampler = None
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(data_set)

    loader = DataLoaderX(data_set, batch_size=args.batch_size, shuffle=(name == 'train' and not args.distributed),
                         num_workers=args.workers, pin_memory=True,
                         collate_fn=DataLoaderX.collate_fn, sampler=train_sampler, drop_last=args.distributed)
    return loader


class DataLoaderX(DataLoader):
    """ 使用prefetch_generator包提供数据预加载功能，同时也自定义batch收集功能 """
    IMAGE_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape((3, 1, 1))
    IMAGE_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape((3, 1, 1))

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

    @staticmethod
    def normalize(images: torch.FloatTensor, args: argparse.Namespace) -> torch.FloatTensor:
        """
        图像标准化
        :param images: 一批图像，NCHW
        :param args: 超参
        :return: 一批标准化后的图像, NCHW
        """
        if args.advprop:
            images = images * 2.0 - 1.0
        else:
            images -= DataLoaderX.IMAGE_MEAN
            images /= DataLoaderX.IMAGE_STD
        return images

    @staticmethod
    def collate_fn(batch: typing.List[typing.Tuple[torch.FloatTensor, typing.Union[int, np.ndarray], str, float]]) \
            -> (torch.FloatTensor, torch.FloatTensor, list, torch.FloatTensor):
        """
        将图像、标签、路径、样本权重的列表，整理为批次形式
        :param batch: （图像、标签、路径、样本权重）元组列表
        :return: 图像批次tensor N3HW，标签批次tensor N 或 (N,num_classes)，路径批次list，样本权重批次tensor N
        """
        image_sequence = []
        target_sequence = []
        path_sequence = []
        weight_sequence = []
        for image, target, path, weight in batch:
            image_sequence.append(image)
            target_sequence.append(target)
            path_sequence.append(path)
            weight_sequence.append(weight)

        # Tensor的列表堆成一个整体Tensor，作为一个批次数据
        image_sequence = torch.utils.data.dataloader.default_collate(image_sequence)
        target_sequence = torch.utils.data.dataloader.default_collate(target_sequence)
        path_sequence = torch.utils.data.dataloader.default_collate(path_sequence)
        weight_sequence = np.array(weight_sequence, dtype=np.float32)
        weight_sequence = torch.utils.data.dataloader.default_collate(weight_sequence)

        return image_sequence, target_sequence, path_sequence, weight_sequence


class MyDatasetFolder(datasets.DatasetFolder):

    def __init__(self, root: str, args: argparse.Namespace):
        """
        自定义Dataset类，返回 图像，标签，路径，样本权重
        :param root: 数据集的根目录
        :param args: 超参设定
        """
        super(MyDatasetFolder, self).__init__(root, self._get_loader(root, args),
                                              datasets.folder.IMG_EXTENSIONS,
                                              transform=None, target_transform=None, is_valid_file=None)
        if args.num_classes != len(self.classes):
            error_info = f'args.num_classes is {args.num_classes}, ' \
                         f'mismatch data set {self.classes}'
            logging.error(error_info)
            raise Exception(error_info)
        self.imgs = self.samples
        # 训练集且开启蒸馏，则使用 教师概率文件中的教师目标
        if args.distill and os.path.basename(root) == 'train':
            self._distill_targets()
        # 训练集且开启课程学习，则使用课程文件给不同样本分配不同的损失权重
        if args.curriculum_learning and os.path.basename(root) == 'train':
            self._curriculum_targets()
        else:
            for index, (path, target) in enumerate(self.samples):
                self.samples[index] = (path, target, 1.0)

    def __getitem__(self, index: int) -> (torch.FloatTensor, typing.Union[int, np.ndarray], str, float):
        """
        获取第index个训练样本，包含：图像，标签，路径，样本权重
        :param index: 样本下标
        :return: torch.float32的(3, H, W)数组，int或float32 np.ndarray类别数标签，str路径，float样本权重
        """
        path, target, weight = self.samples[index]
        sample = self.loader(path)
        return sample, target, path, weight

    def _distill_targets(self):
        """
        解析教师模型输出的（多个）训练集概率文件，将原来样本的target修正为概率文件中对应的target
        samples: (path, target: int) -> (path, target: float32 np.ndarray)
        """
        # 0. 查找概率文件
        file_dir = os.path.dirname(self.root)
        file_paths = glob.glob(f'{file_dir}/distill*.txt')
        if not file_paths:
            logging.warning('Distillation Mode has been activated, '
                            'but can not find teacher target file (like: distill*.txt)')
            raise Exception('Not Distillation File (like: distill*.txt)')

        # 1. 文件解析，统计样本的多个教师模型概率 (sum(概率向量)，教师个数）
        path_target_map = {path: (None, 0) for path, _ in self.samples}
        for file_path in file_paths:
            with open(file_path, 'r') as teacher_file:
                for line in teacher_file:
                    line = line.strip().split(',')
                    path = line[0]
                    # 忽略教师文件中多余的样本标签
                    if path not in path_target_map:
                        continue
                    target = np.array([float(num) for num in line[1:]], dtype=np.float32)
                    old_target, count = path_target_map[path]
                    if old_target is None:
                        old_target = target
                    else:
                        old_target += target
                    path_target_map[path] = (old_target, count + 1)

        # 2. 多教师概率文件预测结果融合
        num_classes = len(self.classes)
        for index, (path, target) in enumerate(self.samples):
            if path_target_map[path][0] is None:
                self.samples[index] = (path, np.eye(num_classes, dtype=np.float32)[target])
            else:
                teacher_targets, count = path_target_map[path]
                self.samples[index] = (path, teacher_targets / count)

        logging.info(f'Distill knowledge from file {file_paths} !!!')

    def _curriculum_targets(self):
        """
        从课程文件中解析对应样本在该课程下的样本权重，计算loss时使用
        samples: (path, target) -> (path, target, weight: float)
        """
        file_path = os.path.join(os.path.dirname(self.root), 'curriculum.txt')
        if not file_path:
            logging.warning('Curriculum Learning Mode has been activated, '
                            'but can not find curriculum file (curriculum.txt)')
            raise Exception('Not Curriculum File (curriculum.txt)')

        # 0. 解析课程文件，统计样本权重
        path_weight_map = {path: 0.0 for path, _ in self.samples}
        with open(file_path, 'r') as curriculum_file:
            for line in curriculum_file:
                line = line.strip().split(',')
                path = line[0]
                if path not in path_weight_map:
                    continue
                weight = float(line[1])
                path_weight_map[path] = weight

        for index, (path, target) in enumerate(self.samples):
            self.samples[index] = (path, target, path_weight_map[path])

        logging.info(f'Get Curriculum Weight from file {file_path} !!!')

    @staticmethod
    def _get_loader(root: str, args: argparse.Namespace) -> typing.Callable[[str], torch.FloatTensor]:
        """
        获取图像加载及预处理器
        :param root: 数据集根目录
        :param args: 超参设置
        :return: 图像加载及预处理器，输入图像路径，得到图像的torch.FloatTensor （3, H, W）
        """
        logging.info(f'Using image size: {args.image_size}')

        transforms_list = [
            cv2.imread,
            Rescale(args.image_size),
            # 由于预训练模型是PIL加载训练的，所以这一步需要把数据转为PIL的rgb格式
            partial(cv2.cvtColor, code=cv2.COLOR_BGR2RGB),
            transforms.ToTensor(),
        ]

        # 训练集时开启数据增强才有效
        if args.aug and os.path.basename(root) == 'train':
            logging.info('=================== Load Data with Augmentation ! ===================')
            transforms_list.insert(1, MyAugment())

        return transforms.Compose(transforms_list)


if __name__ == '__main__':
    my_args = argparse.Namespace(data='data', num_classes=3, batch_size=3, workers=0,
                                 image_size=[400, 224], advprop=False, aug=True,
                                 distill=False, curriculum_learning=False, distributed=False)

    data_loader = load(my_args, name='train')
    for temp_samples, temp_targets, temp_paths, temp_weights in data_loader:
        print(temp_paths)
        temp_samples = temp_samples.numpy().transpose(0, 2, 3, 1)[..., ::-1]
        for temp_sample in temp_samples:
            cv2.imshow('a', temp_sample)
            cv2.waitKey(0)
