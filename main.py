# -*- coding: utf-8 -*-
"""
File main.py

总入口 
"""
import os
import random
import datetime
import logging
import traceback
from functools import partial

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import LambdaLR
import torch.multiprocessing as mp
import torch.distributed as dist
import apex
from apex import amp

import models as my_models
from config import parser
import utils
import dataloader
import applications
import criterions
from optim.torchtools.optim import (
    AdamW, RAdam, RangerLars, Ralamb,
    Novograd, LookaheadAdam, Ranger
)


def main():
    """
    3种运行方式：
    1. 单CPU运行模式；
    2. 单GPU运行模式；
    3. 分布式运行模式：多机多卡 或 单机多卡。
    分布式优势：1.支持同步BN； 2.DDP每个训练有独立进程管理，训练速度更快，显存均衡；
    """
    args = parser.parse_args()
    # 根据训练机器和超参，选择运行方式
    num_gpus_available = torch.cuda.device_count()
    if num_gpus_available == 0:
        args.gpus = 0
    elif args.gpus > num_gpus_available:
        raise ValueError(f'--gpus(-g {args.gpus}) can not greater than available device({num_gpus_available})')

    # 根据每个节点的GPU数量调整world size
    args.world_size = args.gpus * args.nodes
    if not args.cuda or args.world_size == 0:
        # 1. cpu运行模式
        args.cuda = False
        args.gpus = 0
        args.distributed = False
    elif args.world_size == 1:
        # 2. 单GPU运行模式
        args.distributed = False
    elif args.world_size > 1:
        # 3. 分布式运行模式
        args.distributed = True
    else:
        raise ValueError(f'Check config parameters --nodes/-n={args.nodes} and --gpus/-g={args.gpus}!')

    if args.distributed and args.gpus > 1:
        # use torch.multiprocessing.spawn to launch distributed processes
        mp.spawn(main_worker, nprocs=args.gpus, args=(args,))
    else:
        # Simply call main_worker function
        main_worker(0, args)


def main_worker(gpu, args):
    """
    模型训练、测试、转JIT、蒸馏文件制作
    :param gpu: 运行的gpu id
    :param args: 运行超参
    """
    args.gpu = gpu
    utils.generate_logger(f"{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}-{gpu}.log")
    logging.info(f'args: {args}')

    # 可复现性
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        logging.warning('You have chosen to seed training. '
                        'This will turn on the CUDNN deterministic setting, '
                        'which can slow down your training considerably! '
                        'You may see unexpected behavior when restarting '
                        'from checkpoints.')

    if args.cuda:
        logging.info(f"Use GPU: {args.gpu} ~")
        if args.distributed:
            args.rank = args.rank * args.gpus + gpu
            dist.init_process_group(backend='nccl', init_method=args.init_method,
                                    world_size=args.world_size, rank=args.rank)
    else:
        logging.info(f"Use CPU ~")

    # 创建/加载模型，使用预训练模型时，需要自己先下载好放到 pretrained 文件夹下，以网络名词命名
    logging.info(f"=> creating model '{args.arch}'")
    model = my_models.get_model(args.arch, args.pretrained, num_classes=args.num_classes)

    # 重加载之前训练好的模型
    if args.resume:
        if os.path.isfile(args.resume):
            logging.info(f"=> loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume, map_location=torch.device('cpu'))
            acc = model.load_state_dict(checkpoint['state_dict'], strict=True)
            logging.info(f'missing keys of models: {acc.missing_keys}')
            del checkpoint
        else:
            raise Exception(f"No checkpoint found at '{args.resume}' to be resumed")

    # 模型信息
    image_height, image_width = args.image_size
    logging.info(f'Model {args.arch} input size: ({image_height}, {image_width})')
    utils.summary(size=(image_height, image_width), channel=3, model=model)

    # 模型转换：转为 torch.jit.script
    if args.jit:
        if not args.resume:
            raise Exception('Option --resume must specified!')
        applications.convert_to_jit(model, args=args)
        return

    if args.criterion == 'softmax':
        criterion = criterions.HybridCELoss(args=args)  # 混合策略多分类
    elif args.criterion == 'bce':
        criterion = criterions.HybridBCELoss(args=args)  # 混合策略多标签二分类
    else:
        raise NotImplementedError(f'Not loss function {args.criterion}')

    if args.cuda:
        if args.distributed and args.sync_bn:
            model = apex.parallel.convert_syncbn_model(model)
        torch.cuda.set_device(args.gpu)
        model.cuda(args.gpu)
        criterion = criterion.cuda(args.gpu)

    if args.knowledge in ('train', 'test', 'val'):
        torch.set_flush_denormal(True)
        distill_loader = dataloader.load(args, name=args.knowledge)
        applications.distill(distill_loader, model, criterion, args, is_confuse_matrix=True)
        return

    if args.make_curriculum in ('train', 'test', 'val'):
        torch.set_flush_denormal(True)
        curriculum_loader = dataloader.load(args, name=args.make_curriculum)
        applications.make_curriculum(curriculum_loader, model, criterion, args, is_confuse_matrix=True)
        return

    if args.visual_data in ('train', 'test', 'val'):
        torch.set_flush_denormal(True)
        test_loader = dataloader.load(args, name=args.visual_data)
        applications.Visualize.visualize(test_loader, model, args)
        return

    # 优化器
    opt_set = {
        'sgd': partial(torch.optim.SGD, momentum=args.momentum),
        'adam': torch.optim.Adam, 'adamw': AdamW,
        'radam': RAdam, 'ranger': Ranger, 'lookaheadadam': LookaheadAdam,
        'ralamb': Ralamb, 'rangerlars': RangerLars,
        'novograd': Novograd,
    }
    optimizer = opt_set[args.opt](model.parameters(), lr=args.lr)  # weight decay转移到train那里了
    # 随机均值平均优化器
    # from optim.swa import SWA
    # optimizer = SWA(optimizer, swa_start=10, swa_freq=5, swa_lr=0.05)

    # 混合精度训练
    if args.cuda:
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

    if args.distributed:
        model = apex.parallel.DistributedDataParallel(model)
    else:
        model = torch.nn.DataParallel(model)

    if args.train:
        train_loader = dataloader.load(args, 'train')
        val_loader = dataloader.load(args, 'val')
        scheduler = LambdaLR(optimizer,
                             lambda epoch: adjust_learning_rate(epoch, args=args))
        applications.train(train_loader, val_loader, model, criterion, optimizer, scheduler, args)
        args.evaluate = True

    if args.evaluate:
        torch.set_flush_denormal(True)
        test_loader = dataloader.load(args, name='test')
        acc, loss, paths_targets_preds_probs = applications.test(test_loader, model,
                                                                 criterion, args, is_confuse_matrix=True)
        logging.info(f'Evaluation: * Acc@1 {acc:.3f} and loss {loss:.3f}.')
        logging.info(f'Evaluation Result:\n')
        for path, target, pred, prob in paths_targets_preds_probs:
            logging.info(path + ' ' + str(target) + ' ' + str(pred) + ' ' + ','.join([f'{num:.2f}' for num in prob]))
        logging.info('Evaluation Over~')


def adjust_learning_rate(epoch, args):
    """ 根据warmup设置、迭代代数、设置的学习率，调整每一代的学习率
    :param epoch: 当前epoch数
    :param args: 使用warmup代数、学习率
    """
    # lr_rates = [0.1, 1., 10., 100., 1e-10]
    # epochs = [2, 4, 6, 8, 10]
    lr_ratios = np.array([0.1, 1., 0.1, 0.01, 0.001])
    epoch_step = (args.epochs - args.warmup) / 4.0
    epochs = np.array([args.warmup,
                       args.warmup + int(1.5 * epoch_step),
                       args.warmup + int(2.5 * epoch_step),
                       args.warmup + int(3.5 * epoch_step),
                       args.epochs])
    if args.lr_ratios is not None:
        lr_ratios = np.array(args.lr_ratios)
    if args.lr_steps is not None:
        epochs = np.array(args.lr_steps)

    for i, e in enumerate(epochs):
        if e > epoch:
            return lr_ratios[i]
        elif e == epoch:
            next_rate = lr_ratios[i]
            if len(lr_ratios) > i + 1:
                next_rate = lr_ratios[i + 1]
            logging.info(f'===== lr decay rate: {lr_ratios[i]} -> {next_rate} =====')

    return lr_ratios[-1]


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logging.error(traceback.format_exc())
        raise e
