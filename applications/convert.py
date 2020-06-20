# -*- coding: utf-8 -*-
"""
File convert.py

模型转换：转 torch.jit.script
"""
import argparse
import logging

import torch


def convert_to_jit(model: torch.nn.Module, args: argparse.Namespace):
    """
    将模型转为JIT格式，利于部署
    :param model: 待转格式模型
    :param args: 转模型超参
    """
    logging.info('Converting model ...')
    image_height, image_width = args.image_size

    model.eval()
    rand_image = torch.rand(1, 3, image_height, image_width)
    with torch.no_grad():
        with torch.autograd.profiler.profile(use_cuda=args.cuda) as prof:
            model(rand_image)
        logging.info(prof)
        torch_model = torch.jit.trace(model, (rand_image,))
        torch_model.save(f'checkpoints/jit_{args.arch}.pt')
    logging.info('Save with jit script mode over ~ ')
