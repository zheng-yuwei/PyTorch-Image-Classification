# -*- coding: utf-8 -*-
""" 使用graphviz进行网络可视化 """
import torch
from torchviz import make_dot

import models as my_models


if __name__ == '__main__':
    model = my_models.get_model('mobilenetv3_large')
    width = 224
    vis_graph = make_dot(model(torch.randn((1, 3, width, width))),
                         params=dict(model.named_parameters()))
    vis_graph.view()
