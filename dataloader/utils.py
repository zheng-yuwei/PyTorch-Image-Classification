# -*- coding: utf-8 -*-
"""
File utils.py

工具函数
"""


def get_rescale_size(src_h: int, src_w: int, target_h: int, target_w: int) -> \
        ((int, int), (int, int, int, int)):
    """
    按长边等比例缩放，短边pad 0
    :param src_h: 源尺寸高
    :param src_w: 源尺寸宽
    :param target_h: 目标尺寸高
    :param target_w: 目标尺寸宽
    :return: （缩放后高，缩放后宽），（左边需要pad的宽度，右边需要pad的宽度，上边需要pad的宽度，下边需要pad的宽度）
    """
    # 等比例缩放
    scale = max(src_h / target_h, src_w / target_w)
    new_h, new_w = int(src_h / scale), int(src_w / scale)
    # padding
    left_more_pad, top_more_pad = 0, 0
    if new_w % 2 != 0:
        left_more_pad = 1
    if new_h % 2 != 0:
        top_more_pad = 1
    left = right = (target_w - new_w) // 2
    top = bottom = (target_h - new_h) // 2
    left += left_more_pad
    top += top_more_pad
    return (new_h, new_w), (left, right, top, bottom)
