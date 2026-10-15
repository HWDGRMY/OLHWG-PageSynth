#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: layout_metrics.py
@description: 计算布局评估所需的 8 个二元几何特征（论文表2）。
"""

from typing import List

import numpy as np


def binary_geometric_features(gt_boxes: List[List[float]]) -> np.ndarray:
    """计算相邻字符之间的 8 个二元几何特征。

    Args:
        gt_boxes: 每个元素为 [height, width, center_y, delta_x] 的列表。

    Returns:
        形状为 (L-1, 8) 的特征矩阵。
    """
    l = len(gt_boxes)
    features = []
    for i in range(1, l):
        h_curr, w_curr, c_curr, d_curr = gt_boxes[i]
        h_prev, w_prev, c_prev, d_prev = gt_boxes[i - 1]

        f1 = c_curr - c_prev
        f2 = d_curr + w_prev / 2 + w_curr / 2
        f3 = c_curr - c_prev + h_curr / 2 - h_prev / 2
        f4 = c_curr - c_prev - h_curr / 2 + h_prev / 2
        f5 = d_curr + w_prev
        f6 = d_curr + w_curr
        f7 = h_curr / h_prev if h_prev != 0 else h_curr
        f8 = w_curr / w_prev if w_prev != 0 else w_curr

        features.append([f1, f2, f3, f4, f5, f6, f7, f8])
    return np.array(features, dtype=np.float32)