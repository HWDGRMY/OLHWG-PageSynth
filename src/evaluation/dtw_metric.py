#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: dtw_metric.py
@description: 归一化 DTW 距离计算。
"""

import math
import numpy as np
from fastdtw import fastdtw


def _euclidean_distance(p1, p2) -> float:
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def _total_euclidean_distance(points: np.ndarray) -> float:
    return sum(_euclidean_distance(points[i], points[i + 1]) for i in range(len(points) - 1))


def norm_dtw(seq_1: np.ndarray, seq_2: np.ndarray) -> float:
    """归一化 DTW 距离。

    Args:
        seq_1: (N, 2) 轨迹坐标。
        seq_2: (M, 2) 轨迹坐标。

    Returns:
        归一化后的 DTW 距离。
    """
    fast_d, _ = fastdtw(seq_1, seq_2)
    len2 = _total_euclidean_distance(seq_2)
    return fast_d / max(len2, 1e-6)