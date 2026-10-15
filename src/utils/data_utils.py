#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: data_utils.py
@description: 轨迹降采样、去padding、坐标归一化等工具。
"""

from typing import List, Tuple

import numpy as np
import torch


def clear_traj(traj: np.ndarray) -> np.ndarray:
    """去除轨迹末尾的 padding（state == -1）。"""
    l = len(traj)
    for i in range(1, l + 1):
        if traj[l - i, 2] == 1:
            return traj[:l - i + 1]
    return traj


def rdp(points: List[List[float]], epsilon: float) -> List[List[float]]:
    """Ramer-Douglas-Peucker 轨迹简化算法。"""
    if len(points) < 3:
        return points
    dmax, index = 0, 0
    start, end = points[0], points[-1]
    for i in range(1, len(points) - 1):
        d = _perpendicular_distance(points[i], start, end)
        if d > dmax:
            dmax, index = d, i
    if dmax > epsilon:
        r1 = rdp(points[:index + 1], epsilon)
        r2 = rdp(points[index:], epsilon)
        return r1[:-1] + r2
    return [start, end]


def _perpendicular_distance(point, start_point, end_point) -> float:
    x, y, p = point
    x1, y1, _ = start_point
    x2, y2, _ = end_point
    if p == -1:
        return float('inf')
    if x2 == x1:
        return abs(x - x1)
    slope = (y2 - y1) / (x2 - x1)
    intercept = y1 - slope * x1
    return abs(slope * x - y + intercept) / np.sqrt(slope ** 2 + 1)


def down_sample(traj: np.ndarray, epsilon: float = 0.02) -> np.ndarray:
    """对轨迹进行降采样（RDP 算法包装）。"""
    simplified = rdp(traj.tolist(), epsilon)
    return np.array(simplified, dtype=np.float32)


def center_and_normalize(traj: np.ndarray) -> np.ndarray:
    """将轨迹重心平移到原点，并把高度缩放到 1。"""
    sequence = traj[:, :2]
    centroid = np.mean(sequence, axis=0)
    centered = sequence - centroid
    height = np.max(centered[:, 1]) - np.min(centered[:, 1])
    if height > 0:
        centered = centered / height
    traj[:, :2] = centered
    return traj


def pad_tensor(line: torch.Tensor, k: int = 8) -> torch.Tensor:
    """将序列长度补齐到 k 的倍数（尾部填 [-1 抬笔]）。"""
    b, l, _ = line.shape
    if l % k == 0:
        return line
    pad_l = k - (l % k)
    padding = torch.tensor([0, 0, -1]).unsqueeze(0).repeat(b, pad_l, 1).to(line.device)
    return torch.cat([line, padding], dim=1)