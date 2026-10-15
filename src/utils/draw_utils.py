#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: draw_utils.py
@description: 绘制单字、单行、整页轨迹的辅助函数。
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def _plot_trajectory(traj: np.ndarray, line_width: float = 1.0, color: str = 'black'):
    """在 matplotlib 当前画布上绘制一条轨迹（含抬笔切分）。"""
    strokes = np.split(traj, np.where(traj[:, 2] == -1)[0] + 1)
    strokes.pop()
    for s in strokes:
        plt.plot(s[:, 0], -s[:, 1], linewidth=line_width, color=color)
    plt.gca().set_aspect(1)


def draw_char(traj: np.ndarray, save_path: str, line_width: float = 1.5):
    """绘制单个字。"""
    plt.figure()
    _plot_trajectory(traj, line_width)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=700, bbox_inches="tight")
    plt.close()


def draw_line(traj: np.ndarray, save_path: str, line_width: float = 0.8):
    """绘制一整行。"""
    plt.figure()
    _plot_trajectory(traj, line_width, color='blue')
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=700, bbox_inches="tight")
    plt.close()


def draw_page(traj: np.ndarray, save_path: str, line_width: float = 0.5):
    """绘制整页。"""
    plt.figure(figsize=(8, 12))
    _plot_trajectory(traj, line_width, color='black')
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()