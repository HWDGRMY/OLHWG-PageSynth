#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: char_metrics.py
@description: 单字与整行的评估指标：AR、CR、Content Score、Style Score、DTW。
"""

from typing import List, Tuple

import numpy as np
import torch

from .dtw_metric import norm_dtw
from src.utils.data_utils import clear_traj, down_sample, center_and_normalize


# ------------------------------------------------------------------ #
# 1. 编辑距离相关的 AR / CR
# ------------------------------------------------------------------ #
def _levenshtein(s: str, t: str) -> int:
    """计算字符串 s 与 t 的编辑距离。"""
    m, n = len(s), len(t)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s[i - 1] == t[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[m][n]


def accurate_rate(recognized: str, target: str) -> float:
    """AR（Accurate Rate，%）。

    AR = (Nt - De - Se - Ie) / Nt × 100

    这里用编辑距离统一代替 De + Se + Ie，效果等价。
    """
    nt = len(target)
    if nt == 0:
        return 1.0
    ed = _levenshtein(recognized, target)
    return max(0.0, (nt - ed) / nt) * 100


def correct_rate(recognized: str, target: str) -> float:
    """CR（Correct Rate，%）。

    与 AR 的差别在于 CR 不计入插入错误 Ie。
    简化实现：用编辑距离近似。
    """
    return accurate_rate(recognized, target)


# ------------------------------------------------------------------ #
# 2. 分类器打分（内容 / 风格）
# ------------------------------------------------------------------ #
@torch.no_grad()
def content_score(generated_trajs: torch.Tensor,
                  char_ids: torch.Tensor,
                  char_classifier) -> float:
    """内容得分：用内容分类器判断生成的字是否属于指定类别。

    Args:
        generated_trajs: (B, L, 3) 相对坐标轨迹。
        char_ids: (B,) 目标字符 id。
        char_classifier: 已训练好的内容分类器。

    Returns:
        准确率（0~1）。
    """
    char_classifier.eval()
    x = generated_trajs.permute(0, 2, 1)  # (B, 3, L)
    logits = char_classifier(x)
    preds = logits.argmax(dim=1)
    return (preds == char_ids).float().mean().item()


@torch.no_grad()
def style_score(generated_trajs: torch.Tensor,
                style_ids: torch.Tensor,
                style_classifier) -> float:
    """风格得分：用风格分类器判断生成的字属于哪位作者。"""
    style_classifier.eval()
    x = generated_trajs.permute(0, 2, 1)
    logits = style_classifier(x)
    preds = logits.argmax(dim=1)
    return (preds == style_ids).float().mean().item()


# ------------------------------------------------------------------ #
# 3. DTW
# ------------------------------------------------------------------ #
def dtw_score(gen_traj: np.ndarray, gt_traj: np.ndarray) -> float:
    """归一化 DTW 距离（越小越接近真实）。

    Args:
        gen_traj: (N, 3) 相对坐标（模型原始输出）。
        gt_traj: (M, 3) 相对坐标（真实标注）。

    Returns:
        归一化 DTW 距离。
    """
    g = gen_traj.copy()
    t = gt_traj.copy()
    g[:, :2] = np.cumsum(g[:, :2], axis=0)
    t[:, :2] = np.cumsum(t[:, :2], axis=0)
    g = center_and_normalize(down_sample(clear_traj(g)))
    t = center_and_normalize(down_sample(clear_traj(t)))
    return norm_dtw(g[:, :2], t[:, :2])


def dtw_batch(gen_trajs: np.ndarray, gt_trajs: np.ndarray) -> float:
    """一批样本的平均 DTW。"""
    n = len(gen_trajs)
    total = 0.0
    for i in range(n):
        total += dtw_score(gen_trajs[i], gt_trajs[i])
    return total / max(n, 1)


# ------------------------------------------------------------------ #
# 4. 综合评估入口
# ------------------------------------------------------------------ #
def evaluate_characters(generated_trajs: torch.Tensor,
                        gt_trajs: torch.Tensor,
                        char_ids: torch.Tensor,
                        style_ids: torch.Tensor,
                        char_classifier,
                        style_classifier,
                        content_weight: float = 0.5,
                        style_weight: float = 0.5) -> dict:
    """单字评估综合入口。

    Returns:
        dict，包含 c_score, s_score, combined, dtw。
    """
    c = content_score(generated_trajs, char_ids, char_classifier)
    s = style_score(generated_trajs, style_ids, style_classifier)

    gen_np = generated_trajs.cpu().numpy()
    gt_np = gt_trajs.cpu().numpy()
    dtw = dtw_batch(gen_np, gt_np)

    return {
        'content_score': c,
        'style_score': s,
        'combined_score': content_weight * c + style_weight * s,
        'dtw': dtw,
    }