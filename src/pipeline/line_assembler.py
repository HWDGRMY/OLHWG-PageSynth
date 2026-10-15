#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: line_assembler.py
@description: 单字轨迹归一化缩放 + 单行拼装。

包含两个核心函数：
  - put_char: 把单字轨迹归一化后缩放到指定边界框，供 page_generator 调用。
  - generate_line: 直接生成一整行（不含布局预测，用于 03_infer_single_line 快速验证）。
"""

import logging
from typing import List

import numpy as np
import torch

from src.utils.data_utils import clear_traj, down_sample, center_and_normalize

logger = logging.getLogger(__name__)

# 与官方代码一致的特殊 token
EOS_TOKEN = 4052
START_TOKEN = 4053
END_TOKEN = 4054

# 单字总时间步长
TOTAL_SEQ_LEN = 120

# 单字活跃长度（训练集中位数）
DEFAULT_CHAR_ACTIVE_LEN = 57


# ------------------------------------------------------------------ #
# 1. 单字缩放填入边界框
# ------------------------------------------------------------------ #
def put_char(traj: np.ndarray, bbox: List[float],
             normalize_first: bool = True) -> np.ndarray:
    """把单字轨迹归一化并缩放到指定的边界框内。

    Args:
        traj: (N, 3) 单字绝对坐标轨迹。
        bbox: [ymax, ymin, xmax, xmin]。
        normalize_first: 是否先做 RDP 降采样 + 归一化到高度 1。
            ⚠️ 必须为 True，否则 raw scale (delta 约 ±10) 会被硬塞进
            [0,1] 尺度的框里，整行会退化成锯齿。

    Returns:
        缩放和平移后的轨迹 (N, 3)。
    """
    ymax, ymin, xmax, xmin = bbox
    traj = traj.copy().astype(np.float32)

    # ---- 1. 归一化到高度 1 ---- #
    if normalize_first:
        traj = clear_traj(traj)
        traj = down_sample(traj)
        traj = center_and_normalize(traj)

    # ---- 2. 缩放到框的大小 ---- #
    x_range = max(traj[:, 0].max() - traj[:, 0].min(), 1e-6)
    y_range = max(traj[:, 1].max() - traj[:, 1].min(), 1e-6)
    traj[:, 0] = (traj[:, 0] - traj[:, 0].min()) / x_range * (xmax - xmin) + xmin
    traj[:, 1] = (traj[:, 1] - traj[:, 1].min()) / y_range * (ymax - ymin) + ymin
    return traj


# ------------------------------------------------------------------ #
# 2. 单字生成（官方 label 格式）
# ------------------------------------------------------------------ #
def _generate_char_traj(char: str, ddpm, mydict: List[str],
                        style_features, device: torch.device,
                        char_active_len: int) -> np.ndarray:
    """生成单个字的绝对坐标轨迹。

    Args:
        char: 单个字符。
        ddpm: DDPM 模型。
        mydict: 字符 → id 字典。
        style_features: 风格特征。
        device: 计算设备。
        char_active_len: 字的活跃步数。

    Returns:
        (char_active_len, 3) 绝对坐标轨迹。
    """
    if char not in mydict:
        raise ValueError(f"'{char}' not in mydict")

    char_id = mydict.index(char)
    L = TOTAL_SEQ_LEN

    # 官方 label 格式：[START] + char_id*(len-2) + [END] + EOS*(L-len)
    label = ([START_TOKEN]
             + [char_id] * (char_active_len - 2)
             + [END_TOKEN]
             + [EOS_TOKEN] * (L - char_active_len))
    duration = [[char_active_len, L - char_active_len]]

    with torch.no_grad():
        traj = ddpm.generate([label], [duration], style_features)

    # (1, 3, L) → (L, 3)，只取前 char_active_len 步
    char_traj = traj.permute(0, 2, 1)[0].cpu().numpy()[:char_active_len]
    char_traj[:, 2] = np.sign(char_traj[:, 2])
    char_traj[:, 0:2] = np.cumsum(char_traj[:, 0:2], axis=0)
    return char_traj


# ------------------------------------------------------------------ #
# 3. 单行生成（骨架版，用于 03_infer_single_line 快速验证）
# ------------------------------------------------------------------ #
def generate_line(text: str, ddpm, style_encoder, box_lstm,
                   device: torch.device, mydict: List[str],
                   style_features=None,
                   y_offset: float = 0.0,
                   char_active_len: int = DEFAULT_CHAR_ACTIVE_LEN,
                   char_width: float = 0.7,
                   char_advance: float = 0.8) -> np.ndarray:
    """生成一整行手写轨迹（不使用布局 LSTM，仅按固定宽度水平铺开）。

    用于 03_infer_single_line.py 快速验证；正式整页生成请用
    page_generator.generate_page（它带 box_lstm 布局预测）。

    Args:
        text: 一行文本。
        ddpm: DDPM 模型。
        style_encoder: 风格编码器（此处不直接用，保留接口一致）。
        box_lstm: 布局 LSTM（此处不直接用，保留接口一致）。
        device: 计算设备。
        mydict: 字符 → id 字典。
        style_features: 长度 4 的风格特征列表，**不能为 None**。
        y_offset: 该行的垂直偏移。
        char_active_len: 每个字的活跃步数。
        char_width: 每个字在行内占据的宽度。
        char_advance: 相邻字的 x 推进量。

    Returns:
        (N, 3) 整行绝对坐标轨迹。
    """
    if style_features is None:
        raise ValueError(
            "style_features is None —— 必须从缓存 .pt 里加载一个作者的风格特征。"
        )

    # 过滤掉字典里没有的字
    chars = [c for c in text if c in mydict]
    if not chars:
        logger.warning(f"No valid characters in '{text}', returning empty line.")
        return np.zeros((0, 3), dtype=np.float32)

    # 逐字生成
    all_char_trajs = []
    for ch in chars:
        try:
            char_traj = _generate_char_traj(
                ch, ddpm, mydict, style_features, device, char_active_len,
            )
            all_char_trajs.append(char_traj)
        except Exception as e:
            logger.error(f"Failed to generate '{ch}': {e}")
            continue

    # 水平铺开 + 填入框
    line_traj = np.array([[0.0, 0.0, -1.0]], dtype=np.float32)
    x_cursor = 0.0

    for char_traj in all_char_trajs:
        # 每个字先归一化
        char_traj = clear_traj(char_traj)
        char_traj = down_sample(char_traj)
        char_traj = center_and_normalize(char_traj)

        # 直接用 put_char 也可以，但这里明确用固定框
        bbox = [
            y_offset + 0.5,   # ymax
            y_offset - 0.5,   # ymin
            x_cursor + char_width,   # xmax
            x_cursor,                # xmin
        ]
        scaled = put_char(char_traj, bbox, normalize_first=False)
        line_traj = np.concatenate([line_traj, scaled], axis=0)
        x_cursor += char_advance

    return line_traj[1:]