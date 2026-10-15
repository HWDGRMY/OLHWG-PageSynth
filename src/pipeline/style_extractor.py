#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: style_extractor.py
@description: 从 datas.npy 提取作者风格特征（与官方 get_style_batch_10 一致）。
"""

import logging
from typing import List

import numpy as np
import torch

logger = logging.getLogger(__name__)

STYLE_SEQ_LEN = 1000
CHARS_PER_SAMPLE = 15


def build_style_ref(writer_chars: list,
                    seq_len: int = STYLE_SEQ_LEN,
                    num_chars: int = CHARS_PER_SAMPLE) -> np.ndarray:
    """从一个作者的单字列表里随机挑 num_chars 个，拼成 seq_len 长度的序列。

    Args:
        writer_chars: list，每个元素是 [char, traj]。
        seq_len: 目标长度。
        num_chars: 拼接多少个单字。

    Returns:
        形状 (1, seq_len, 3) 的 numpy 数组。
    """
    char_num = len(writer_chars)
    if char_num < num_chars:
        raise ValueError(f"Not enough chars ({char_num}) to sample {num_chars}.")

    chosen = np.random.choice(char_num, num_chars, replace=False)
    temp = np.zeros((seq_len, 3), dtype=np.float32)
    total = 0
    for ci in chosen:
        traj = writer_chars[ci][1]
        length = len(traj)
        if total + length > seq_len:
            remaining = seq_len - total
            temp[total:total + remaining] = traj[:remaining]
            total = seq_len
            break
        temp[total:total + length] = traj
        total += length
    if total < seq_len:
        temp[total:, 2] = -1.0
    return temp[None, ...]  # (1, seq_len, 3)


def extract_style_features(datas, writer_id: int, style_encoder,
                            device: torch.device,
                            num_samples: int = 3) -> List[torch.Tensor]:
    """为单个作者提取风格特征。

    Args:
        datas: 已加载的 datas.npy（list，每个元素是一个作者的单字列表）。
        writer_id: 作者编号（1-based，与 metadata.csv 一致）。
        style_encoder: StyleEncoder。
        device: 计算设备。
        num_samples: 采样多少次取平均（提高鲁棒性）。

    Returns:
        长度 4 的列表 [x, en1, en2, en3]，每个 (1, C, L)。
    """
    # datas.npy 索引：writer_id - 1（按你的数据集约定）
    datas_idx = writer_id - 1
    if datas_idx < 0 or datas_idx >= len(datas):
        raise IndexError(f"writer_id {writer_id} out of datas.npy range")

    writer_chars = datas[datas_idx]
    if len(writer_chars) == 0:
        raise ValueError(f"Writer {writer_id} has no char data.")

    style_encoder.eval()
    all_features = []
    with torch.no_grad():
        for _ in range(num_samples):
            ref = build_style_ref(writer_chars)
            ref_t = torch.from_numpy(ref).float().to(device).permute(0, 2, 1)
            x, en1, en2, en3 = style_encoder(ref_t)
            all_features.append([x, en1, en2, en3])

    features = []
    for i in range(4):
        stacked = torch.stack([f[i] for f in all_features], dim=0)
        features.append(stacked.mean(dim=0))
    return features