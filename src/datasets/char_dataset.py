#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: char_dataset.py
@description: 单字手写数据集，用于 DDPM 扩散模型的训练和评估。
"""

import logging
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .load_wptt import load_wptt  # 若需要直接读 wptt 文件也可用它

logger = logging.getLogger(__name__)

MAX_SEQ_LEN = 120  # 官方模型单字最大长度


class CharDataset(Dataset):
    """单字手写数据集。

    datas.npy 结构（参考官方代码）：
        datas[writer_id] = list of [char, traj]
        traj: (N, 3) 相对坐标轨迹

    Args:
        datas_path: datas.npy 路径。
        mydict_path: mydict.npy 路径（字符到 id）。
        seq_len: 单字最大长度，超出截断，不足 padding。
    """

    def __init__(self, datas_path: str, mydict_path: str, seq_len: int = MAX_SEQ_LEN):
        self.datas = np.load(datas_path, allow_pickle=True).tolist()
        self.mydict = np.load(mydict_path, allow_pickle=True).tolist()
        self.seq_len = seq_len

        # 展平为 (writer_id, char, traj) 的三元组
        self.samples = []
        for writer_id, writer_data in enumerate(self.datas):
            for item in writer_data:
                char, traj = item[0], item[1]
                if char in self.mydict:
                    self.samples.append((writer_id, char, traj))
        logger.info(f"Loaded {len(self.samples)} char samples from {len(self.datas)} writers.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        writer_id, char, traj = self.samples[idx]
        char_id = self.mydict.index(char)

        arr = np.zeros((self.seq_len, 3), dtype=np.float32)
        n = min(len(traj), self.seq_len)
        arr[:n] = traj[:n]
        arr[n:, 2] = -1.0  # padding 用抬笔标记

        return {
            'traj': torch.from_numpy(arr),           # (seq_len, 3)
            'char_id': torch.tensor(char_id, dtype=torch.long),
            'writer_id': torch.tensor(writer_id, dtype=torch.long),
        }


def collate_fn(batch):
    """PyTorch DataLoader 的 collate 函数。"""
    trajs = torch.stack([b['traj'] for b in batch], dim=0)      # (B, L, 3)
    char_ids = torch.stack([b['char_id'] for b in batch], dim=0)
    writer_ids = torch.stack([b['writer_id'] for b in batch], dim=0)
    return {'traj': trajs, 'char_id': char_ids, 'writer_id': writer_ids}


def get_style_batch_10(char_datas: list, style_ids: np.ndarray,
                       seq_len: int = 1000) -> np.ndarray:
    """从每位作者处采样 15 个字符并拼接，构造风格参考序列。

    Args:
        char_datas: 已加载的 datas.npy 列表。
        style_ids: 形状 (B,) 的作者 id 数组。
        seq_len: 目标序列长度。

    Returns:
        形状 (B, seq_len, 3) 的 numpy 数组。
    """
    bs = len(style_ids)
    batch = np.zeros((bs, seq_len, 3), dtype=np.float32)

    for i, sid in enumerate(style_ids):
        data = char_datas[sid]
        char_num = len(data)
        char_ids = np.random.choice(char_num, min(15, char_num), replace=False)
        temp = np.zeros((seq_len, 3), dtype=np.float32)
        total = 0
        for cid in char_ids:
            traj = data[cid][1]
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
        batch[i] = temp
    return batch