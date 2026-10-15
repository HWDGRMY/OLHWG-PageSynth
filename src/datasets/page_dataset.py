#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: page_dataset.py
@description: 解析 .wptt 文件，把整页轨迹拆分为多行并提取文本。
"""

import logging
from typing import List, Dict, Any, Tuple

import numpy as np
import torch

from .load_wptt import load_wptt

logger = logging.getLogger(__name__)

# 官方模型的字符编码约定
EOS_TOKEN = 4052
START_TOKEN = 4053
END_TOKEN = 4054


def strokes_to_trajectory(strokes: List[np.ndarray]) -> np.ndarray:
    """将 load_wptt 返回的 strokes 列表拼装成 (N, 3) 轨迹。

    约定 state：一笔的最后一个点 state = -1，其余点 state = 1。

    Args:
        strokes: list，每个元素形状为 (num_points, 2)。

    Returns:
        形状为 (N, 3) 的绝对坐标轨迹。
    """
    if not strokes:
        return np.zeros((0, 3), dtype=np.float32)
    pieces = []
    for stroke in strokes:
        n = len(stroke)
        if n == 0:
            continue
        seg = np.zeros((n, 3), dtype=np.float32)
        seg[:, :2] = stroke
        seg[:, 2] = 1.0
        seg[-1, 2] = -1.0  # 一笔的终点标记抬笔
        pieces.append(seg)
    return np.concatenate(pieces, axis=0) if pieces else np.zeros((0, 3), dtype=np.float32)


def convert_to_delta(traj: np.ndarray) -> np.ndarray:
    """绝对坐标 → 相邻点相对位移（保留 state 不变）。"""
    if len(traj) <= 1:
        return traj.copy()
    delta = traj.copy()
    delta[1:, 0] = traj[1:, 0] - traj[:-1, 0]
    delta[1:, 1] = traj[1:, 1] - traj[:-1, 1]
    return delta


def parse_wptt_file(wptt_path: str) -> np.ndarray:
    """读取 .wptt 并返回整页绝对坐标轨迹 (N, 3)。"""
    data = load_wptt(wptt_path)
    return strokes_to_trajectory(data['strokes'])


def parse_wptt_lines(wptt_path: str) -> List[Dict[str, Any]]:
    """按行解析 .wptt。

    Args:
        wptt_path: .wptt 文件路径。

    Returns:
        list of dict，每个元素包含：
            - 'text': 该行的文本字符串（已解码）
            - 'traj': 该行的绝对坐标轨迹 (N, 3)
            - 'stroke_count': 该行包含的笔画数
    """
    data = load_wptt(wptt_path)
    strokes = data['strokes']
    lines = data['lines']

    result: List[Dict[str, Any]] = []
    for line_info in lines:
        idx_list = line_info['stroke_indices']
        text = line_info['text']
        line_strokes = [strokes[i] for i in idx_list if 0 <= i < len(strokes)]
        line_traj = strokes_to_trajectory(line_strokes)
        result.append({
            'text': text,
            'traj': line_traj,
            'stroke_count': len(line_strokes),
        })
    return result


def extract_page_text(wptt_path: str) -> str:
    """提取整页文本，行与行之间用 '\\n' 拼接。

    这是 04/05 脚本里读取 test_page 文本的入口。
    """
    lines = parse_wptt_lines(wptt_path)
    return '\n'.join(line['text'] for line in lines)


def page_char_ids(text: str, mydict: List[str]) -> List[int]:
    """将整页文本逐字转换为 id 列表（忽略换行）。

    Args:
        text: 含 '\\n' 的整页文本。
        mydict: 字符到 id 的映射字典（list 形式，用 .index 查询）。

    Returns:
        一个 list，每个元素是「一行」的字符 id 列表。
    """
    per_line_ids: List[List[int]] = []
    for line in text.split('\n'):
        if not line:
            continue
        line_ids = []
        for ch in line:
            if ch in mydict:
                line_ids.append(mydict.index(ch))
            else:
                logger.warning(f"Character '{ch}' not in dictionary, skipped.")
        if line_ids:
            per_line_ids.append(line_ids)
    return per_line_ids


def get_line_batch(all_datas_path: str, mydict_path: str,
                   char_boxes_path: str,
                   dictionary_path: str,
                   batch_size: int = 32, max_char: int = 25,
                   device: str = 'cpu'):
    """从 all_datas.npy 中随机抽取一批行级布局数据，用于训练 BoxLstm。

    Args:
        all_datas_path: all_datas.npy 路径。
        mydict_path: 单字字典。
        char_boxes_path: 由 count.py 统计得到的字符先验 npy。
        dictionary_path: 行字典（行级模型使用）。
        batch_size: 批大小。
        max_char: 一行最多截取的字符数。
        device: 设备字符串。

    Returns:
        (target_seq, ref_seq)：分别为 (B, max_char, 4) 和 (B, max_char, 6)。
    """
    datas = np.load(all_datas_path, allow_pickle=True)
    dictionary = np.load(dictionary_path, allow_pickle=True).tolist()
    char_boxes = np.load(char_boxes_path, allow_pickle=True)

    target_seq = np.zeros((batch_size, max_char, 4), dtype=np.float32)
    ref_seq = np.zeros((batch_size, max_char, 6), dtype=np.float32)

    writers = np.random.randint(0, len(datas), batch_size)
    for k, writer in enumerate(writers):
        data = datas[writer]
        line_num = data['line_num']
        line_id = np.random.randint(0, line_num)
        line = data[f'{line_id}'].copy()
        line[:, 0:2] = np.cumsum(line[:, 0:2], axis=0)
        tag = data[f'{line_id}tag']
        len_tag = min(len(tag), max_char)
        duration = np.array([0] + data[f'{line_id}duration'])
        d = np.cumsum(duration)

        x_max_ = 0.0
        for cid in range(len_tag):
            char_traj = line[d[cid]:d[cid + 1], :]
            if len(char_traj) == 0:
                continue
            h = char_traj[:, 1].max() - char_traj[:, 1].min()
            w = char_traj[:, 0].max() - char_traj[:, 0].min()
            center = (char_traj[:, 1].max() + char_traj[:, 1].min()) / 2
            x_min = char_traj[:, 0].min()
            x_max = char_traj[:, 0].max()

            target_seq[k, cid] = [h, w, center, x_min - x_max_]
            x_max_ = x_max

            char = tag[cid]
            if char not in dictionary:
                continue
            idx = dictionary.index(char)
            ref_seq[k, cid, 0] = char_boxes[idx]['h'][0]
            ref_seq[k, cid, 1] = char_boxes[idx]['h'][1]
            ref_seq[k, cid, 2] = char_boxes[idx]['w'][0]
            ref_seq[k, cid, 3] = char_boxes[idx]['w'][1]
            ref_seq[k, cid, 4] = char_boxes[idx]['center'][0]
            ref_seq[k, cid, 5] = char_boxes[idx]['center'][1]

    return (torch.from_numpy(target_seq).float().to(device),
            torch.from_numpy(ref_seq).float().to(device))