#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: page_generator.py
@description: 整页生成——基于 .wptt 提取文本与换行，逐行生成后拼装为整页轨迹。
"""

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

from src.datasets.page_dataset import parse_wptt_lines
from src.pipeline.line_assembler import put_char

logger = logging.getLogger(__name__)

# 与官方代码一致的特殊 token
EOS_TOKEN = 4052
START_TOKEN = 4053
END_TOKEN = 4054

# 单字总时间步长（batch 内 max 长度）
TOTAL_SEQ_LEN = 120

# 单字活跃长度（训练集中位数，官方模型训练时每个字的实际步数）
DEFAULT_CHAR_ACTIVE_LEN = 57


# ------------------------------------------------------------------ #
# 1. 布局预测：用 BoxLstm 生成每个字的边界框
# ------------------------------------------------------------------ #
def predict_boxes(box_lstm, text: str, dictionary: List[str],
                  char_boxes: np.ndarray, device: torch.device,
                  len_prefix: int = 10,
                  prefix_boxes: Optional[np.ndarray] = None) -> List[List[float]]:
    """为一行文本预测每个字符的边界框。

    Args:
        box_lstm: 已加载权重的 BoxLstm。
        text: 该行的文本。
        dictionary: 行级字符字典。
        char_boxes: 由 count.py 统计得到的先验 (npy)。
        device: 计算设备。
        len_prefix: 使用多少个前置框作为条件（0 表示无条件）。
        prefix_boxes: 可选的 (len_prefix, 4) 前置框。

    Returns:
        每个字的边界框 [ymax, ymin, xmax, xmin]。
    """
    box_lstm.eval()
    len_tag = len(text)

    # ---- 构造 ref_seq：(1, len_tag, 6) 每个字符的先验统计 ---- #
    ref_seq = np.zeros((1, len_tag, 6), dtype=np.float32)
    for i, ch in enumerate(text):
        if ch not in dictionary:
            ref_seq[0, i] = [0.5, 0.05, 0.5, 0.05, 0.5, 0.05]
            continue
        idx = dictionary.index(ch)
        ref_seq[0, i, 0] = char_boxes[idx]['h'][0]
        ref_seq[0, i, 1] = char_boxes[idx]['h'][1]
        ref_seq[0, i, 2] = char_boxes[idx]['w'][0]
        ref_seq[0, i, 3] = char_boxes[idx]['w'][1]
        ref_seq[0, i, 4] = char_boxes[idx]['center'][0]
        ref_seq[0, i, 5] = char_boxes[idx]['center'][1]
    ref_seq_t = torch.from_numpy(ref_seq).float().to(device)

    with torch.no_grad():
        if len_prefix > 0 and prefix_boxes is not None and len_prefix < len_tag:
            prefix_boxes = np.asarray(prefix_boxes, dtype=np.float32)[:len_prefix]
            target_shift = np.zeros((1, len_tag, 4), dtype=np.float32)
            target_shift[0, :len_prefix] = prefix_boxes

            prefix_seq = box_lstm.func_in(ref_seq_t[:, :len_prefix, :])
            prefix_seq = torch.cat(
                [torch.from_numpy(target_shift[:, :len_prefix, :]).float().to(device),
                 prefix_seq],
                dim=-1,
            )
            ref_seq_rest = ref_seq_t[:, len_prefix:, :]
            box_list = box_lstm.get_box(ref_seq_rest, prefix_seq)
            box_arr = torch.cat(box_list).detach().cpu().numpy()[:, 0, :]
            full_box = np.concatenate([prefix_boxes, box_arr], axis=0)
        else:
            box_list = box_lstm.get_box(ref_seq_t)
            full_box = torch.cat(box_list).detach().cpu().numpy()[:, 0, :]

    # ---- 转 [ymax, ymin, xmax, xmin] ---- #
    bounding_boxes = []
    x_left = 0.0
    for i in range(full_box.shape[0]):
        h = float(full_box[i, 0])
        w = float(full_box[i, 1])
        c = float(full_box[i, 2])
        d = float(full_box[i, 3])

        ymax = c + h / 2
        ymin = c - h / 2
        xmin = x_left + d
        xmax = xmin + w
        bounding_boxes.append([ymax, ymin, xmax, xmin])
        x_left = xmax
    return bounding_boxes


# ------------------------------------------------------------------ #
# 2. 单字生成：官方 label 格式
# ------------------------------------------------------------------ #
def generate_char_trajs(text: str, ddpm, mydict: List[str],
                        style_features, device: torch.device,
                        char_active_len: int = DEFAULT_CHAR_ACTIVE_LEN
                        ) -> List[np.ndarray]:
    """为一行中的每个字符生成相对坐标轨迹。

    关键：使用官方 label 构造方式
        [START] + [char_id]*(char_len-2) + [END] + [EOS]*(TOTAL - char_len)
        duration = [[char_len, TOTAL - char_len]]
    然后只取前 char_len 步作为该字的有效轨迹。

    Args:
        text: 该行文本。
        ddpm: DDPM 模型。
        mydict: 单字字典。
        style_features: 长度 4 的风格特征列表。
        device: 计算设备。
        char_active_len: 每个字的活跃步数（默认 57）。

    Returns:
        list of (N, 3) 绝对坐标轨迹。
    """
    ddpm.eval()
    L = TOTAL_SEQ_LEN
    result = []

    for ch in text:
        if ch not in mydict:
            logger.warning(f"Character '{ch}' not in mydict, skipped.")
            continue

        char_id = mydict.index(ch)
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
        result.append(char_traj)
    return result


# ------------------------------------------------------------------ #
# 3. 整页生成：逐行拼装
# ------------------------------------------------------------------ #
def generate_page(wptt_path: str, ddpm, style_encoder, box_lstm,
                  device: torch.device,
                  dictionary: List[str], mydict: List[str],
                  char_boxes: np.ndarray,
                  style_features=None,
                  line_height: float = 1.2,
                  len_prefix: int = 10,
                  char_active_len: int = DEFAULT_CHAR_ACTIVE_LEN) -> np.ndarray:
    """根据 .wptt 中的文本与换行，生成整页手写轨迹。

    Args:
        wptt_path: 参考页面路径，用于提取文本与换行。
        ddpm / style_encoder / box_lstm: 已加载的模型。
        device: 计算设备。
        dictionary: 行级字符字典。
        mydict: 单字字典。
        char_boxes: 字符先验 npy。
        style_features: 4 页风格提取的特征（list of 4 tensors）。
        line_height: 行间距。
        len_prefix: 布局条件生成时使用的前缀长度。
        char_active_len: 单字的活跃步数。

    Returns:
        整页绝对坐标轨迹 (N, 3)。
    """
    if style_features is None:
        raise ValueError("style_features is None —— 必须从缓存 .pt 里加载。")

    # ---- 1. 从 .wptt 提取每行文本 ---- #
    lines = parse_wptt_lines(wptt_path)
    logger.info(f"Page has {len(lines)} lines.")

    full_page = np.array([[0.0, 0.0, -1.0]], dtype=np.float32)
    y_offset = 0.0

    for line_idx, line_info in enumerate(lines):
        text = line_info['text']
        if not text:
            continue
        logger.info(f"Generating line {line_idx + 1}/{len(lines)}: {text[:20]}...")

        try:
            # 2. 用 BoxLstm 预测该行每个字的边界框
            boxes = predict_boxes(
                box_lstm, text, dictionary, char_boxes,
                device, len_prefix=len_prefix,
            )

            # 3. 逐字生成轨迹（官方 label 格式）
            char_trajs = generate_char_trajs(
                text, ddpm, mydict, style_features, device,
                char_active_len=char_active_len,
            )

            # 4. 缩放并拼装该行
            line_traj = np.array([[0.0, 0.0, -1.0]], dtype=np.float32)
            for i, char_traj in enumerate(char_trajs):
                if i >= len(boxes):
                    break
                scaled = put_char(char_traj.copy(), boxes[i])
                scaled[:, 1] += y_offset
                line_traj = np.concatenate([line_traj, scaled], axis=0)

            full_page = np.concatenate([full_page, line_traj[1:]], axis=0)
        except Exception as e:
            logger.error(f"Failed to generate line {line_idx}: {e}")
            continue

        y_offset += line_height

    return full_page[1:]