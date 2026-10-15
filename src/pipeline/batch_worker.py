#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: batch_worker.py
@description: 批量处理 1020 个作者的整页生成任务，支持断点续传。
"""

import logging
from pathlib import Path
from typing import Dict, Any

import torch
import numpy as np
from tqdm import tqdm

from .page_generator import generate_page

logger = logging.getLogger(__name__)


def run_batch(author_split: Dict[int, Dict[str, Any]],
              ddpm, style_encoder, box_lstm,
              device: torch.device, mydict: dict,
              output_dir: str, style_cache_dir: str) -> None:
    """遍历所有作者，为每个作者生成一页。

    Args:
        author_split: 由 01_prepare_data.py 生成的 author_split 字典。
        ddpm / style_encoder / box_lstm: 已加载的模型。
        device: 计算设备。
        mydict: 字符字典。
        output_dir: 输出根目录。
        style_cache_dir: 风格特征缓存目录。
    """
    ddpm.eval()
    style_encoder.eval()
    box_lstm.eval()

    output_dir = Path(output_dir)
    style_cache_dir = Path(style_cache_dir)

    for writer_id in tqdm(sorted(author_split.keys()), desc="Batch generating"):
        save_dir = output_dir / f"author_{writer_id:04d}"
        save_path = save_dir / "generated_page.npy"

        # 断点续传
        if save_path.exists():
            continue

        test_page_info = author_split[writer_id].get('test_page')
        if not test_page_info:
            continue

        style_cache_path = style_cache_dir / f"{writer_id}.pt"
        if not style_cache_path.exists():
            logger.warning(f"Style cache missing for writer {writer_id}, skipping.")
            continue

        try:
            style_features = torch.load(style_cache_path, map_location=device)
            # 读取该页的真实文本（需从配套的 .txt 标签文件中提取）
            text = _read_page_text(test_page_info)

            save_dir.mkdir(parents=True, exist_ok=True)
            with torch.no_grad():
                page_traj = generate_page(
                    text, ddpm, style_encoder, box_lstm,
                    device, mydict, style_features=style_features,
                )
            np.save(save_path, page_traj)

        except Exception as e:
            logger.error(f"Writer {writer_id} failed: {e}")
            continue


def _read_page_text(test_page_info: str) -> str:
    """从 .wptt 配套的 .txt 标签文件中读取整页文本（含换行）。

    CASIA 数据集中，每个 .wptt 通常有一个同名的 .txt 提供内容。
    具体格式以你的数据为准。
    """
    # 占位实现，需根据实际标签文件格式调整
    txt_path = Path(test_page_info).with_suffix('.txt')
    if not txt_path.exists():
        raise FileNotFoundError(f"Label file not found: {txt_path}")
    with open(txt_path, 'r', encoding='utf-8') as f:
        return f.read()