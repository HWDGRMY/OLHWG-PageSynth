#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 01_prepare_data.py
@description: 从 datas_full.npy 提取 1020 位作者的风格特征，缓存到 style_features/。
              连续索引 0~1019 作为 key，与 datas_full.npy 一致。
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import yaml
from tqdm import tqdm

from src.models.style_encoder import StyleEncoder

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

STYLE_SEQ_LEN = 1000
CHARS_PER_SAMPLE = 15


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=str, default='configs/default.yaml')
    p.add_argument('--num_samples', type=int, default=1,
                   help="风格特征采样次数（官方用 1）")
    return p.parse_args()


def build_style_ref(writer_chars, seq_len=STYLE_SEQ_LEN, num_chars=CHARS_PER_SAMPLE):
    """从一个作者的单字列表里随机挑 num_chars 个拼成 seq_len 长度的序列。"""
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
    return temp[None, ...]


def extract_style_features(writer_chars, style_encoder, device, num_samples=1):
    """返回 [x, en1, en2, en3]，每个形状 (1, C, L)。"""
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


def main():
    args = parse_args()
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    device = torch.device(config['generation']['device']
                          if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")

    # 加载 datas_full.npy（连续索引 0~1019）
    datas_path = 'data/processed/datas_full.npy'
    datas_dict = np.load(datas_path, allow_pickle=True).item()
    logger.info(f"Loaded datas_full: {len(datas_dict)} writers")
    cont_ids = sorted(datas_dict.keys())
    logger.info(f"连续索引范围: {cont_ids[0]} ~ {cont_ids[-1]}")

    # 加载 writer_id_map（用于后续反查原编号）
    map_path = Path('data/processed/writer_id_map.json')
    if not map_path.exists():
        logger.error(f"{map_path} 不存在，先跑 00d_rebuild_datas_from_pot.py")
        return

    # 加载风格编码器
    style_ckpt = config['models']['style_encoder_ckpt']
    logger.info(f"Loading style encoder: {style_ckpt}")
    style_encoder = StyleEncoder()
    style_encoder.load_state_dict(torch.load(style_ckpt, map_location='cpu'))
    style_encoder.to(device)
    style_encoder.eval()

    # 缓存目录
    cache_dir = Path(config['paths']['style_cache_dir'])
    cache_dir.mkdir(parents=True, exist_ok=True)

    # 逐作者提取
    logger.info("Extracting style features...")
    success, failed = 0, 0
    for cont_idx in tqdm(cont_ids, desc="Caching style features"):
        cache_path = cache_dir / f"{cont_idx}.pt"
        if cache_path.exists():
            success += 1
            continue

        writer_chars = datas_dict.get(cont_idx)
        if writer_chars is None or len(writer_chars) < CHARS_PER_SAMPLE:
            logger.warning(f"writer cont_idx={cont_idx} has too few chars, skip")
            failed += 1
            continue

        try:
            feats = extract_style_features(writer_chars, style_encoder, device,
                                           num_samples=args.num_samples)
            torch.save([t.cpu() for t in feats], cache_path)
            success += 1
        except Exception as e:
            logger.error(f"writer cont_idx={cont_idx} failed: {e}")
            failed += 1

    logger.info(f"Done. Success={success}, Failed={failed}")


if __name__ == '__main__':
    main()