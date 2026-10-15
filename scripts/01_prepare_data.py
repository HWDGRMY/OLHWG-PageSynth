#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 01_prepare_data.py
@description: 预处理数据，切分 4+1 页，并为每个作者提取缓存风格特征。

数据源：
    - datas.npy         单字数据集（list of writer_char_lists）
    - metadata.csv      作者划分元数据
    - style_encoder     预训练风格编码器

输出：
    - configs/author_split.yaml
    - data/processed/style_features/{writer_id}.pt
"""

import argparse
import logging
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm

from src.models.style_encoder import StyleEncoder
from src.pipeline.style_extractor import extract_style_features

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def main():
    args = parse_args()
    config = load_config(args.config)

    device = torch.device(config['generation']['device']
                          if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")

    # ---------- 1. 读取 metadata，构造 4+1 划分 ---------- #
    metadata_path = PROJECT_ROOT / config['paths']['metadata_csv']
    logger.info(f"Loading metadata from {metadata_path}")
    df = pd.read_csv(metadata_path)
    logger.info(f"metadata rows = {len(df)}, unique writers = {df['writer_id'].nunique()}")

    author_split = {}
    for writer_id, group in df.groupby('writer_id'):
        train_pages = group[group['split'] == 'Train']['file'].tolist()
        test_pages = group[group['split'] == 'Test']['file'].tolist()
        author_split[int(writer_id)] = {
            'train_pages': train_pages,
            'test_page': test_pages[0] if test_pages else None,
        }

    split_config_path = PROJECT_ROOT / 'configs' / 'author_split.yaml'
    with open(split_config_path, 'w', encoding='utf-8') as f:
        yaml.dump(author_split, f, allow_unicode=True)
    logger.info(f"Saved author split config to {split_config_path}")

    # ---------- 2. 加载 datas.npy（单字数据） ---------- #
    datas_path = PROJECT_ROOT / config['data']['datas_npy']
    logger.info(f"Loading datas.npy from {datas_path}")
    datas = np.load(datas_path, allow_pickle=True).tolist()
    logger.info(f"datas.npy contains {len(datas)} writers")

    # ---------- 3. 加载风格编码器 ---------- #
    style_ckpt = PROJECT_ROOT / config['models']['style_encoder_ckpt']
    logger.info(f"Loading style encoder from {style_ckpt}")
    style_encoder = StyleEncoder()
    style_encoder.load_state_dict(torch.load(style_ckpt, map_location='cpu'))
    style_encoder.to(device)
    style_encoder.eval()

    # ---------- 4. 逐个作者提取并缓存风格特征 ---------- #
    style_cache_dir = PROJECT_ROOT / config['paths']['style_cache_dir']
    style_cache_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Extracting and caching style features...")
    ok, skipped, failed = 0, 0, 0

    for writer_id in tqdm(sorted(author_split.keys()), desc="Caching style features"):
        cache_path = style_cache_dir / f"{writer_id}.pt"
        if cache_path.exists():
            skipped += 1
            continue

        # datas.npy 索引：writer_id - 1
        datas_idx = writer_id - 1
        if datas_idx < 0 or datas_idx >= len(datas):
            skipped += 1
            continue

        writer_chars = datas[datas_idx]
        if len(writer_chars) == 0:
            skipped += 1
            continue

        try:
            style_features = extract_style_features(
                datas, writer_id, style_encoder, device,
                num_samples=1,     # ★ 关键：与官方一致，只采样一次
            )
            torch.save([t.cpu() for t in style_features], cache_path)
            ok += 1
        except Exception as e:
            logger.error(f"writer {writer_id} failed: {e}")
            failed += 1

    logger.info(f"Done. OK={ok}, Skipped={skipped}, Failed={failed}")


if __name__ == '__main__':
    main()