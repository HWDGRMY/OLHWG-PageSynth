#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 00c_build_page_stats.py
@description: 从 .wptt 提取每位作者的行距、行倾斜统计量。
              - 行距：相邻行 y 间距 / 平均字高（归一化）
              - 行倾斜：每行字符中心点线性回归斜率
              - 缩进：不在此统计，生成时固定用 2 字符宽
"""

import os
import sys
import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import yaml

from src.datasets.load_wptt import load_wptt

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def normalize_path(raw_path: str, base_data_dir: str) -> str:
    """把 metadata.csv 里的 Windows 绝对路径转成本地相对路径。"""
    p = Path(str(raw_path).replace('\\', '/'))
    filename = p.name
    split_folder = None
    for part in p.parts:
        if part.startswith('WPTT'):
            split_folder = part
            break
    if split_folder is None:
        return None
    return str(Path(base_data_dir) / split_folder / filename)


def process_page(wptt_path: str):
    """解析一页 wptt，返回每行的 y_top/y_center/char_centers/char_heights。"""
    try:
        data = load_wptt(wptt_path)
    except Exception as e:
        logger.debug(f"load_wptt failed for {wptt_path}: {e}")
        return None

    strokes = data['strokes']
    lines = data['lines']
    if len(lines) < 2:
        return None

    line_infos = []
    for line in lines:
        idx_list = line['stroke_indices']
        text = line['text']
        if len(text) == 0 or len(idx_list) == 0:
            continue

        line_strokes = [strokes[i] for i in idx_list if 0 <= i < len(strokes)]
        if len(line_strokes) == 0:
            continue

        pts = np.concatenate(line_strokes, axis=0)  # (N, 2)
        y_min = pts[:, 1].min()
        y_max = pts[:, 1].max()

        # 按 text 均分笔画 → 每个字的中心点 + 字高
        n_chars = len(text)
        n_strokes = len(idx_list)
        char_centers = []
        char_heights = []
        for i in range(n_chars):
            start = int(round(i * n_strokes / n_chars))
            end = int(round((i + 1) * n_strokes / n_chars))
            if start == end:
                continue
            char_pts = []
            for si in idx_list[start:end]:
                if 0 <= si < len(strokes):
                    char_pts.append(strokes[si])
            if not char_pts:
                continue
            cp = np.concatenate(char_pts, axis=0)
            xc = (cp[:, 0].max() + cp[:, 0].min()) / 2
            yc = (cp[:, 1].max() + cp[:, 1].min()) / 2
            h = cp[:, 1].max() - cp[:, 1].min()
            char_centers.append((xc, yc))
            char_heights.append(h)

        if len(char_centers) < 2:
            continue

        line_infos.append({
            'y_top': y_min,
            'y_center': (y_min + y_max) / 2,
            'char_centers': char_centers,
            'char_heights': char_heights,
        })

    return line_infos if len(line_infos) >= 2 else None


def aggregate_page(line_infos):
    """从一页的多行信息里聚合行距和行倾斜。"""
    line_infos = sorted(line_infos, key=lambda x: x['y_top'])

    y_centers = np.array([i['y_center'] for i in line_infos])
    y_spacings = np.diff(y_centers)
    y_spacings = y_spacings[y_spacings > 0]

    # 平均字高（归一化基准）
    all_heights = []
    for info in line_infos:
        all_heights.extend(info['char_heights'])
    mean_char_h = float(np.mean(all_heights)) if all_heights else 1.0

    if mean_char_h <= 0 or len(y_spacings) == 0:
        return None

    y_spacings_norm = y_spacings / mean_char_h   # 单位：多少个字高

    # 行倾斜：每行字符中心点做线性回归
    slopes = []
    for info in line_infos:
        cc = info['char_centers']
        if len(cc) < 3:
            continue
        xs = np.array([c[0] for c in cc])
        ys = np.array([c[1] for c in cc])
        if xs.max() - xs.min() < mean_char_h * 0.5:
            continue
        xs_norm = (xs - xs.min()) / mean_char_h
        ys_norm = ys / mean_char_h
        slope, _ = np.polyfit(xs_norm, ys_norm, 1)
        slopes.append(float(slope))

    return {
        'y_spacing_mean': float(np.mean(y_spacings_norm)),
        'y_spacing_std':  float(np.std(y_spacings_norm)),
        'slope_median':   float(np.median(slopes)) if slopes else 0.0,
        'slope_std':      float(np.std(slopes)) if slopes else 0.05,
        'n_lines':        len(line_infos),
    }


def main():
    with open('configs/default.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    raw_data_dir = config['paths']['raw_data_dir']
    metadata_csv = config['paths']['metadata_csv']

    df = pd.read_csv(metadata_csv)
    logger.info(f"metadata: {len(df)} rows, {df['writer_id'].nunique()} writers")

    writer_pages = df.groupby('writer_id')['file'].apply(list).to_dict()

    page_stats = {}
    success, failed = 0, 0

    for writer_id, pages in writer_pages.items():
        if writer_id % 100 == 0:
            logger.info(f"  writer_id {writer_id}: ok={success}, failed={failed}")

        all_spacings, all_slopes = [], []
        for p in pages:
            path = normalize_path(p, raw_data_dir)
            if path is None or not os.path.exists(path):
                continue
            infos = process_page(path)
            if infos is None:
                continue
            agg = aggregate_page(infos)
            if agg is None:
                continue
            all_spacings.append(agg['y_spacing_mean'])
            all_slopes.append(agg['slope_median'])

        if len(all_spacings) == 0:
            failed += 1
            continue

        page_stats[int(writer_id)] = {
            'y_spacing_mean': float(np.mean(all_spacings)),
            'y_spacing_std':  float(np.std(all_spacings)) if len(all_spacings) > 1 else 0.08,
            'slope_median':   float(np.median(all_slopes)) if all_slopes else 0.0,
            'slope_std':      float(np.std(all_slopes)) if len(all_slopes) > 1 else 0.05,
            'n_pages':        len(all_spacings),
        }
        success += 1

    logger.info(f"\n成功: {success}, 失败: {failed}")

    if page_stats:
        ys = [s['y_spacing_mean'] for s in page_stats.values()]
        sl = [s['slope_median'] for s in page_stats.values()]
        logger.info(f"行距: min={min(ys):.3f}, max={max(ys):.3f}, mean={np.mean(ys):.3f}")
        logger.info(f"倾斜: min={min(sl):.4f}, max={max(sl):.4f}, mean={np.mean(sl):.4f}")

        for wid in [1, 15, 16, 100]:
            if wid in page_stats:
                s = page_stats[wid]
                logger.info(f"  writer {wid}: 行距={s['y_spacing_mean']:.3f}±{s['y_spacing_std']:.3f}, "
                            f"倾斜={s['slope_median']:.4f}±{s['slope_std']:.4f}, "
                            f"页数={s['n_pages']}")

    save_path = Path('data/processed/page_stats.npy')
    save_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(save_path, np.array(page_stats, dtype=object), allow_pickle=True)
    logger.info(f"Saved: {save_path}")


if __name__ == '__main__':
    main()