#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 06_evaluate.py
@description: 评估生成结果与真实页面的指标
@usage: python scripts/06_evaluate.py --config configs/default.yaml
"""

import argparse
import logging
import yaml
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.evaluation.layout_metrics import calculate_binary_geometric_features
from src.evaluation.dtw_metric import calculate_dtw

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate generated results.")
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    with open('configs/author_split.yaml', 'r', encoding='utf-8') as f:
        author_split = yaml.safe_load(f)

    output_dir = Path(config['paths']['output_dir'])

    metrics = {
        'layout_diff': [],
        'dtw': []
    }

    logger.info("Starting evaluation...")
    for writer_id in tqdm(author_split.keys(), desc="Evaluating"):
        gen_page_path = output_dir / f"author_{writer_id:04d}" / "generated_page.npy"  # 假设你保存了轨迹 npy 文件
        real_page_path = author_split[writer_id]['test_page']

        if not gen_page_path.exists():
            continue

        try:
            gen_traj = np.load(gen_page_path)
            # 这里需要加载真实的 wptt 文件并解析
            # real_traj = parse_wptt_file(real_page_path)

            # 计算布局几何特征差异 (∇1 到 ∇8)
            # layout_diff = calculate_binary_geometric_features(gen_traj, real_traj)
            # metrics['layout_diff'].append(layout_diff)

            # 计算 DTW
            # dtw = calculate_dtw(gen_traj, real_traj)
            # metrics['dtw'].append(dtw)

        except Exception as e:
            logger.error(f"Evaluation failed for author {writer_id}: {e}")

    # 打印平均结果
    # logger.info(f"Average Layout Diff: {np.mean(metrics['layout_diff'], axis=0)}")
    # logger.info(f"Average DTW: {np.mean(metrics['dtw'])}")


if __name__ == '__main__':
    main()