#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: rebuild_datas_from_pot.py
@description: 用 .pot 文件重建 datas.npy。
              关键处理：
              1. .pot 是 (x, y, state) 绝对像素
              2. datas 是 (y, x, state) 相对坐标
              3. 所以要交换 x/y + center_and_normalize + 差分
"""

import os, sys, re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT); sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import logging

sys.path.append('src/datasets')
from load_pot import load_pot_trajs
from src.utils.data_utils import center_and_normalize

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# .pot 目录（按加载顺序）
POT_DIRS = [
    'data/raw/Pot1.0Train',
    'data/raw/Pot1.0Test_2',
    'data/raw/Pot1.1Train_2',
    'data/raw/Pot1.1Test_2',
    'data/raw/Pot1.2Test',
    'data/raw/Pot1.2Train_2',
]


def pot_abs_to_datas_format(traj_abs: np.ndarray) -> np.ndarray:
    """把 .pot 的 (x, y) 绝对像素坐标转成 datas 的 (y, x) 相对坐标格式。

    流程：
        1. 交换 x/y  -> 从 (x, y) 变 (y, x)
        2. center_and_normalize（高度缩放到 1）
        3. 差分（第 0 行保留绝对）
    """
    t = traj_abs.copy()
    # 1. 交换 x/y
    # t[:, [0, 1]] = t[:, [1, 0]]
    # 2. 归一化
    t = center_and_normalize(t)
    # 3. 差分
    diff = t.copy()
    if len(t) > 1:
        diff[1:, 0] = t[1:, 0] - t[:-1, 0]
        diff[1:, 1] = t[1:, 1] - t[:-1, 1]
    return diff


def main():
    # 收集所有 .pot 文件
    all_pots = []
    for d in POT_DIRS:
        if not os.path.exists(d):
            logger.warning(f"目录不存在: {d}")
            continue
        files = sorted(Path(d).glob('*.pot'))
        logger.info(f"{d}: {len(files)} 个 .pot")
        for f in files:
            m = re.findall(r'\d+', f.stem)
            if m:
                all_pots.append((int(m[0]), str(f)))

    all_pots.sort()
    logger.info(f"总计 {len(all_pots)} 个 .pot 文件")

    datas_new = []
    for i, (num, path) in enumerate(all_pots):
        if (i + 1) % 100 == 0:
            logger.info(f"  处理 {i+1}/{len(all_pots)}, 当前作者数 {len(datas_new)}")

        try:
            samples = load_pot_trajs(path)
        except Exception as e:
            logger.error(f"  [FAIL] {path}: {e}")
            datas_new.append([])
            continue

        writer_data = []
        for char, traj_abs in samples:
            if char == '?':
                continue
            try:
                traj = pot_abs_to_datas_format(traj_abs)
                writer_data.append([char, traj])
            except Exception:
                continue

        datas_new.append(writer_data)

    # 保存
    out = 'data/processed/datas_full.npy'
    np.save(out, np.array(datas_new, dtype=object), allow_pickle=True)
    logger.info(f"\nSaved: {out}")

    lens = [len(w) for w in datas_new]
    logger.info(f"  作者数: {len(datas_new)}")
    logger.info(f"  字数: min={min(lens)}, max={max(lens)}, mean={np.mean(lens):.0f}")


if __name__ == '__main__':
    main()