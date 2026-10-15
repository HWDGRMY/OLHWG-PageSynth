#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 00b_build_char_lengths.py
@description: 统计每个字符在行数据集中的平均轨迹长度，输出 char_lengths.npy。
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import yaml


def main():
    with open('configs/default.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    datas = np.load(config['data']['all_datas_npy'], allow_pickle=True)
    dictionary = np.load(config['data']['dictionary_npy'], allow_pickle=True).tolist()
    dict_size = len(dictionary)
    print(f"all_datas: {len(datas)} writers, dict: {dict_size} chars")

    lengths = [[] for _ in range(dict_size - 3)]

    for writer in range(len(datas)):
        if writer % 200 == 0:
            print(f"  writer {writer}/{len(datas)}")
        data = datas[writer]
        line_num = data['line_num']
        for i in range(line_num):
            tag = data[f'{i}tag']
            duration = np.array([0] + data[f'{i}duration'])
            d = np.cumsum(duration)
            for cid in range(len(tag)):
                L = d[cid + 1] - d[cid]
                if L == 0:
                    continue
                ch = tag[cid]
                if ch not in dictionary:
                    continue
                idx = dictionary.index(ch)
                if idx >= dict_size - 3:
                    continue
                lengths[idx].append(int(L))

    # 汇总为平均值
    result = np.zeros(dict_size - 3, dtype=np.int32)
    empty = 0
    for i in range(dict_size - 3):
        if len(lengths[i]) == 0:
            empty += 1
            result[i] = 60  # 默认值
        else:
            result[i] = int(np.median(lengths[i]))  # 用中位数更鲁棒
    # 夹在合理范围 [20, 120]
    result = np.clip(result, 20, 120)

    save_path = Path('data/processed/char_lengths.npy')
    save_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(save_path, result)
    print(f"\nSaved char_lengths.npy -> {save_path}")
    print(f"  total: {len(result)}, empty: {empty}")
    print(f"  min={result.min()}, max={result.max()}, mean={result.mean():.1f}")
    # 抽样看几个常见字
    for ch in ['今', '一', '晚', '家', '好', '藏']:
        if ch in dictionary:
            i = dictionary.index(ch)
            print(f"  '{ch}': median_len={result[i]}")


if __name__ == '__main__':
    main()