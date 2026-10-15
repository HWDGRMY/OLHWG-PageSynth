#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 00_build_char_boxes.py
@description: 从 all_datas.npy 统计每个字符的边界框先验分布。
"""
import os, sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import yaml


def cal_box(traj):
    x_max, x_min = np.max(traj[:, 0]), np.min(traj[:, 0])
    y_max, y_min = np.max(traj[:, 1]), np.min(traj[:, 1])
    return y_max - y_min, x_max - x_min, (y_max + y_min) / 2


def gaussian(arr):
    arr = np.array(arr)
    return float(np.mean(arr)), float(np.std(arr))


def main():
    with open('configs/default.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    datas = np.load(config['data']['all_datas_npy'], allow_pickle=True)
    dictionary = np.load(config['data']['dictionary_npy'], allow_pickle=True).tolist()
    dict_size = len(dictionary)
    print(f"all_datas: {len(datas)} writers, dict: {dict_size} chars")

    boxes = [{'h': [], 'w': [], 'center': []} for _ in range(dict_size - 3)]
    for writer in range(len(datas)):
        if writer % 200 == 0:
            print(f"  writer {writer}/{len(datas)}")
        data = datas[writer]
        line_num = data['line_num']
        for i in range(line_num):
            line = data[f'{i}'].copy()
            line[:, 0:2] = np.cumsum(line[:, 0:2], axis=0)
            tag = data[f'{i}tag']
            duration = np.array([0] + data[f'{i}duration'])
            d = np.cumsum(duration)
            for cid in range(len(tag)):
                ct = line[d[cid]:d[cid+1], :]
                if len(ct) == 0:
                    continue
                ch = tag[cid]
                if ch not in dictionary:
                    continue
                idx = dictionary.index(ch)
                if idx >= dict_size - 3:
                    continue
                h, w, c = cal_box(ct)
                boxes[idx]['h'].append(h)
                boxes[idx]['w'].append(w)
                boxes[idx]['center'].append(c)

    result = []
    empty = 0
    for i in range(dict_size - 3):
        cb = boxes[i]
        if len(cb['h']) < 5:
            empty += 1
            result.append({'h': [0.5, 0.1], 'w': [0.5, 0.1], 'center': [0.5, 0.1]})
            continue
        result.append({
            'h': list(gaussian(cb['h'])),
            'w': list(gaussian(cb['w'])),
            'center': list(gaussian(cb['center'])),
        })

    save_path = Path(config['data']['char_boxes_npy'])
    save_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(save_path, np.array(result, dtype=object), allow_pickle=True)
    print(f"Saved char_boxes.npy -> {save_path}")
    print(f"  total: {len(result)}, empty(<5 samples): {empty}")


if __name__ == '__main__':
    main()
