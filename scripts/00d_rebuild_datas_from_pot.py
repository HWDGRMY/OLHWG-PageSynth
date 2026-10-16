#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 00d_rebuild_datas_from_pot.py
@description: 从 .pot 文件重建 datas_full.npy。
              连续索引 0~1019 作为 key，同时保存原编号映射表。
"""

import os
import re
import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from tqdm import tqdm

sys.path.append('./src/datasets')
from load_pot import load_pot_trajs


POT_DIRS = [
    'data/raw/Pot1.0Train',
    'data/raw/Pot1.0Test_2',
    'data/raw/Pot1.1Train_2',
    'data/raw/Pot1.1Test_2',
    'data/raw/Pot1.2Train_2',
    'data/raw/Pot1.2Test',
]


def main():
    # ---------- 1. 按原编号收集 ----------
    datas_by_orig_id = {}

    for d in POT_DIRS:
        if not os.path.exists(d):
            print(f"跳过不存在的目录: {d}")
            continue
        files = sorted([f for f in os.listdir(d) if f.endswith('.pot')])
        print(f"\n处理 {d} ({len(files)} 个文件)")
        for f in tqdm(files):
            m = re.findall(r'\d+', f)
            if not m:
                continue
            orig_id = int(m[0])
            if orig_id in datas_by_orig_id:
                continue
            path = os.path.join(d, f)
            try:
                samples = load_pot_trajs(path)
                if len(samples) > 0:
                    datas_by_orig_id[orig_id] = samples
            except Exception as e:
                print(f"  解析 {path} 失败: {e}")

    print(f"\n成功解析: {len(datas_by_orig_id)} 位作者（按原编号）")

    if len(datas_by_orig_id) == 0:
        print("没有解析到任何作者，退出。")
        return

    # ---------- 2. 按原编号排序，映射到连续 0~N-1 ----------
    orig_ids = sorted(datas_by_orig_id.keys())
    orig_to_cont = {int(o): i for i, o in enumerate(orig_ids)}
    cont_to_orig = {i: int(o) for i, o in enumerate(orig_ids)}

    # 连续索引的 datas
    datas_cont = {}
    for cont_idx, orig in enumerate(orig_ids):
        datas_cont[cont_idx] = datas_by_orig_id[orig]

    # ---------- 3. 统计信息 ----------
    counts = [len(v) for v in datas_cont.values()]
    print(f"\n连续索引范围: 0 ~ {len(datas_cont) - 1}")
    print(f"字数: min={min(counts)}, max={max(counts)}, mean={np.mean(counts):.0f}")
    print(f"低于 500 字的作者: {sum(1 for c in counts if c < 500)}")

    print(f"\n原编号示例:")
    for cont_idx in [0, 1, 2, 420, 500, 800, 1000, len(datas_cont) - 1]:
        if cont_idx in cont_to_orig:
            print(f"  连续索引 {cont_idx}  <-  原编号 {cont_to_orig[cont_idx]}")

    # ---------- 4. 保存 ----------
    save_path = Path('data/processed/datas_full.npy')
    np.save(save_path, np.array(datas_cont, dtype=object), allow_pickle=True)
    print(f"\nSaved: {save_path}")

    # 映射表（json 方便查看）
    map_path = Path('data/processed/writer_id_map.json')
    with open(map_path, 'w', encoding='utf-8') as f:
        json.dump({
            'orig_to_cont': {str(k): v for k, v in orig_to_cont.items()},
            'cont_to_orig': {str(k): v for k, v in cont_to_orig.items()},
        }, f, ensure_ascii=False, indent=2)
    print(f"Saved: {map_path}")


if __name__ == '__main__':
    main()