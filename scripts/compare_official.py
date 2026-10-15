#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: compare_official.py
@description: 完全照抄官方 train_ddpm.py::test_model 的推理方式，
              一次生成 bs=10 个字，跑 3 轮，用于判断：
              - 如果官方方式也崩 → 模型上限
              - 如果官方方式更好 → 我们的 pipeline 有 bug
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

from src.models.unet_ddpm import DDPM
from src.models.style_encoder import StyleEncoder

EOS, SOS, END = 4052, 4053, 4054


def pad8(tensor, k=8):
    """官方 train_ddpm.py::pad"""
    b, l, _ = tensor.shape
    if l % k == 0:
        return tensor
    pad_l = k - (l % k)
    p = torch.tensor([0, 0, -1]).unsqueeze(0).repeat(b, pad_l, 1).to(tensor.device)
    return torch.cat([tensor, p], dim=1)


def parser_label(label):
    """官方 train_ddpm.py::parser_label"""
    duration = []
    l = 0
    total = len(label)
    for k in range(total):
        if label[k] == EOS:
            duration.append(total - k)
            return duration
        elif label[k] == END:
            l += 1
            duration.append(l)
            l = 0
        else:
            l += 1
    return duration


def get_style_batch_10(char_datas, style_ids, seq_len=1000):
    """官方 train_ddpm.py::get_style_batch_10"""
    bs = len(style_ids)
    batch = np.zeros((bs, seq_len, 3), dtype=np.float32)
    for i in range(bs):
        sid = style_ids[i]
        data = char_datas[sid]
        char_num = len(data)
        char_ids = np.random.choice(char_num, min(15, char_num), replace=False)
        temp = np.zeros((seq_len, 3), dtype=np.float32)
        total = 0
        for cid in char_ids:
            traj = data[cid][1]
            length = len(traj)
            if total + length > seq_len:
                remaining = seq_len - total
                temp[total:total + remaining] = traj[:remaining]
                total = seq_len
                break
            temp[total:total + length] = traj
            total += length
        if total < seq_len:
            temp[total:, 2] = -1.
        batch[i] = temp
    return batch


def get_batch(char_datas, mydict, bs=10):
    """官方 train_ddpm.py::get_batch 完全复刻"""
    labels = []
    lens = []
    lines = np.zeros((bs, 120, 3), dtype=np.float32)
    ids = np.random.choice(len(char_datas), bs, replace=False).tolist()
    char_names = []

    for i in range(bs):
        sid = ids[i]
        char_data = char_datas[sid]
        num = len(char_data)
        j = np.random.randint(0, num)
        char = char_data[j][0]
        char_id = mydict.index(char)
        traj = char_data[j][1]
        char_names.append(char)

        length = traj.shape[0]
        lens.append(length)
        lines[i, :length, :] = traj
        lines[i, length:, 2] = -1.

        label = [char_id] * length
        label[0] = SOS
        label[-1] = END
        labels.append(label)

    lines = torch.from_numpy(lines).float()
    batch = pad8(lines, 8)              # L = batch 内最长 pad 到 8 倍数

    l = batch.shape[1]
    padded_labels = []
    for label in labels:
        if len(label) < l:
            label = label + [EOS] * (l - len(label))
        else:
            label = label[:l]
        padded_labels.append(label)

    durations = [parser_label(lbl) for lbl in padded_labels]
    style_refs = get_style_batch_10(char_datas, ids)
    return batch, padded_labels, durations, style_refs, ids, char_names, lens


def draw_subplot(ax, traj, title, color='b'):
    strokes = np.split(traj, np.where(traj[:, 2] == -1)[0] + 1)
    strokes.pop()
    for s in strokes:
        ax.plot(s[:, 0], -s[:, 1], color + '-', linewidth=1.2)
    ax.set_aspect(1)
    ax.set_title(title, fontsize=9)
    ax.axis('off')


def main():
    with open('configs/default.yaml') as f:
        config = yaml.safe_load(f)
    device = torch.device(config['generation']['device'] if torch.cuda.is_available() else 'cpu')

    print("Loading data...")
    char_datas = np.load(config['data']['datas_npy'], allow_pickle=True).tolist()
    mydict = np.load(config['data']['mydict_npy'], allow_pickle=True).tolist()

    print("Loading models...")
    ddpm = DDPM()
    ddpm.setup_diffusion(device)
    ddpm.load_state_dict(torch.load(config['models']['ddpm_ckpt'], map_location='cpu'))
    ddpm.to(device).eval()

    style_en = StyleEncoder()
    style_en.load_state_dict(torch.load(config['models']['style_encoder_ckpt'], map_location='cpu'))
    style_en.to(device).eval()

    print("Sampling batch (bs=10)...")
    batch, labels, durations, style_refs, ids, char_names, lens = get_batch(
        char_datas, mydict, bs=10
    )
    print(f"  batch L = {batch.shape[1]}")
    print(f"  作者 ids = {ids}")
    print(f"  字符 = {char_names}")
    print(f"  真实长度 = {lens}")
    print(f"  durations = {durations}")

    style_refs_t = torch.from_numpy(style_refs).float().to(device).permute(0, 2, 1)
    with torch.no_grad():
        x, en1, en2, en3 = style_en(style_refs_t)
    style_features = [x, en1, en2, en3]

    print("Generating 3 rounds...")
    fig, axes = plt.subplots(4, 10, figsize=(22, 9))

    # 第 0 行：真值
    real_batch = batch.cpu().numpy()
    for i in range(10):
        t = real_batch[i].copy()
        t[:, 0:2] = np.cumsum(t[:, 0:2], axis=0)  # 相对 → 绝对
        draw_subplot(axes[0, i], t, f"real: {char_names[i]}", color='r')

    # 第 1~3 行：官方 generate（3 轮）
    for r in range(3):
        with torch.no_grad():
            traj = ddpm.generate(labels, durations, style_features).permute(0, 2, 1)
        traj[:, :, 2] = torch.sgn(traj[:, :, 2])
        traj = traj.cpu().numpy()
        for i in range(10):
            t = traj[i].copy()
            t[:, 0:2] = np.cumsum(t[:, 0:2], axis=0)
            draw_subplot(axes[r + 1, i], t, f"gen#{r+1}: {char_names[i]}")

    plt.tight_layout()
    plt.savefig('compare_official.png', dpi=120, bbox_inches='tight')
    plt.close()
    print("Saved compare_official.png")


if __name__ == '__main__':
    main()