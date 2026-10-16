#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 04_infer_single_page.py
@description: 单页生成 —— 连续索引 0~1019 作为 author_id。
              - 用 writer_id_map.json 反查原编号
              - 从原作者测试页 .wptt 提取文本（也可 --text 覆盖）
              - 批处理单字生成
              - 页面级风格：真实行距、段首缩进、行倾斜
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import yaml

from src.models.unet_ddpm import DDPM
from src.models.style_encoder import StyleEncoder
from src.models.box_lstm import BoxLstm
from src.utils.draw_utils import draw_page
from src.datasets.load_wptt import load_wptt

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

EOS, SOS, END = 4052, 4053, 4054
CHAR_SEQ_LEN = 120
TYPICAL_CHAR_H = 0.5


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=str, default='configs/default.yaml')
    p.add_argument('--author_id', type=int, required=True,
                   help="连续索引 0~1019（对应 datas_full.npy 的 key）")
    p.add_argument('--text', type=str, default=None,
                   help="可选：手动指定文本，覆盖测试页。用 \\n 表示换行。")
    p.add_argument('--output_dir', type=str, default='outputs/pages')
    return p.parse_args()


# ============================================================
# 反查原编号
# ============================================================
def load_writer_map(map_path='data/processed/writer_id_map.json'):
    with open(map_path, 'r', encoding='utf-8') as f:
        m = json.load(f)
    return {int(k): v for k, v in m['cont_to_orig'].items()}


# ============================================================
# 从测试页 .wptt 提取文本 + 每行缩进
# ============================================================
def load_test_page_text(orig_id, split_yaml_path='configs/author_split.yaml'):
    """返回 lines_info: [{'text':..., 'indent_chars':...}, ...]"""
    with open(split_yaml_path, 'r', encoding='utf-8') as f:
        author_split = yaml.safe_load(f)

    entry = author_split.get(orig_id) or author_split.get(str(orig_id))
    if entry is None:
        raise KeyError(f"orig_id {orig_id} 不在 author_split.yaml 里")

    test_page = entry.get('test_page')
    if not test_page:
        raise ValueError(f"orig_id {orig_id} 没有 test_page")
    if not os.path.exists(test_page):
        raise FileNotFoundError(f"test_page 不存在: {test_page}")

    wptt = load_wptt(test_page)
    strokes = wptt['strokes']

    lines_info = []
    line_widths = []
    for line in wptt['lines']:
        text = line.get('text', '')
        if not text or not text.strip():
            continue
        idx_list = line['stroke_indices']
        line_strokes = [strokes[i] for i in idx_list if 0 <= i < len(strokes)]
        if not line_strokes:
            continue
        pts = np.concatenate(line_strokes, axis=0)
        x_min = float(pts[:, 0].min())
        y_min = float(pts[:, 1].min())
        y_max = float(pts[:, 1].max())
        char_height = float(y_max - y_min)
        line_width = float(pts[:, 0].max() - pts[:, 0].min())
        lines_info.append({
            'text': text,
            'x_min': x_min,
            'char_height': char_height,
            'line_width': line_width,
            'n_chars': len(text),
        })
        line_widths.append(line_width / max(len(text), 1))

    if not lines_info:
        raise ValueError(f"{test_page} 没有有效行")

    # 用字宽算缩进
    avg_char_w_pixels = float(np.mean(line_widths)) if line_widths else 30.0
    x_mins = [info['x_min'] for info in lines_info]
    min_x = min(x_mins)

    for info in lines_info:
        info['indent_chars'] = (info['x_min'] - min_x) / avg_char_w_pixels

    logger.info(f"从 {test_page} 提取 {len(lines_info)} 行文本")
    for info in lines_info[:5]:
        logger.info(f"  x_min={info['x_min']:.1f}, indent_chars={info['indent_chars']:.2f}")

    return lines_info, test_page


# ============================================================
# 布局生成
# ============================================================
def predict_boxes(box_lstm, text, dictionary, char_boxes, device):
    box_lstm.eval()
    len_tag = len(text)
    ref_seq = np.zeros((1, len_tag, 6), dtype=np.float32)
    for i, ch in enumerate(text):
        if ch not in dictionary:
            ref_seq[0, i] = [0.5, 0.05, 0.5, 0.05, 0.5, 0.05]
            continue
        idx = dictionary.index(ch)
        ref_seq[0, i, 0] = char_boxes[idx]['h'][0]
        ref_seq[0, i, 1] = char_boxes[idx]['h'][1]
        ref_seq[0, i, 2] = char_boxes[idx]['w'][0]
        ref_seq[0, i, 3] = char_boxes[idx]['w'][1]
        ref_seq[0, i, 4] = char_boxes[idx]['center'][0]
        ref_seq[0, i, 5] = char_boxes[idx]['center'][1]
    ref_seq_t = torch.from_numpy(ref_seq).float().to(device)

    with torch.no_grad():
        box_list = box_lstm.get_box(ref_seq_t)
    full_box = torch.cat(box_list).detach().cpu().numpy()[:, 0, :]

    bboxes = []
    x_left = 0.0
    for i in range(full_box.shape[0]):
        h, w, c, d = full_box[i]
        ymax = c + h / 2
        ymin = c - h / 2
        xmin = x_left + d
        xmax = xmin + w
        bboxes.append([ymax, ymin, xmax, xmin])
        x_left = xmax
    return bboxes


def build_real_lengths(writer_chars):
    real_lens = {}
    if not writer_chars:
        return real_lens
    for item in writer_chars:
        char, traj = item[0], item[1]
        real_lens.setdefault(char, []).append(len(traj))
    return {c: int(np.median(v)) for c, v in real_lens.items()}


# ============================================================
# 批处理单字生成
# ============================================================
def gen_char_trajs(text, ddpm, mydict, style_features, device,
                   real_lens=None, L: int = 120, default_len: int = 57):
    ddpm.eval()
    labels, durations, char_lens = [], [], []
    for ch in text:
        if ch not in mydict:
            logger.warning(f"'{ch}' not in mydict, skip")
            continue
        cid = mydict.index(ch)
        char_len = real_lens.get(ch, default_len) if real_lens else default_len
        char_len = max(8, min(char_len, L - 8))
        pad_len = L - char_len
        labels.append([SOS] + [cid] * (char_len - 2) + [END] + [EOS] * pad_len)
        durations.append([char_len, pad_len])
        char_lens.append(char_len)

    if not labels:
        return []

    bs = len(labels)
    style_bs = [t.repeat(bs, 1, 1) for t in style_features]

    with torch.no_grad():
        trajs = ddpm.generate(labels, durations, style_bs)

    out = []
    for i in range(bs):
        traj = trajs[i].permute(1, 0).cpu().numpy()[:char_lens[i]]
        traj[:, 2] = np.sign(traj[:, 2])
        traj[:, 0:2] = np.cumsum(traj[:, 0:2], axis=0)
        out.append(traj)
    return out


def put_char(traj, bbox):
    ymax, ymin, xmax, xmin = bbox
    traj = traj.copy().astype(np.float32)
    xr = max(traj[:, 0].max() - traj[:, 0].min(), 1e-6)
    yr = max(traj[:, 1].max() - traj[:, 1].min(), 1e-6)
    traj[:, 0] = (traj[:, 0] - traj[:, 0].min()) / xr * (xmax - xmin) + xmin
    traj[:, 1] = (traj[:, 1] - traj[:, 1].min()) / yr * (ymax - ymin) + ymin
    return traj


# ============================================================
# 主流程
# ============================================================
def main():
    args = parse_args()
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    device = torch.device(config['generation']['device']
                          if torch.cuda.is_available() else 'cpu')
    cont_idx = args.author_id
    output_dir = Path(args.output_dir) / f"author_{cont_idx:04d}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 反查原编号
    cont_to_orig = load_writer_map()
    orig_id = cont_to_orig.get(cont_idx)
    if orig_id is None:
        logger.error(f"连续索引 {cont_idx} 不在映射表里")
        return
    logger.info(f"author_id={cont_idx} -> 原编号={orig_id}")

    logger.info("Loading models...")
    ddpm = DDPM()
    ddpm.setup_diffusion(device)
    ddpm.load_state_dict(torch.load(config['models']['ddpm_ckpt'], map_location='cpu'))
    ddpm.to(device).eval()

    style_en = StyleEncoder()
    style_en.load_state_dict(torch.load(config['models']['style_encoder_ckpt'], map_location='cpu'))
    style_en.to(device).eval()

    box_lstm = BoxLstm(device)
    box_lstm.load_state_dict(torch.load(config['models']['box_lstm_ckpt'], map_location='cpu'))
    box_lstm.to(device).eval()

    dictionary = np.load(config['data']['dictionary_npy'], allow_pickle=True).tolist()
    mydict = np.load(config['data']['mydict_npy'], allow_pickle=True).tolist()
    char_boxes = np.load(config['data']['char_boxes_npy'], allow_pickle=True)

    datas_dict = np.load('data/processed/datas_full.npy', allow_pickle=True).item()
    logger.info(f"Loaded datas_full: {len(datas_dict)} writers")

    # 风格特征（用连续索引）
    cache = Path(config['paths']['style_cache_dir']) / f"{cont_idx}.pt"
    if not cache.exists():
        logger.error(f"Style cache missing: {cache}. Run 01_prepare_data.py first.")
        return
    style_features = [t.to(device) for t in torch.load(cache, map_location='cpu')]
    logger.info(f"Loaded style features: {cache}")

    # 真实字长（用连续索引）
    writer_chars = datas_dict.get(cont_idx, [])
    real_lens = build_real_lengths(writer_chars)
    logger.info(f"Real lengths: {len(real_lens)} chars for writer {cont_idx}")

    # 页面级统计（用原编号）
    page_stats_path = Path('data/processed/page_stats.npy')
    page_stats = np.load(page_stats_path, allow_pickle=True).item() if page_stats_path.exists() else {}

    if orig_id in page_stats:
        s = page_stats[orig_id]
        y_spacing_ratio = s['y_spacing_mean']
        y_spacing_std = max(s['y_spacing_std'], 0.05)
        line_slope = s['slope_median']
        logger.info(f"Page stats (orig {orig_id}): "
                    f"y_spacing={y_spacing_ratio:.3f}±{y_spacing_std:.3f}, "
                    f"slope={line_slope:.4f}")
    else:
        logger.warning(f"No page stats for orig {orig_id}, use defaults")
        y_spacing_ratio, y_spacing_std, line_slope = 1.85, 0.15, 0.0

    y_step_base = y_spacing_ratio * TYPICAL_CHAR_H
    y_step_std = y_spacing_std * TYPICAL_CHAR_H

    # 平均字宽（用于缩进换单位）
    valid_widths = [cb['w'][0] for cb in char_boxes
                    if isinstance(cb, dict) and 'w' in cb and cb['w'][0] > 0]
    avg_char_w = float(np.mean(valid_widths)) if valid_widths else 0.5
    logger.info(f"avg char width={avg_char_w:.3f}")

    # 文本来源
    if args.text:
        raw_lines = [l.strip() for l in args.text.replace('\\n', '\n').split('\n') if l.strip()]
        lines_info = [{'text': t, 'indent_chars': 0.0} for t in raw_lines]
        src_info = "命令行 --text"
    else:
        try:
            lines_info, test_page = load_test_page_text(orig_id)
            src_info = f"测试页 {test_page}"
        except Exception as e:
            logger.error(f"无法从测试页提取文本: {e}")
            return
    logger.info(f"文本来源: {src_info}, 共 {len(lines_info)} 行")

    # 逐行生成
    full_page = np.array([[0.0, 0.0, -1.0]], dtype=np.float32)
    y_offset = 0.0
    page_t0 = time.time()

    for li, info in enumerate(lines_info):
        line_text = info['text']
        line_indent = info['indent_chars'] * avg_char_w
        line_t0 = time.time()
        logger.info(f"[{li + 1}/{len(lines_info)}] indent={line_indent:.3f} | "
                    f"{line_text[:30]}{'...' if len(line_text) > 30 else ''}")
        try:
            boxes = predict_boxes(box_lstm, line_text, dictionary, char_boxes, device)
            char_trajs = gen_char_trajs(
                line_text, ddpm, mydict, style_features, device,
                real_lens=real_lens, L=CHAR_SEQ_LEN,
            )

            y_step = float(np.random.normal(y_step_base, y_step_std))
            y_step = max(y_step, 0.6)

            line_traj = np.array([[0.0, 0.0, -1.0]], dtype=np.float32)
            for i, ct in enumerate(char_trajs):
                if i >= len(boxes):
                    break
                scaled = put_char(ct, boxes[i])

                x_center = (boxes[i][2] + boxes[i][3]) / 2
                scaled[:, 1] += line_slope * TYPICAL_CHAR_H * x_center
                scaled[:, 0] += line_indent
                scaled[:, 1] += y_offset

                line_traj = np.concatenate([line_traj, scaled], axis=0)

            full_page = np.concatenate([full_page, line_traj[1:]], axis=0)
        except Exception as e:
            logger.error(f"Line {li} failed: {e}")
            import traceback
            traceback.print_exc()
            continue

        y_offset += y_step
        logger.info(f"  行 {li + 1} 耗时 {time.time() - line_t0:.1f}s, y_offset={y_offset:.3f}")

    logger.info(f"整页耗时 {time.time() - page_t0:.1f}s")

    # 保存
    png_path = output_dir / "generated_page.png"
    npy_path = output_dir / "generated_page.npy"
    draw_page(full_page[1:], str(png_path))
    np.save(npy_path, full_page[1:])
    logger.info(f"Saved PNG -> {png_path}")
    logger.info(f"Saved NPY -> {npy_path}")


if __name__ == '__main__':
    main()