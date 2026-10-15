#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 04_infer_single_page.py
@description: 单页生成 —— 严格对齐官方 train_ddpm.py::test_model 的推理方式。
              - char_len 用该字在 datas 里的真实长度
              - L 固定 120（与训练一致）
              - style_ref 从 datas 里现场提取
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
import torch
import yaml

from src.models.unet_ddpm import DDPM
from src.models.style_encoder import StyleEncoder
from src.models.box_lstm import BoxLstm
from src.utils.draw_utils import draw_page

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

EOS, SOS, END = 4052, 4053, 4054
CHAR_SEQ_LEN = 120


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=str, default='configs/default.yaml')
    p.add_argument('--author_id', type=int, required=True)
    p.add_argument('--text', type=str, default=None,
                   help="使用 \\n 表示换行。默认会用一段测试文本。")
    p.add_argument('--output_dir', type=str, default='outputs/pages')
    return p.parse_args()


def predict_boxes(box_lstm, text, dictionary, char_boxes, device):
    """用 BoxLstm 预测一行中每个字的边界框。"""
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


def build_real_lengths(datas, writer_id: int) -> dict:
    """建立 "字 → 该作者写的真实长度" 映射（取中位数）。"""
    real_lens = {}
    if writer_id - 1 >= len(datas):
        return real_lens
    for item in datas[writer_id - 1]:
        char, traj = item[0], item[1]
        real_lens.setdefault(char, []).append(len(traj))
    return {c: int(np.median(v)) for c, v in real_lens.items()}


def gen_char_trajs(text, ddpm, mydict, style_features, device,
                   real_lens: dict, L: int = 120, default_len: int = 57):
    """逐字生成。char_len 用该字的真实长度，L 固定 120。

    完全对齐官方 train_ddpm.py::test_model 的推理方式。
    """
    ddpm.eval()
    out = []
    for ch in text:
        if ch not in mydict:
            logger.warning(f"'{ch}' not in mydict, skip")
            out.append(np.zeros((1, 3), dtype=np.float32))
            continue

        cid = mydict.index(ch)
        char_len = real_lens.get(ch, default_len)
        char_len = max(8, min(char_len, L - 8))
        pad_len = L - char_len

        label = [SOS] + [cid] * (char_len - 2) + [END] + [EOS] * pad_len
        duration = [[char_len, pad_len]]

        with torch.no_grad():
            traj = ddpm.generate([label], duration, style_features)

        traj = traj.permute(0, 2, 1)[0].cpu().numpy()[:char_len]
        traj[:, 2] = np.sign(traj[:, 2])
        traj[:, 0:2] = np.cumsum(traj[:, 0:2], axis=0)
        out.append(traj)
    return out


def put_char(traj, bbox):
    """把单字轨迹缩放到指定边界框内。"""
    ymax, ymin, xmax, xmin = bbox
    traj = traj.copy().astype(np.float32)
    xr = max(traj[:, 0].max() - traj[:, 0].min(), 1e-6)
    yr = max(traj[:, 1].max() - traj[:, 1].min(), 1e-6)
    traj[:, 0] = (traj[:, 0] - traj[:, 0].min()) / xr * (xmax - xmin) + xmin
    traj[:, 1] = (traj[:, 1] - traj[:, 1].min()) / yr * (ymax - ymin) + ymin
    return traj


def main():
    args = parse_args()
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    device = torch.device(config['generation']['device']
                          if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir) / f"author_{args.author_id:04d}"
    output_dir.mkdir(parents=True, exist_ok=True)

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

    # 加载各种数据
    dictionary = np.load(config['data']['dictionary_npy'], allow_pickle=True).tolist()
    mydict = np.load(config['data']['mydict_npy'], allow_pickle=True).tolist()
    char_boxes = np.load(config['data']['char_boxes_npy'], allow_pickle=True)
    datas = np.load(config['data']['datas_npy'], allow_pickle=True).tolist()

    # 加载缓存风格
    cache = Path(config['paths']['style_cache_dir']) / f"{args.author_id}.pt"
    if not cache.exists():
        logger.error(f"Style cache missing: {cache}. Run 01_prepare_data.py first.")
        return
    style_features = [t.to(device) for t in torch.load(cache, map_location='cpu')]
    logger.info(f"Loaded style features: {cache}")

    # 该字的真实长度表
    real_lens = build_real_lengths(datas, args.author_id)
    logger.info(f"Real lengths built: {len(real_lens)} chars for writer {args.author_id}")

    # 准备文本
    if args.text is None:
        text = "今天天气真好\n我想去公园散步\n晚上回家吃饭"
    else:
        text = args.text.replace('\\n', '\n')
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    logger.info(f"Generating {len(lines)} line(s).")

    # 逐行生成
    full_page = np.array([[0.0, 0.0, -1.0]], dtype=np.float32)
    y_offset = 0.0
    line_height = config['generation'].get('line_height', 1.2)

    for li, line_text in enumerate(lines):
        logger.info(f"[{li+1}/{len(lines)}] {line_text}")
        try:
            boxes = predict_boxes(box_lstm, line_text, dictionary, char_boxes, device)
            char_trajs = gen_char_trajs(
                line_text, ddpm, mydict, style_features, device,
                real_lens=real_lens, L=CHAR_SEQ_LEN,
            )
            line_traj = np.array([[0.0, 0.0, -1.0]], dtype=np.float32)
            for i, ct in enumerate(char_trajs):
                if i >= len(boxes):
                    break
                scaled = put_char(ct, boxes[i])
                scaled[:, 1] += y_offset
                line_traj = np.concatenate([line_traj, scaled], axis=0)
            full_page = np.concatenate([full_page, line_traj[1:]], axis=0)
        except Exception as e:
            logger.error(f"Line {li} failed: {e}")
            import traceback
            traceback.print_exc()
            continue
        y_offset += line_height

    # 保存
    png_path = output_dir / "generated_page.png"
    npy_path = output_dir / "generated_page.npy"
    draw_page(full_page[1:], str(png_path))
    np.save(npy_path, full_page[1:])
    logger.info(f"Saved PNG -> {png_path}")
    logger.info(f"Saved NPY -> {npy_path}")


if __name__ == '__main__':
    main()