#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 05_run_batch_1020.py
@description: 批量生成 1020 位作者的伪造手写整页。
              - 每个 worker 独立加载模型
              - 断点续传
              - n_workers 可调
"""

import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import yaml
from tqdm import tqdm

from src.models.unet_ddpm import DDPM
from src.models.style_encoder import StyleEncoder
from src.models.box_lstm import BoxLstm
from src.datasets.load_wptt import load_wptt

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(process)d] %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

EOS, SOS, END = 4052, 4053, 4054
CHAR_SEQ_LEN = 120
TYPICAL_CHAR_H = 0.5
INDENT_CHARS = 2.0


# ============================================================
# 全局 worker 状态（每个子进程独立一份）
# ============================================================
_WORKER = {}


def _load_datas_dict(path):
    """兼容 .item()/list 两种保存方式，统一返回 {cont_idx: data}。"""
    arr = np.load(path, allow_pickle=True)
    try:
        obj = arr.item()
    except ValueError:
        obj = arr
    if isinstance(obj, dict):
        return {int(k): v for k, v in obj.items()}
    if isinstance(obj, (list, tuple, np.ndarray)):
        return {i: v for i, v in enumerate(obj)}
    raise ValueError(f"Unexpected datas type: {type(obj)}")


def _init_worker(config_path):
    """子进程启动时执行一次：加载模型和数据到 _WORKER。"""
    _WORKER.clear()

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    _WORKER['config'] = config

    device = torch.device(config['generation']['device']
                          if torch.cuda.is_available() else 'cpu')
    _WORKER['device'] = device

    ddpm = DDPM()
    ddpm.setup_diffusion(device)
    ddpm.load_state_dict(torch.load(config['models']['ddpm_ckpt'], map_location='cpu'))
    ddpm.to(device).eval()
    _WORKER['ddpm'] = ddpm

    style_en = StyleEncoder()
    style_en.load_state_dict(torch.load(config['models']['style_encoder_ckpt'], map_location='cpu'))
    style_en.to(device).eval()
    _WORKER['style_en'] = style_en

    box_lstm = BoxLstm(device)
    box_lstm.load_state_dict(torch.load(config['models']['box_lstm_ckpt'], map_location='cpu'))
    box_lstm.to(device).eval()
    _WORKER['box_lstm'] = box_lstm

    _WORKER['dictionary'] = np.load(config['data']['dictionary_npy'], allow_pickle=True).tolist()
    _WORKER['mydict'] = np.load(config['data']['mydict_npy'], allow_pickle=True).tolist()
    _WORKER['char_boxes'] = np.load(config['data']['char_boxes_npy'], allow_pickle=True)
    _WORKER['datas_dict'] = _load_datas_dict('data/processed/datas_full.npy')

    ps_path = Path('data/processed/page_stats.npy')
    _WORKER['page_stats'] = np.load(ps_path, allow_pickle=True).item() if ps_path.exists() else {}

    with open('configs/author_split.yaml', 'r', encoding='utf-8') as f:
        _WORKER['author_split'] = yaml.safe_load(f)

    with open('data/processed/writer_id_map.json', 'r', encoding='utf-8') as f:
        wmap = json.load(f)
    _WORKER['cont_to_orig'] = wmap['cont_to_orig']

    logger.info(f"Worker init done on {device}")


# ============================================================
# 单字/布局核心函数
# ============================================================
def _predict_boxes(text, dictionary, char_boxes, box_lstm, device):
    box_lstm.eval()
    len_tag = len(text)
    ref_seq = np.zeros((1, len_tag, 6), dtype=np.float32)
    for i, ch in enumerate(text):
        if ch not in dictionary:
            ref_seq[0, i] = [0.5, 0.05, 0.5, 0.05, 0.5, 0.05]
            continue
        idx = dictionary.index(ch)
        ref_seq[0, i] = [char_boxes[idx]['h'][0], char_boxes[idx]['h'][1],
                         char_boxes[idx]['w'][0], char_boxes[idx]['w'][1],
                         char_boxes[idx]['center'][0], char_boxes[idx]['center'][1]]
    ref_seq_t = torch.from_numpy(ref_seq).float().to(device)
    with torch.no_grad():
        box_list = box_lstm.get_box(ref_seq_t)
    full_box = torch.cat(box_list).detach().cpu().numpy()[:, 0, :]

    bboxes = []
    x_left = 0.0
    for i in range(full_box.shape[0]):
        h, w, c, d = full_box[i]
        ymax, ymin = c + h / 2, c - h / 2
        xmin = x_left + d
        xmax = xmin + w
        bboxes.append([ymax, ymin, xmax, xmin])
        x_left = xmax
    return bboxes


def _build_real_lengths(datas_dict, cont_idx):
    real_lens = {}
    if cont_idx not in datas_dict:
        return real_lens
    for item in datas_dict[cont_idx]:
        char, traj = item[0], item[1]
        real_lens.setdefault(char, []).append(len(traj))
    return {c: int(np.median(v)) for c, v in real_lens.items()}


def _gen_char_trajs_batch(text, ddpm, mydict, style_features, device,
                          real_lens, L=CHAR_SEQ_LEN, default_len=57):
    ddpm.eval()
    labels, durations, char_lens = [], [], []
    for ch in text:
        if ch not in mydict:
            continue
        cid = mydict.index(ch)
        char_len = real_lens.get(ch, default_len)
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


def _put_char(traj, bbox):
    ymax, ymin, xmax, xmin = bbox
    traj = traj.copy().astype(np.float32)
    xr = max(traj[:, 0].max() - traj[:, 0].min(), 1e-6)
    yr = max(traj[:, 1].max() - traj[:, 1].min(), 1e-6)
    traj[:, 0] = (traj[:, 0] - traj[:, 0].min()) / xr * (xmax - xmin) + xmin
    traj[:, 1] = (traj[:, 1] - traj[:, 1].min()) / yr * (ymax - ymin) + ymin
    return traj


# ============================================================
# 单作者处理
# ============================================================
def _process_one_author(task):
    """处理单个作者。返回 (cont_idx, orig_id, ok, msg, elapsed_sec)。"""
    cont_idx, orig_id, test_page, output_dir = task

    output_dir = Path(output_dir)
    save_npy = output_dir / f"author_{cont_idx:04d}.npy"
    save_meta = output_dir / f"author_{cont_idx:04d}.json"

    # 断点续传
    if save_npy.exists() and save_meta.exists():
        return (cont_idx, orig_id, True, "skipped", 0.0)

    t0 = time.time()
    try:
        device = _WORKER['device']

        # 1) 风格特征
        cache = Path(_WORKER['config']['paths']['style_cache_dir']) / f"{cont_idx}.pt"
        if not cache.exists():
            return (cont_idx, orig_id, False, "style cache missing", time.time() - t0)
        style_features = [t.to(device) for t in torch.load(cache, map_location='cpu')]

        # 2) 从测试页提取每行文本 + 缩进
        if not os.path.exists(test_page):
            return (cont_idx, orig_id, False, f"test_page missing: {test_page}", time.time() - t0)
        wptt = load_wptt(test_page)
        strokes = wptt['strokes']
        lines_info = []
        for line in wptt['lines']:
            t = line.get('text', '')
            if not t or not t.strip():
                continue
            idx_list = line['stroke_indices']
            line_strokes = [strokes[i] for i in idx_list if 0 <= i < len(strokes)]
            if not line_strokes:
                continue
            pts = np.concatenate(line_strokes, axis=0)
            x_min = float(pts[:, 0].min())
            line_width = float(pts[:, 0].max() - pts[:, 0].min())
            char_w_px = line_width / max(len(t), 1)
            lines_info.append({'text': t, 'x_min': x_min, 'char_w_px': char_w_px})

        if not lines_info:
            return (cont_idx, orig_id, False, "no valid lines", time.time() - t0)

        min_x = min(info['x_min'] for info in lines_info)
        avg_w_px = float(np.mean([info['char_w_px'] for info in lines_info]))
        for info in lines_info:
            info['indent_chars'] = (info['x_min'] - min_x) / max(avg_w_px, 1.0)

        # 3) 页面级统计
        ps = _WORKER['page_stats']
        if orig_id in ps:
            s = ps[orig_id]
            y_spacing_ratio, y_spacing_std = s['y_spacing_mean'], max(s['y_spacing_std'], 0.05)
            line_slope = s['slope_median']
        else:
            y_spacing_ratio, y_spacing_std, line_slope = 1.85, 0.15, 0.0

        y_step_base = y_spacing_ratio * TYPICAL_CHAR_H
        y_step_std = y_spacing_std * TYPICAL_CHAR_H

        # 4) 逐行生成
        real_lens = _build_real_lengths(_WORKER['datas_dict'], cont_idx)
        valid_widths = [cb['w'][0] for cb in _WORKER['char_boxes']
                        if isinstance(cb, dict) and 'w' in cb and cb['w'][0] > 0]
        avg_char_w = float(np.mean(valid_widths)) if valid_widths else 0.5

        full_page = np.array([[0.0, 0.0, -1.0]], dtype=np.float32)
        y_offset = 0.0
        for info in lines_info:
            line_text = info['text']
            line_indent = info['indent_chars'] * avg_char_w

            boxes = _predict_boxes(line_text, _WORKER['dictionary'], _WORKER['char_boxes'],
                                   _WORKER['box_lstm'], device)
            char_trajs = _gen_char_trajs_batch(line_text, _WORKER['ddpm'], _WORKER['mydict'],
                                                style_features, device, real_lens)
            y_step = max(float(np.random.normal(y_step_base, y_step_std)), 0.6)

            line_traj = np.array([[0.0, 0.0, -1.0]], dtype=np.float32)
            for i, ct in enumerate(char_trajs):
                if i >= len(boxes):
                    break
                scaled = _put_char(ct, boxes[i])
                x_center = (boxes[i][2] + boxes[i][3]) / 2
                scaled[:, 1] += line_slope * TYPICAL_CHAR_H * x_center
                scaled[:, 0] += line_indent
                scaled[:, 1] += y_offset
                line_traj = np.concatenate([line_traj, scaled], axis=0)

            full_page = np.concatenate([full_page, line_traj[1:]], axis=0)
            y_offset += y_step

        # 5) 保存
        output_dir.mkdir(parents=True, exist_ok=True)
        np.save(save_npy, full_page[1:])
        with open(save_meta, 'w', encoding='utf-8') as f:
            json.dump({
                'cont_idx': cont_idx,
                'orig_id': orig_id,
                'test_page': test_page,
                'num_lines': len(lines_info),
                'num_chars': int(sum(len(info['text']) for info in lines_info)),
                'elapsed_sec': round(time.time() - t0, 2),
            }, f, ensure_ascii=False, indent=2)

        return (cont_idx, orig_id, True, "ok", time.time() - t0)
    except Exception as e:
        import traceback
        return (cont_idx, orig_id, False,
                f"{type(e).__name__}: {e}\n{traceback.format_exc()[:400]}",
                time.time() - t0)


# ============================================================
# 主流程
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/default.yaml')
    parser.add_argument('--n_workers', type=int, default=2,
                        help="并行 worker 数。GPU 已饱和时增加无效，1~3 建议")
    parser.add_argument('--output_dir', default='outputs/fake_pages')
    parser.add_argument('--author_ids', type=str, default=None,
                        help="逗号分隔的连续索引，例如 '0,1,2'；不填则全部 1020 位")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # 构造任务列表
    with open('data/processed/writer_id_map.json', 'r', encoding='utf-8') as f:
        wmap = json.load(f)
    cont_to_orig = {int(k): int(v) for k, v in wmap['cont_to_orig'].items()}

    with open('configs/author_split.yaml', 'r', encoding='utf-8') as f:
        author_split = yaml.safe_load(f)

    if args.author_ids:
        cont_ids = [int(x.strip()) for x in args.author_ids.split(',') if x.strip()]
    else:
        cont_ids = sorted(cont_to_orig.keys())

    tasks = []
    for cont_idx in cont_ids:
        orig_id = cont_to_orig.get(cont_idx)
        if orig_id is None:
            continue
        entry = author_split.get(orig_id) or author_split.get(str(orig_id))
        if not entry:
            continue
        test_page = (entry.get('test_page') or '').replace('\\', '/')
        if not test_page:
            continue
        tasks.append((cont_idx, orig_id, test_page, args.output_dir))

    logger.info(f"任务总数: {len(tasks)}  |  worker 数: {args.n_workers}")

    # 并行执行
    t0 = time.time()
    ok, skip, fail = 0, 0, 0
    fail_msgs = []

    with ProcessPoolExecutor(
        max_workers=args.n_workers,
        initializer=_init_worker,
        initargs=(args.config,),
    ) as ex:
        futures = {ex.submit(_process_one_author, t): t for t in tasks}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Generating"):
            cont_idx, orig_id, success, msg, elapsed = fut.result()
            if success:
                if msg == "skipped":
                    skip += 1
                else:
                    ok += 1
            else:
                fail += 1
                fail_msgs.append((cont_idx, orig_id, msg))
                logger.error(f"cont_idx={cont_idx} orig_id={orig_id} failed: {msg}")

    total_time = time.time() - t0
    logger.info(f"\n===== 完成 =====")
    logger.info(f"成功: {ok}, 跳过: {skip}, 失败: {fail}")
    logger.info(f"总耗时: {total_time/60:.1f} 分钟")

    # 保存失败日志
    if fail_msgs:
        fail_log = Path(args.output_dir) / '_failed.json'
        with open(fail_log, 'w', encoding='utf-8') as f:
            json.dump([{'cont_idx': c, 'orig_id': o, 'msg': m} for c, o, m in fail_msgs],
                      f, ensure_ascii=False, indent=2)
        logger.info(f"失败详情: {fail_log}")


if __name__ == '__main__':
    main()