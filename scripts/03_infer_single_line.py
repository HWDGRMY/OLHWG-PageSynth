#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 03_infer_single_line.py
@description: 单行手写生成推理脚本。
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.models.unet_ddpm import DDPM
from src.models.style_encoder import StyleEncoder
from src.models.box_lstm import BoxLstm
from src.pipeline.line_assembler import generate_line
from src.utils.draw_utils import draw_line

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Single line inference.")
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    parser.add_argument('--text', type=str, required=True)
    parser.add_argument('--author_id', type=int, default=1,
                        help='Which writer style to borrow (default 1).')
    parser.add_argument('--output_dir', type=str, default='outputs/single_line')
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    device = torch.device(config['generation']['device']
                          if torch.cuda.is_available() else 'cpu')
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # ---- 加载模型 ---- #
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

    mydict = np.load(config['data']['mydict_npy'], allow_pickle=True).tolist()

    # ---- 借用某个作者的风格缓存 ---- #
    style_cache = Path(config['paths']['style_cache_dir']) / f"{args.author_id}.pt"
    if not style_cache.exists():
        logger.warning(f"Style cache {style_cache} missing, using None (may fail).")
        style_features = None
    else:
        style_features = [t.to(device) for t in torch.load(style_cache, map_location='cpu')]
        logger.info(f"Borrowed style from writer {args.author_id}.")

    # ---- 生成 ---- #
    logger.info(f"Generating line: {args.text}")
    with torch.no_grad():
        traj = generate_line(
            text=args.text,
            ddpm=ddpm,
            style_encoder=style_en,
            box_lstm=box_lstm,
            device=device,
            mydict=mydict,
            style_features=style_features,
            y_offset=0.0,
        )

    save_path = Path(args.output_dir) / "generated_line.png"
    draw_line(traj, str(save_path))
    logger.info(f"Saved to {save_path}")


if __name__ == '__main__':
    main()