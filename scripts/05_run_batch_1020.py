#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 05_run_batch_1020.py
@description: 批量生成所有作者的页面，支持断点续传。
@usage: python scripts/05_run_batch_1020.py --config configs/default.yaml
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.models.unet_ddpm import DDPM
from src.models.style_encoder import StyleEncoder
from src.models.box_lstm import BoxLstm
from src.pipeline.page_generator import generate_page
from src.utils.draw_utils import draw_page

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Batch generation for all writers.")
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    with open('configs/author_split.yaml', 'r', encoding='utf-8') as f:
        author_split = yaml.safe_load(f)

    device = torch.device(config['generation']['device']
                          if torch.cuda.is_available() else 'cpu')
    output_dir = Path(config['paths']['output_dir'])
    style_cache_dir = Path(config['paths']['style_cache_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- 模型一次性加载 ---- #
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

    # ---- 遍历所有作者 ---- #
    writer_ids = sorted(author_split.keys())
    for writer_id in tqdm(writer_ids, desc="Batch generating"):
        save_dir = output_dir / f"author_{writer_id:04d}"
        save_path = save_dir / "generated_page.png"
        if save_path.exists():
            continue

        test_page_path = author_split[writer_id].get('test_page')
        if not test_page_path:
            continue

        style_cache = style_cache_dir / f"{writer_id}.pt"
        if not style_cache.exists():
            logger.warning(f"Style cache missing for writer {writer_id}, skipping.")
            continue

        try:
            style_features = [t.to(device) for t in torch.load(style_cache, map_location='cpu')]
            save_dir.mkdir(parents=True, exist_ok=True)

            with torch.no_grad():
                page_traj = generate_page(
                    wptt_path=test_page_path,
                    ddpm=ddpm,
                    style_encoder=style_en,
                    box_lstm=box_lstm,
                    device=device,
                    dictionary=dictionary,
                    mydict=mydict,
                    char_boxes=char_boxes,
                    style_features=style_features,
                    line_height=config['generation'].get('line_height', 1.2),
                    len_prefix=config['generation'].get('len_prefix', 10),
                )

            draw_page(page_traj, str(save_path))
        except Exception as e:
            logger.error(f"Writer {writer_id} failed: {e}")
            continue

    logger.info("Batch generation done.")


if __name__ == '__main__':
    main()