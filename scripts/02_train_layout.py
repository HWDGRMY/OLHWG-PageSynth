#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 02_train_layout.py
@description: 训练行布局生成器 (LSTM)
@usage: python scripts/02_train_layout.py --config configs/default.yaml
"""

import argparse
import logging
import yaml
import torch
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.models.box_lstm import Box_Lstm
# 假设你在 src/datasets 下实现了 get_line_batch 用于加载 all_datas.npy
from src.datasets.page_dataset import get_line_batch

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Layout LSTM.")
    parser.add_argument('--config', type=str, default='configs/default.yaml', help='Path to config file.')
    parser.add_argument('--epochs', type=int, default=20000, help='Number of training epochs.')
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    device = torch.device(config['generation']['device'] if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # 初始化模型
    model = Box_Lstm(device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 假设有数据加载逻辑
    # dataset = ...

    logger.info("Start training layout generator...")
    model.train()
    for epoch in range(args.epochs):
        # 模拟训练循环
        # target_seq, ref_seq = get_line_batch(...)
        # loss = model.get_loss(target_seq, ref_seq)
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()

        if epoch % 3000 == 0 and epoch > 0:
            save_path = Path("checkpoints/finetuned") / f"boxlstm{epoch}.pth"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), save_path)
            logger.info(f"Saved model to {save_path}")


if __name__ == '__main__':
    main()