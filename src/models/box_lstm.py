#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: box_lstm.py
@description: 行布局生成器（LSTM），预测每个字符的边界框。
"""

import torch
import torch.nn as nn


class HParams:
    def __init__(self):
        self.hidden_size = 128
        self.n_layers = 2
        self.dropout = 0.1
        self.emb_size = 32


class BoxLstm(nn.Module):
    """布局生成 LSTM（原 Box_Lstm）。

    Args:
        device: 计算设备。
    """

    def __init__(self, device: torch.device, hp: HParams = None):
        super().__init__()
        self.hp = hp or HParams()
        self.device = device
        self.lstm = nn.LSTM(
            self.hp.emb_size + 4, self.hp.hidden_size, self.hp.n_layers,
            batch_first=True, dropout=self.hp.dropout, bidirectional=False,
        )
        self.func_out = nn.Linear(self.hp.hidden_size, 4)
        self.func_in = nn.Linear(6, self.hp.emb_size)
        self.loss = nn.L1Loss()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = torch.zeros(self.hp.n_layers, inputs.shape[0], self.hp.hidden_size).to(self.device)
        cell = torch.zeros(self.hp.n_layers, inputs.shape[0], self.hp.hidden_size).to(self.device)
        x, _ = self.lstm(inputs, (hidden, cell))
        return self.func_out(x)

    def get_loss(self, target_seq: torch.Tensor, ref_seq: torch.Tensor) -> torch.Tensor:
        """训练时的损失。

        Args:
            target_seq: (B, L, 4) 真实边界框 [h, w, center, delta]
            ref_seq: (B, L, 6) 参考先验 [h_mu, h_sigma, w_mu, w_sigma, c_mu, c_sigma]
        """
        target_seq_shift = torch.zeros_like(target_seq)
        target_seq_shift[:, 1:, :] = target_seq[:, :-1, :]
        input_seq = self.func_in(ref_seq)
        input_seq = torch.cat([target_seq_shift, input_seq], -1)
        out_seq = self.forward(input_seq)
        return self.loss(target_seq, out_seq)

    def get_box(self, ref_seq: torch.Tensor, prefix_seq: torch.Tensor = None):
        """推理时逐步生成边界框。

        Args:
            ref_seq: (B, NUM, 6) 字符先验。
            prefix_seq: (B, L_prefix, 4+dim) 条件前缀，可选。

        Returns:
            长度 NUM 的列表，每项形状 (B, 1, 4)。
        """
        bs, num, _ = ref_seq.shape
        if prefix_seq is None:
            sos = torch.zeros(bs, 1, 4).to(self.device)
            hidden = torch.zeros(self.hp.n_layers, bs, self.hp.hidden_size).to(self.device)
            cell = torch.zeros(self.hp.n_layers, bs, self.hp.hidden_size).to(self.device)
            hidden_cell = (hidden, cell)
        else:
            hidden = torch.zeros(self.hp.n_layers, bs, self.hp.hidden_size).to(self.device)
            cell = torch.zeros(self.hp.n_layers, bs, self.hp.hidden_size).to(self.device)
            hidden_cell = (hidden, cell)
            x, hidden_cell = self.lstm(prefix_seq, hidden_cell)
            x = self.func_out(x)
            sos = x[:, -1, :].unsqueeze(1)

        boxes = []
        for i in range(num):
            inputs = self.func_in(ref_seq[:, i, :]).unsqueeze(1)
            inputs = torch.cat([sos, inputs], -1)
            outputs, hidden_cell = self.lstm(inputs, hidden_cell)
            sos = self.func_out(outputs)
            boxes.append(sos)
        return boxes

# 向后兼容别名
Box_Lstm = BoxLstm