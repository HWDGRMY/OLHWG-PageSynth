#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: style_encoder.py
@description: 风格编码器，从多尺度特征中提取书法风格；以及对比学习投影头。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .basemodel import Conv1d, EncoderBlock


class StyleEncoder(nn.Module):
    """多尺度风格编码器（原 style_encoer，修正拼写）。

    Args:
        dim: 基础通道维度。
        num_classes: 分类头类别数（训练时使用，推理时可忽略）。
    """

    def __init__(self, dim: int = 128, num_classes: int = 1020):
        super().__init__()
        self.linear = Conv1d(in_channels=3, out_channels=dim, kernel_size=3)
        self.en_block1 = nn.Sequential(EncoderBlock(dim, dim * 2, 2))
        self.en_block2 = nn.Sequential(EncoderBlock(dim * 2, dim * 4, 2))
        self.en_block3 = nn.Sequential(EncoderBlock(dim * 4, dim * 8, 2))
        self.elu = nn.ELU()

        # 分类头（用于训练时监督），推理时不会用到
        self.func_c = nn.Linear(1024, num_classes)
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor):
        """前向传播。

        Args:
            x: 形状为 (B, 3, L) 的轨迹。

        Returns:
            元组 (x, en1, en2, en3)，四个不同尺度的特征。
        """
        x = self.linear(x)                        # (B, dim, L)
        en1 = self.en_block1(self.elu(x))         # (B, 2dim, L/2)
        en2 = self.en_block2(self.elu(en1))       # (B, 4dim, L/4)
        en3 = self.en_block3(self.elu(en2))       # (B, 8dim, L/8)
        return x, en1, en2, en3

    def loss_function(self, x: torch.Tensor, style_ids: torch.Tensor):
        """分类损失（训练风格编码器用）。

        Args:
            x: 形状 (B, L, 3)。
            style_ids: 形状 (B,) 的标签。

        Returns:
            (loss, accuracy)
        """
        x = x.permute(0, 2, 1)
        _, _, _, f = self.forward(x)
        f = torch.mean(f, dim=2)
        p = self.func_c(f)
        loss = self.cross_entropy(p, style_ids)
        _, predicted = torch.max(p.data, 1)
        accuracy = (predicted == style_ids).sum().item() / style_ids.size(0)
        return loss, accuracy


class StyleProjector(nn.Module):
    """多尺度对比学习投影头（仅训练时使用）。"""

    def __init__(self, dim: int = 128):
        super().__init__()
        self.project1 = nn.Linear(2 * dim, 2 * dim)
        self.project2 = nn.Linear(4 * dim, 4 * dim)
        self.project3 = nn.Linear(8 * dim, 8 * dim)
        self.logit_scale = 2.0
        self.w1, self.w2, self.w3 = 0.01, 0.1, 0.1

    def contrastive_loss_all(self, en1: torch.Tensor, en2: torch.Tensor, en3: torch.Tensor) -> torch.Tensor:
        """多尺度对比损失之和。"""
        en1 = self.project1(en1.permute(0, 2, 1))
        en2 = self.project2(en2.permute(0, 2, 1))
        en3 = self.project3(en3.permute(0, 2, 1))
        loss1 = self._contrastive_loss(en1)
        loss2 = self._contrastive_loss(en2)
        loss3 = self._contrastive_loss(en3)
        return self.w1 * loss1 + self.w2 * loss2 + self.w3 * loss3

    def _contrastive_loss(self, en: torch.Tensor) -> torch.Tensor:
        """单个尺度下的对比损失（InfoNCE）。"""
        batch_size = en.shape[0]
        features = en[:, :50, :].mean(dim=1)
        features_ = en[:, 50:100, :].mean(dim=1)
        sim_matrix = torch.mm(features, features_.t()) * self.logit_scale
        label = torch.arange(batch_size).to(sim_matrix).long()
        loss_1 = F.nll_loss(F.log_softmax(sim_matrix, dim=1), label)
        loss_2 = F.nll_loss(F.log_softmax(sim_matrix.t(), dim=1), label)
        return loss_1 + loss_2

# ------------------------------------------------------------------ #
# 向后兼容别名：官方代码用的是 style_encoer / style_projector（拼写有误）
# 为了不破坏任何已有引用，这里做别名映射。
# ------------------------------------------------------------------ #
style_encoer = StyleEncoder
style_encoder = StyleEncoder
style_projector = StyleProjector