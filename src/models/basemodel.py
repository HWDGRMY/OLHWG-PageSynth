#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: basemodel.py
@description: 神经网络基础组件，包括时间编码、1D卷积、残差单元及编解码器。
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionEmbeddings(nn.Module):
    """正弦位置编码（用于时间步）。

    Args:
        dim: 嵌入维度。
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        """计算正弦位置编码。

        Args:
            time: 形状为 (B,) 的时间步张量。

        Returns:
            形状为 (B, dim) 的位置编码。
        """
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


def cosine_schedule(initial_scale: float, final_scale: float, total_steps: int) -> torch.Tensor:
    """余弦噪声调度。

    Args:
        initial_scale: 初始缩放。
        final_scale: 最终缩放。
        total_steps: 总步数。

    Returns:
        形状为 (total_steps,) 的缩放序列。
    """
    scales = []
    for step in range(total_steps):
        proportion = step / total_steps
        decay = 0.5 * (1 + torch.cos(torch.tensor(np.pi * proportion)))
        scale = initial_scale * (1 - decay) + final_scale * decay
        scales.append(scale)
    return torch.tensor(scales)


class Conv1d(nn.Module):
    """带右侧 padding 的 1D 卷积。

    Args:
        in_channels: 输入通道数。
        out_channels: 输出通道数。
        kernel_size: 卷积核大小。
        dilation: 空洞率。
        stride: 步长。
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 dilation: int = 1, stride: int = 1):
        super().__init__()
        self.pad = dilation * (kernel_size - 1)
        self.conv = nn.Conv1d(
            in_channels=in_channels, out_channels=out_channels,
            kernel_size=kernel_size, dilation=dilation, stride=stride
        )
        nn.init.xavier_uniform_(self.conv.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, [0, self.pad])
        return self.conv(x)


class TransConv1d(nn.Module):
    """1D 转置卷积。"""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 dilation: int = 1, stride: int = 1):
        super().__init__()
        self.pad = int(((2 * dilation - 1) * stride - dilation + 1) / 2)
        self.transconv = nn.ConvTranspose1d(
            in_channels=in_channels, out_channels=out_channels,
            kernel_size=kernel_size, stride=stride, padding=self.pad
        )
        nn.init.xavier_uniform_(self.transconv.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.transconv(x)


class ResidualUnit(nn.Module):
    """残差单元（输入输出通道需一致）。"""

    def __init__(self, in_channels: int, out_channels: int, dilation: int, kernel_size: int = 3):
        super().__init__()
        self.layers = nn.Sequential(
            Conv1d(in_channels=in_channels, out_channels=out_channels,
                   kernel_size=kernel_size, dilation=dilation),
            nn.ELU(),
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.layers(x)


class EncoderBlock(nn.Module):
    """编码器块。"""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        kernel = stride * 2 if stride != 1 else 1
        self.layers = nn.Sequential(
            ResidualUnit(in_channels, in_channels, dilation=1), nn.ELU(),
            ResidualUnit(in_channels, in_channels, dilation=3), nn.ELU(),
            ResidualUnit(in_channels, in_channels, dilation=5), nn.ELU(),
            Conv1d(in_channels, out_channels, kernel_size=kernel, stride=stride),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class DecoderBlock(nn.Module):
    """解码器块。"""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        kernel = stride * 2 if stride != 1 else 1
        self.layers = nn.Sequential(
            TransConv1d(in_channels, out_channels, kernel_size=kernel, stride=stride), nn.ELU(),
            ResidualUnit(out_channels, out_channels, dilation=1), nn.ELU(),
            ResidualUnit(out_channels, out_channels, dilation=3), nn.ELU(),
            ResidualUnit(out_channels, out_channels, dilation=5),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)