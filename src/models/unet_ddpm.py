#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: unet_ddpm.py
@description: 1D U-Net + DDPM 扩散模型，用于单字手写轨迹生成。
"""

import logging

import numpy as np
import torch
import torch.nn as nn

from .basemodel import (SinusoidalPositionEmbeddings, EncoderBlock,
                        DecoderBlock, cosine_schedule)

logger = logging.getLogger(__name__)


class HParams:
    """DDPM 训练超参数，与官方 train_ddpm.py 一致。"""
    def __init__(self):
        self.bs = 64
        self.epo = 2_000_001
        self.lr = 1e-4
        self.min_lr = 1e-5
        self.lr_decay = 0.9995
        self.grad_clip = 1.0

        self.time_emb_size = 16
        self.char_emb_size = 256
        self.time_step = 1000
        self.beta_0 = 1e-4
        self.beta_T = 2e-2

        # 采样噪声系数
        self.sigma_scale = 0.8


class CrossAttention(nn.Module):
    """QKV 交叉注意力。"""

    def __init__(self, dim1: int, dim2: int, q_dim: int, k_dim: int, v_dim: int):
        super().__init__()
        self.q_func = nn.Linear(dim1, q_dim)
        self.k_func = nn.Linear(dim2, k_dim)
        self.v_func = nn.Linear(dim2, v_dim)
        self.out_func = nn.Linear(v_dim, dim1)

    def forward(self, x: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        f = f.permute(0, 2, 1)
        q, k, v = self.q_func(x), self.k_func(f), self.v_func(f)
        attn_weights = torch.softmax(torch.bmm(q, k.transpose(1, 2)) / (256 ** 0.5), dim=-1)
        output = self.out_func(torch.bmm(attn_weights, v)) + x
        return output.permute(0, 2, 1)


class Unet(nn.Module):
    """1D U-Net 去噪器。"""

    def __init__(self, in_channels: int, out_channels: int, dim: int = 128):
        super().__init__()
        self.en_block1 = EncoderBlock(in_channels, dim)
        self.en_block2 = EncoderBlock(dim * 2, dim * 2)
        self.en_block3 = EncoderBlock(dim * 4, dim * 4)
        self.en_block4 = EncoderBlock(dim * 8, dim * 8)
        self.downblock1 = EncoderBlock(dim, dim * 2, 2)
        self.downblock2 = EncoderBlock(dim * 2, dim * 4, 2)
        self.downblock3 = EncoderBlock(dim * 4, dim * 8, 2)

        self.de_block1 = EncoderBlock(dim * 4, dim * 4)
        self.de_block2 = EncoderBlock(dim * 2, dim * 2)
        self.de_block3 = EncoderBlock(dim, in_channels)
        self.de_block4 = EncoderBlock(in_channels, out_channels)
        self.upblock1 = DecoderBlock(dim * 8, dim * 4, 2)
        self.upblock2 = DecoderBlock(dim * 4, dim * 2, 2)
        self.upblock3 = DecoderBlock(dim * 2, dim, 2)

        self.cross_attention1 = CrossAttention(in_channels, 128, 256, 256, 256)
        self.cross_attention2 = CrossAttention(dim * 2, 256, 256, 256, 256)
        self.cross_attention3 = CrossAttention(dim * 4, 512, 256, 256, 256)
        self.cross_attention4 = CrossAttention(dim * 8, 1024, 256, 256, 256)
        self.elu = nn.ELU()

    def forward(self, x: torch.Tensor, style_features=None) -> torch.Tensor:
        en1 = self.elu(self.en_block1(x))
        en2 = self.elu(self.downblock1(en1))
        en3 = self.elu(self.en_block2(en2))
        en4 = self.elu(self.downblock2(en3))
        en5 = self.elu(self.en_block3(en4))
        en6 = self.elu(self.downblock3(en5))

        de1 = self.elu(self.en_block4(en6))
        de1 = self.cross_attention4(de1, style_features[3])
        de2 = self.elu(self.upblock1(de1))
        de3 = self.elu(self.de_block1(de2 + en5))
        de3 = self.cross_attention3(de3, style_features[2])
        de4 = self.elu(self.upblock2(de3 + en4))
        de5 = self.elu(self.de_block2(de4 + en3))
        de5 = self.cross_attention2(de5, style_features[1])
        de6 = self.elu(self.upblock3(de5 + en2))
        de7 = self.elu(self.de_block3(de6 + en1))
        de7 = self.cross_attention1(de7, style_features[0])
        return self.de_block4(de7)


class DDPM(nn.Module):
    """DDPM 扩散模型，封装前向加噪、反向去噪和采样生成。"""

    def __init__(self, hp: HParams = None,
                 input_channel: int = None, output_channel: int = 3):
        super().__init__()
        self.hp = hp or HParams()
        if input_channel is None:
            input_channel = self.hp.time_emb_size + self.hp.char_emb_size + 4

        self.time_emb = SinusoidalPositionEmbeddings(self.hp.time_emb_size)
        self.denoiser = Unet(input_channel, output_channel)

        self.beta = None
        self.alpha = None
        self.alpha_cumprod = None

        self.char_embeddings = nn.Embedding(4055, self.hp.char_emb_size)
        self.loss = nn.MSELoss()

    def setup_diffusion(self, device: torch.device):
        self.beta = cosine_schedule(self.hp.beta_T, self.hp.beta_0,
                                     self.hp.time_step).to(device)
        self.alpha = 1 - self.beta
        self.alpha_cumprod = torch.cumprod(self.alpha, -1).to(device)
        logger.info(f"[DDPM] time_step={self.hp.time_step}, "
                    f"beta[0]={self.beta[0].item():.6f}, "
                    f"beta[-1]={self.beta[-1].item():.6f}, "
                    f"alpha_cumprod[0]={self.alpha_cumprod[0].item():.6f}, "
                    f"alpha_cumprod[-1]={self.alpha_cumprod[-1].item():.8f}, "
                    f"sigma_scale={self.hp.sigma_scale}")

    def get_positions(self, durations, device: torch.device) -> torch.Tensor:
        positions = []
        for duration in durations:
            partitions = [torch.linspace(0, 1, d) for d in duration]
            concatenated = torch.cat(partitions).unsqueeze(0).unsqueeze(0)
            positions.append(concatenated)
        return torch.cat(positions).to(device)

    def get_contents_times(self, inputs: torch.Tensor, t: torch.Tensor,
                            tags, durations) -> torch.Tensor:
        device = inputs.device
        tags_tensor = torch.from_numpy(np.array(tags)).to(device)
        content_features = self.char_embeddings(tags_tensor).permute(0, 2, 1)
        bs, _, length = inputs.shape

        inputs_t = self.time_emb(t).unsqueeze(2)
        if inputs_t.shape[0] != bs:
            inputs_t = inputs_t.repeat(bs, 1, length)
        else:
            inputs_t = inputs_t.repeat(1, 1, length)

        batch_positions = self.get_positions(durations, device)
        return torch.cat([inputs, batch_positions, content_features, inputs_t], 1)

    def pred_noise(self, inputs: torch.Tensor, style_features=None) -> torch.Tensor:
        return self.denoiser(inputs, style_features)

    def sample_x_t(self, x: torch.Tensor, t: torch.Tensor):
        noise = torch.randn_like(x).to(x.device)
        if t.shape[0] == 1:
            x_t = (torch.sqrt(self.alpha_cumprod[t]) * x
                   + torch.sqrt(1 - self.alpha_cumprod[t]) * noise)
            return x_t, noise
        x_t = torch.zeros_like(x)
        for i in range(x.shape[0]):
            x_t[i] = (torch.sqrt(self.alpha_cumprod[t[i]]) * x[i]
                      + torch.sqrt(1 - self.alpha_cumprod[t[i]]) * noise[i])
        return x_t, noise

    def remove_noise(self, x_t: torch.Tensor, t: torch.Tensor,
                      predict_noise: torch.Tensor) -> torch.Tensor:
        temp = x_t - (self.beta[t] / torch.sqrt(1 - self.alpha_cumprod[t])) * predict_noise
        return temp / torch.sqrt(self.alpha[t])

    def estimate_x0(self, x_t: torch.Tensor, t: torch.Tensor,
                     predict_noise: torch.Tensor) -> torch.Tensor:
        return ((x_t - torch.sqrt(1 - self.alpha_cumprod[t]) * predict_noise)
                / torch.sqrt(self.alpha_cumprod[t]))

    @torch.no_grad()
    def generate(self, tags, durations, style_features=None) -> torch.Tensor:
        """从纯噪声出发逐步生成轨迹。

        Args:
            tags: list of list，字符 id 序列。
            durations: list of list，每个字符占用步数（含 EOS padding）。
            style_features: 长度 4 的风格特征列表。

        Returns:
            形状为 (B, 3, L) 的生成轨迹。
        """
        device = self.alpha.device
        bs, length = len(tags), len(tags[0])
        x = torch.randn(bs, 3, length).to(device)

        for i in range(self.hp.time_step):
            t = self.hp.time_step - 1 - i
            t_tensor = torch.tensor([t]).to(device)
            inputs = self.get_contents_times(x, t=t_tensor, tags=tags, durations=durations)
            predict_noise = self.pred_noise(inputs, style_features)
            x = self.remove_noise(x, t_tensor, predict_noise)

            if t > 15:
                sigma = torch.sqrt(
                    self.hp.sigma_scale * self.beta[t]
                    * (1 - self.alpha_cumprod[t - 1])
                    / (1 - self.alpha_cumprod[t])
                )
                x += sigma * torch.randn_like(x)

        return x