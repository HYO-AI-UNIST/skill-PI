from __future__ import annotations

import dataclasses
import hashlib
import math
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

class VitMLP(nn.Module):
    """ViT FeedForward.  shape: x [B, P, vit_width] -> [B, P, vit_mlp] -> [B, P, vit_width]."""

    def __init__(self, width: int, mlp_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(width, mlp_dim)  # vit_width -> vit_mlp
        self.fc2 = nn.Linear(mlp_dim, width)  # vit_mlp -> vit_width

    def forward(self, x: Tensor) -> Tensor:
        # x[B,P,vit_width] --fc1--> [B,P,vit_mlp] --gelu--> [B,P,vit_mlp] --fc2--> [B,P,vit_width]
        return self.fc2(F.gelu(self.fc1(x), approximate="tanh"))


class VitEncoderBlock(nn.Module):
    """표준 pre-LN 트랜스포머 인코더 블록 (MHSA + MLP). (openpi siglip.Encoder1DBlock)

    shape 기호:  B=batch, P=패치(토큰) 수, vit_width=ViT width.
    x [B, P, vit_width] -> [B, P, vit_width] (차원 불변).
    """

    def __init__(self, width: int, num_heads: int, mlp_dim: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(width)
        self.attn = nn.MultiheadAttention(width, num_heads, batch_first=True)  # 패치끼리 full attention
        self.ln2 = nn.LayerNorm(width)
        self.mlp = VitMLP(width, mlp_dim)

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, P, vit_width]
        y = self.ln1(x)  # [B, P, vit_width]
        y, _ = self.attn(y, y, y, need_weights=False)  # [B, P, vit_width]
        x = x + y  # residual
        y = self.ln2(x)
        return x + self.mlp(y)  # [B, P, vit_width]


class SigLIP(nn.Module):
    """SigLIP ViT (So400m/14). 이미지를 patch 토큰으로 인코딩한 뒤 PaliGemma width 로 투영.

    shape 흐름 (카메라 1대):
      image [B, 3, 224, 224]
        -> patch conv      [B, vit_width, 16, 16]
        -> flatten         [B, P=256, vit_width]
        -> + posemb        [B, P, vit_width]
        -> encoder x depth [B, P, vit_width]
        -> LayerNorm       [B, P, vit_width]
        -> head (Linear)   [B, P, pg_width]      <- LLM 입력 토큰 차원으로 투영

    openpi siglip._Module 의 pool_type="none" 경로. head 가 vit_width(1152) -> pg_width(2048).
    """

    def __init__(self, image_size, patch_size, width, depth, num_heads, mlp_dim, out_width):
        super().__init__()
        n_height_patches = image_size[0] // patch_size[0]
        n_width_patches = image_size[1] // patch_size[1]
        num_patches = n_height_patches * n_width_patches

        self.patch_embed = nn.Conv2d(3, width, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, width) * (1.0 / math.sqrt(width)))  # [1, P, vit_width]
        self.blocks = nn.ModuleList(
            [VitEncoderBlock(width, num_heads, mlp_dim) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(width)
        # vit width -> paligemma width 로 투영 (LLM 토큰 차원에 맞춤). vit_width -> pg_width
        self.head = nn.Linear(width, out_width)

    def forward(self, image: Tensor) -> Tensor:
        """image: [B, 3, 224, 224] in [-1,1]  ->  tokens: [B, P, pg_width]"""
        x = self.patch_embed(image)  # [B, 3, 224, 224] -> [B, vit_width, 16, 16]
        x = x.flatten(2).transpose(1, 2)  # [B, vit_width, 16, 16] -> [B, vit_width, 256] -> [B, P=256, vit_width]
        x = x + self.pos_embed  # [B, P, vit_width]  (위치 임베딩 더하기)
        for block in self.blocks:
            x = block(x)  # [B, P, vit_width] (불변)
        x = self.norm(x)  # [B, P, vit_width]
        return self.head(x)  # head (vit_width->pg_width): [B, P, vit_width] -> [B, P, pg_width]
