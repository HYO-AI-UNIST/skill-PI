from __future__ import annotations

import dataclasses
import hashlib
import math
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn


def posemb_sincos(pos: Tensor, embedding_dim: int, min_period: float, max_period: float) -> Tensor:
    """스칼라 위치(여기선 flow matching timestep)에 대한 sine-cosine 임베딩.

    shape 기호:  B=batch, E=embedding_dim.
    입력 pos [B] (스칼라 1개/샘플) -> 출력 [B, E] (E 차원 벡터).

    openpi posemb_sincos / create_sinusoidal_pos_embedding 와 동일.
    """
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")
    fraction = torch.linspace(0.0, 1.0, embedding_dim // 2, dtype=torch.float32, device=pos.device)  # [E/2]
    period = min_period * (max_period / min_period) ** fraction  # [E/2]
    scaling = 1.0 / period * 2 * math.pi  # [E/2]
    sinusoid = scaling[None, :] * pos[:, None].float()  # [1,E/2] * [B,1] -> [B, E/2]
    return torch.cat([torch.sin(sinusoid), torch.cos(sinusoid)], dim=-1)  # [B, E]


def make_attn_mask(input_mask: Tensor, mask_ar: Tensor) -> Tensor:
    """big_vision 의 block-wise attention mask 생성. (openpi make_attn_mask 와 동일)

    shape 기호:  B=batch, N=시퀀스 길이(토큰 수).
    입력 [B, N] (1D 마스크 2개) -> 출력 [B, N, N] (2D attention 마스크).

    토큰 j 는, 자신의 cumulative(mask_ar) 가 토큰 i 의 것보다 작거나 같은 토큰 i 를 본다.
    mask_ar 가 1 인 위치는 "여기서부터 새 블록이 시작 = 이전 토큰들이 나를 못 본다" 를 뜻한다.

      [[1 1 1 1 1 1]] -> 순수 causal
      [[0 0 0 1 1 1]] -> prefix-lm (앞 3개는 서로 full, 뒤 3개는 causal)

    Args:
      input_mask : bool[B, N]  유효 토큰이면 True (padding 이면 False)
      mask_ar    : bool/int[B, N]  블록 경계
    Returns:
      attn_mask  : bool[B, N, N]  (query i, key j) i 가 j 를 볼 수 있으면 True
    """
    mask_ar = mask_ar.to(torch.int32).broadcast_to(input_mask.shape)  # [B, N]
    cumsum = torch.cumsum(mask_ar, dim=1)  # [B, N]  블록 인덱스(누적합)
    # [B,1,N] <= [B,N,1]  ->  [B, N(query), N(key)]
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]  # [B, query, key]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]  # [B,1,N] * [B,N,1] -> [B,N,N]  padding 차단
    return attn_mask & valid_mask.bool()  # [B,N,N] & [B,N,N] -> [B, N, N]