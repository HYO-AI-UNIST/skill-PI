from __future__ import annotations

import dataclasses
import math
from typing import Optional
import configparser

import torch
import torch.nn.functional as F
from torch import Tensor, nn

class GemmaMLP(nn.Module):
    """Gemma FeedForward: GeGLU.  out = (gelu(x W_gate) * (x W_up)) W_down

    shape 기호:  B=batch, T=토큰 수, width=expert width, mlp=mlp hidden(mlp_dim).
    forward: x [B, T, width] -> [B, T, mlp] (확장) -> [B, T, width] (복귀).

    openpi gemma.FeedForward 와 동일 구조 (bias 없음).
    """

    def __init__(self, width: int, hidden_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(width, hidden_dim, bias=False)  # width -> mlp
        self.up_proj = nn.Linear(width, hidden_dim, bias=False)  # width -> mlp
        self.down_proj = nn.Linear(hidden_dim, width, bias=False)  # mlp -> width

    def forward(self, x: Tensor) -> Tensor:
        # gate_proj(x)[B,T,mlp] --gelu--> [B,T,mlp] ; up_proj(x)[B,T,mlp] ; 둘을 곱 [B,T,mlp]
        # --down_proj--> [B,T,width]   (x[B,T,width] -> ... -> [B,T,width])
        return self.down_proj(F.gelu(self.gate_proj(x), approximate="tanh") * self.up_proj(x))

class RMSNorm(nn.Module):
    """RMSNorm. cond 가 주어지면 adaptive RMSNorm (adaRMS) 으로 동작.

    shape 기호:  B=batch, T=토큰 수, dim=정규화할 채널 차원(=expert width).
    forward: x [B, T, dim] -> normed [B, T, dim] (차원 불변),  gate [B,1,dim] 또는 None.

    - 일반 RMSNorm : normed * (1 + scale)        (scale 은 learnable, zeros 초기화)
    - adaRMS       : cond(=timestep emb [B, cond_dim]) 로부터 scale/shift/gate 를 생성
                     normed * (1 + scale) + shift , 그리고 residual 용 gate 반환

    openpi gemma.RMSNorm 과 동일. variance 는 float32 로 계산.
    """
    def __init__(self, dim: int, adarms_cond_dim: Optional[int] = None):
        super().__init__()
        self.dim = dim
        self.use_adarms = adarms_cond_dim is not None
        if self.use_adarms:
            # cond[B, cond_dim] -> [B, 3*dim] (scale/shift/gate). zeros 초기화 => 처음엔 identity.
            self.modulation = nn.Linear(adarms_cond_dim, dim * 3)
            nn.init.zeros_(self.modulation.weight)
            nn.init.zeros_(self.modulation.bias)
        else:
            self.scale = nn.Parameter(torch.zeros(dim))  # [dim], (1 + scale) 형태로 사용

    def forward(self, x: Tensor, cond: Optional[Tensor] = None):
        # x: [B, T, dim]
        dtype = x.dtype
        var = x.float().pow(2).mean(dim=-1, keepdim=True)  # [B, T, 1]
        normed = x.float() * torch.rsqrt(var + 1e-6)  # [B, T, dim]
        if not self.use_adarms:
            normed = normed * (1.0 + self.scale.float())  # [B, T, dim]
            return normed.to(dtype), None
        # adaRMS:  cond [B, cond_dim] --modulation--> [B, 3*dim]
        mod = self.modulation(cond.to(self.modulation.weight.dtype))  # [B, cond_dim] -> [B, 3*dim]
        # mod[B,3*dim] --[:,None,:]--> [B,1,3*dim] --chunk(3, dim=-1)--> scale/shift/gate 각 [B,1,dim]
        scale, shift, gate = torch.chunk(mod[:, None, :], 3, dim=-1)
        normed = normed * (1.0 + scale.float()) + shift.float()  # [B, T, dim] (broadcast over T)
        return normed.to(dtype), gate  # gate: [B, 1, dim]

class Linear(nn.Linear):
    def forward(self, x: Optional[Tensor]) -> Optional[Tensor]:
        if x is None:
            return None
        return super().forward(x)