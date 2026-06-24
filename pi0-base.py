"""pi0-base: Pi0 (π0) Vision-Language-Action 모델의 self-contained PyTorch 구현.

openpi (Physical Intelligence) 의 핵심 Pi0 모델을 "베이스부터" 다시 작성한 파일이다.
openpi 원본은 두 가지 구현을 가지고 있다.

  1. JAX/Flax 원본 :  src/openpi/models/pi0.py  +  gemma.py  +  siglip.py
       -> Gemma backbone 을 from-scratch 로 직접 구현한 정석 코드.
  2. PyTorch 포팅  :  src/openpi/models_pytorch/pi0_pytorch.py  +  gemma_pytorch.py
       -> HuggingFace transformers 내부(PaliGemma/Gemma)를 monkey-patch 해서 사용.

이 파일은 (1) 의 from-scratch 구조를 PyTorch 로 충실히 옮긴 것이다. HuggingFace 의존성
없이, Pi0 의 핵심 알고리즘이 한 파일에 모두 들어있다.

----------------------------------------------------------------------------------
Pi0 한 줄 요약
----------------------------------------------------------------------------------
이미지 + 자연어 instruction + 로봇 state 를 받아서, 앞으로 실행할 연속 action chunk
(예: 50 step x 32 dim)를 **flow matching** 으로 생성하는 VLA 모델.

구조적으로는 "두 개의 전문가(expert)가 self-attention 을 공유하는 트랜스포머" 이다.
  - Prefix expert  = PaliGemma (SigLIP ViT 이미지 토큰 + Gemma 언어 토큰)  ... 큰 모델(2B)
  - Suffix expert  = Action Expert (state + noisy action + timestep 토큰)   ... 작은 모델(300M)

==================================================================================
[ SHAPE 표기 범례 ]  - 이 파일 전체에서 아래 약어로 텐서 차원을 표기한다.
==================================================================================
  B          = batch size (동시에 처리하는 샘플/로봇 수)
  3          = 이미지 RGB 채널
  Himg, Wimg = 이미지 높이/너비 (기본 224, 224)
  P          = 카메라 1대의 패치 토큰 수 = (Himg / patch)^2  (224/14 -> 16x16 = 256)
  ncam       = 카메라 수 (예: 3)
  L          = 언어 프롬프트 토큰 길이 (= max_token_len, pi0=48 / pi05=200)
  Ah         = action_horizon (한 번에 예측하는 action step 수, chunk 길이; 예: 50)
  Ad         = action_dim (로봇 action 차원; 예: 32)

  -- 시퀀스 길이 (토큰 개수) --
  Sp         = prefix 시퀀스 길이 = P*ncam + L         (이미지+언어 토큰)
  Ss         = suffix 시퀀스 길이 = (pi0: 1 state + Ah) / (pi05: Ah)
  St         = 전체 시퀀스 길이 = Sp + Ss
  T          = (attention 안에서) query 토큰 수,  S = key 토큰 수

  -- 임베딩(채널) 차원 --
  pg_width   = PaliGemma(=prefix expert) 의 width (gemma_2b -> 2048)
  ax_width   = Action Expert(=suffix expert) 의 width (gemma_300m -> 1024)
  vit_width  = SigLIP ViT 의 내부 width (So400m -> 1152)
  width_i    = i 번째 expert 의 width (expert 마다 다를 수 있음)

  -- attention head 관련 (모든 expert 가 공유해야 하는 값) --
  N (=num_heads)    = query head 수 (예: 8)
  K (=num_kv_heads) = key/value head 수 (GQA; 1 이면 multi-query)
  Hd (=head_dim)    = head 당 차원 (예: 256)
  G = N // K        = query head per kv head (GQA 그룹 크기)

  주의: ExpertAttention 안에서는 변수명으로 H=num_heads, Hd=head_dim 을 쓴다.
        반면 embed_suffix/sample_actions 등 action 쪽에서는 Ah=action_horizon 을 쓴다.
        (둘이 헷갈리지 않게 action 길이는 항상 'Ah' 로 표기한다.)
==================================================================================
"""

from __future__ import annotations

import dataclasses
import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

# =====================================================================================
# 0. Config
# =====================================================================================


@dataclasses.dataclass
class GemmaConfig:
    """하나의 Gemma expert 설정. (openpi gemma.get_config 의 variant 들과 동일)

    주의: 모든 expert 는 self-attention 을 공유하기 위해 head_dim / num_heads /
    num_kv_heads 가 동일해야 한다. width(=d_model) 와 mlp_dim 만 다를 수 있다.
    """

    width: int  # d_model (토큰 임베딩 차원). pg_width 또는 ax_width 가 됨.
    depth: int  # 트랜스포머 레이어 수 (모든 expert 동일해야 함)
    mlp_dim: int  # FeedForward hidden 차원
    num_heads: int  # query head 수 N
    num_kv_heads: int  # key/value head 수 K (GQA; 1 이면 multi-query)
    head_dim: int  # head 당 차원 Hd


# openpi gemma.py 의 variant 들.  (width, depth, mlp_dim, num_heads N, num_kv_heads K, head_dim Hd)
GEMMA_VARIANTS = {
    "dummy": GemmaConfig(width=64, depth=4, mlp_dim=128, num_heads=8, num_kv_heads=1, head_dim=16),
    "gemma_300m": GemmaConfig(width=1024, depth=18, mlp_dim=4096, num_heads=8, num_kv_heads=1, head_dim=256),
    "gemma_2b": GemmaConfig(width=2048, depth=18, mlp_dim=16_384, num_heads=8, num_kv_heads=1, head_dim=256),
}

PALIGEMMA_VOCAB_SIZE = 257_152
IMAGE_RESOLUTION = (224, 224)  # (Himg, Wimg)


@dataclasses.dataclass
class Pi0Config:
    """Pi0 모델 설정. (openpi Pi0Config 의 핵심 필드만)"""

    action_dim: int = 32  # Ad: 로봇 action 차원
    action_horizon: int = 50  # Ah: 한 번에 예측하는 action step 수 (chunk 길이)
    max_token_len: int = 48  # L: 언어 프롬프트 최대 토큰 수 (pi05 면 200)

    paligemma_variant: str = "gemma_2b"  # prefix expert (VLM) -> width = pg_width(2048)
    action_expert_variant: str = "gemma_300m"  # suffix expert (action) -> width = ax_width(1024)

    # SigLIP (So400m/14) 비전 인코더 설정.
    vit_width: int = 1152  # vit_width
    vit_depth: int = 27
    vit_mlp_dim: int = 4304
    vit_num_heads: int = 16
    vit_patch_size: int = 14  # patch -> P = (224/14)^2 = 256 토큰/카메라

    pi05: bool = False  # True 면 adaRMS 로 timestep 주입 + state 를 prefix 로

    dtype: torch.dtype = torch.float32  # 원본은 bfloat16. 베이스 파일은 float32 로 명료하게.

    def paligemma(self) -> GemmaConfig:
        return GEMMA_VARIANTS[self.paligemma_variant]

    def action_expert(self) -> GemmaConfig:
        return GEMMA_VARIANTS[self.action_expert_variant]


# =====================================================================================
# 1. Observation 컨테이너 + 전처리
# =====================================================================================


@dataclasses.dataclass
class Observation:
    """모델 입력. (openpi model.Observation 의 PyTorch 단순화 버전)

    images       : {camera_name: float[B, 3, Himg, Wimg]} , 값 범위 [-1, 1]
    image_masks  : {camera_name: bool[B]} , 해당 카메라가 유효한지
    state        : float[B, Ad] , 로봇 proprioception
    tokenized_prompt      : int[B, L] | None , 언어 토큰 id
    tokenized_prompt_mask : bool[B, L] | None , 유효 토큰 마스크
    """

    images: dict[str, Tensor]
    image_masks: dict[str, Tensor]
    state: Tensor
    tokenized_prompt: Optional[Tensor] = None
    tokenized_prompt_mask: Optional[Tensor] = None


def preprocess_observation(obs: Observation, image_resolution=IMAGE_RESOLUTION) -> Observation:
    """이미지 리사이즈 + 기본 마스크 채우기. (augmentation 은 생략한 최소 버전)

    shape 기호:  B=batch, 3=RGB, Himg/Wimg=이미지 크기.
    각 이미지 [B, 3, h, w] -> [B, 3, Himg, Wimg] 로 통일. 차원 개수는 그대로(4D).

    원본 openpi 는 train 시 RandomCrop/Rotate/ColorJitter augmentation 을 추가하지만,
    베이스 파일에서는 핵심 흐름에 집중하기 위해 resize 와 마스크만 처리한다.
    """
    out_images, out_masks = {}, {}
    for key, image in obs.images.items():
        # image: [B, 3, h, w] -> (필요시) [B, 3, Himg, Wimg]
        if image.shape[-2:] != tuple(image_resolution):
            image = F.interpolate(image, size=image_resolution, mode="bilinear", align_corners=False)
        out_images[key] = image
        if key in obs.image_masks:
            out_masks[key] = obs.image_masks[key]  # [B]
        else:
            out_masks[key] = torch.ones(image.shape[0], dtype=torch.bool, device=image.device)  # [B]
    return Observation(
        images=out_images,
        image_masks=out_masks,
        state=obs.state,
        tokenized_prompt=obs.tokenized_prompt,
        tokenized_prompt_mask=obs.tokenized_prompt_mask,
    )


# =====================================================================================
# 2. 공통 헬퍼: attention mask, positional embedding, RoPE
# =====================================================================================


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


def apply_rope(x: Tensor, positions: Tensor, max_wavelength: int = 10_000) -> Tensor:
    """RoPE (Rotary Position Embedding). openpi _apply_rope 와 동일 (half-split 방식).

    shape 기호:  B=batch, L=시퀀스 길이, H=head 수, Hd=head_dim.
    입력 x [B, L, H, Hd] -> 출력 [B, L, H, Hd] (차원 불변, 값만 회전).
    positions [B, L] 의 절대 위치로 각 토큰 벡터를 회전시킨다.
    """
    head_dim = x.shape[-1]  # Hd
    freq_exponents = (2.0 / head_dim) * torch.arange(head_dim // 2, dtype=torch.float32, device=x.device)  # [Hd/2]
    timescale = max_wavelength**freq_exponents  # [Hd/2]
    radians = positions[..., None].float() / timescale[None, None, :]  # [B,L,1] / [1,1,Hd/2] -> [B, L, Hd/2]
    radians = radians[..., None, :]  # [B, L, 1, Hd/2]  (head 축 H 로 broadcast)
    sin, cos = torch.sin(radians), torch.cos(radians)  # [B, L, 1, Hd/2]
    x1, x2 = torch.split(x.float(), head_dim // 2, dim=-1)  # 각 [B, L, H, Hd/2]
    res = torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)  # [B, L, H, Hd]
    return res.to(x.dtype)


# =====================================================================================
# 3. Gemma 빌딩 블록: RMSNorm(adaRMS), MLP, Attention(mixture), Block
# =====================================================================================


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


class ExpertAttention(nn.Module):
    """여러 expert 의 토큰을 한 시퀀스로 합쳐서 self-attention 하는 모듈. (Pi0 의 핵심)

    shape 기호:
      B   = batch
      Ti  = i 번째 expert 의 입력 토큰 수 (예: prefix=Sp, suffix=Ss)
      T   = 실행되는 expert 토큰을 이어붙인 query 길이 = sum(Ti)
      S   = key 길이 (= T, 단 KV cache 가 있으면 cache_len + T)
      width_i = i 번째 expert 의 width (예: pg_width=2048, ax_width=1024)
      N=num_heads(H), K=num_kv_heads, Hd=head_dim, G=N//K

    각 expert 는 자신만의 q/k/v/out projection 을 갖지만(width 가 다르므로),
    head_dim/num_heads/num_kv_heads 가 같아서 q/k/v 모양이 통일된다 -> 하나로 concat 해
    attention 을 계산 -> 토큰들이 expert 경계를 넘어 서로를 본다.  openpi gemma.Attention 동일.
    """

    def __init__(self, configs: list[GemmaConfig]):
        super().__init__()
        c0 = configs[0]
        assert all(c.head_dim == c0.head_dim for c in configs)  # 공유 조건 강제
        assert all(c.num_heads == c0.num_heads for c in configs)
        assert all(c.num_kv_heads == c0.num_kv_heads for c in configs)
        self.configs = configs
        self.num_heads = c0.num_heads  # N
        self.num_kv_heads = c0.num_kv_heads  # K
        self.head_dim = c0.head_dim  # Hd

        # expert 별 projection (bias 없음, Gemma 관례). width_i -> (head 수)*Hd
        self.q_proj = nn.ModuleList([nn.Linear(c.width, c.num_heads * c.head_dim, bias=False) for c in configs])  # width_i -> N*Hd
        self.k_proj = nn.ModuleList([nn.Linear(c.width, c.num_kv_heads * c.head_dim, bias=False) for c in configs])  # width_i -> K*Hd
        self.v_proj = nn.ModuleList([nn.Linear(c.width, c.num_kv_heads * c.head_dim, bias=False) for c in configs])  # width_i -> K*Hd
        self.o_proj = nn.ModuleList([nn.Linear(c.num_heads * c.head_dim, c.width, bias=False) for c in configs])  # N*Hd -> width_i

    def forward(
        self,
        xs: list[Optional[Tensor]],  # expert 별 입력 [B, Ti, width_i] 또는 None
        positions: Tensor,  # [B, T]  (실행되는 expert 들의 토큰을 이어붙인 전체 위치)
        attn_mask: Tensor,  # bool[B, T, S]
        kv_cache: Optional[tuple[Tensor, Tensor]] = None,  # (cache_k, cache_v) 각 [B, cache_len, K, Hd]
    ):
        H, Hd = self.num_heads, self.head_dim  # H=N(query head), Hd=head_dim
        K = self.num_kv_heads  # K=kv head

        # 1) 각 expert 의 q/k/v 를 계산하고 시퀀스 축(dim=1)으로 이어붙인다.
        qs, ks, vs = [], [], []
        for i, x in enumerate(xs):
            if x is None:  # 이 expert 는 이번에 실행 안 함 (예: cache 채울 때 suffix=None)
                continue
            B, T, _ = x.shape  # x: [B, Ti, width_i]
            # 각 줄: proj 로 채널 변환 후 view 로 head 축 분리 (2번 변형)
            qs.append(self.q_proj[i](x).view(B, T, H, Hd))  # x[B,Ti,width_i] -proj-> [B,Ti,N*Hd] -view-> [B,Ti,N,Hd]
            ks.append(self.k_proj[i](x).view(B, T, K, Hd))  # x[B,Ti,width_i] -proj-> [B,Ti,K*Hd] -view-> [B,Ti,K,Hd]
            vs.append(self.v_proj[i](x).view(B, T, K, Hd))  # x[B,Ti,width_i] -proj-> [B,Ti,K*Hd] -view-> [B,Ti,K,Hd]
        q = torch.cat(qs, dim=1)  # [B, T, N, Hd]   (T = sum(Ti))
        k = torch.cat(ks, dim=1)  # [B, T, K, Hd]
        v = torch.cat(vs, dim=1)  # [B, T, K, Hd]

        # 2) RoPE 적용(차원 불변) 후 query scaling (1/sqrt(Hd)).
        q = apply_rope(q, positions)  # [B, T, N, Hd]
        k = apply_rope(k, positions)  # [B, T, K, Hd]
        q = q * (Hd**-0.5)

        # 3) KV cache (inference 의 prefix 재사용): 이전 k/v 를 시퀀스 앞(dim=1)에 붙인다.
        if kv_cache is not None:
            cache_k, cache_v = kv_cache  # 각 [B, cache_len, K, Hd]
            k = torch.cat([cache_k, k], dim=1)  # [B, S, K, Hd]   (S = cache_len + T)
            v = torch.cat([cache_v, v], dim=1)  # [B, S, K, Hd]
        new_kv_cache = (k, v)  # 다음을 위해 반환 (prefix pass 에서 저장됨)

        # 4) GQA: kv head(K) 를 query head(N) 수에 맞게 복제.
        if K != H:
            g = H // K  # G
            k = k.repeat_interleave(g, dim=2)  # [B, S, K, Hd] -> [B, S, N, Hd]
            v = v.repeat_interleave(g, dim=2)  # [B, S, N, Hd]

        # 5) attention logits (float32 로 안정적으로). b=batch, t=query, s=key, h=head(N), d=head_dim(Hd)
        logits = torch.einsum("bthd,bshd->bhts", q.float(), k.float())  # [B, N, T, S]

        big_neg = -2.3819763e38
        mask = attn_mask[:, None, :, :]  # [B, T, S] -> [B, 1, T, S]  (head 축 broadcast)
        logits = torch.where(mask, logits, torch.full_like(logits, big_neg))  # [B, N, T, S]
        probs = torch.softmax(logits, dim=-1).to(v.dtype)  # [B, N, T, S]  (key 축 정규화)

        encoded = torch.einsum("bhts,bshd->bthd", probs, v)  # [B, N, T, S] x [B, S, N, Hd] -> [B, T, N, Hd]
        B, T = encoded.shape[:2]
        encoded = encoded.reshape(B, T, H * Hd)  # [B, T, N, Hd] -> [B, T, N*Hd]  (head 들 합침)

        # 6) expert 별 out projection 으로 다시 분리. 각 expert 가 자기 슬라이스 [B, Ti, N*Hd] 를 가져가
        #    자기 width_i 로 복귀 [B, Ti, width_i].
        out, start = [], 0
        for i, x in enumerate(xs):
            if x is None:
                out.append(None)
                continue
            end = start + x.shape[1]  # Ti 만큼 슬라이스
            # encoded[B,T,N*Hd] --슬라이스 [:,start:end] (T->Ti)--> [B,Ti,N*Hd] --o_proj (N*Hd->width_i)--> [B,Ti,width_i]
            out.append(self.o_proj[i](encoded[:, start:end]))
            start = end
        return out, new_kv_cache  # out: list of [B, Ti, width_i]


class GemmaBlock(nn.Module):
    """트랜스포머 블록 하나. 각 expert 에 대해 pre-norm + gated residual.

    shape 기호:  B=batch, Ti=expert i 토큰 수, width_i=expert i width.
    각 expert 의 토큰 [B, Ti, width_i] 는 블록을 지나도 모양 불변 (값만 갱신).
    attention 단계에서만 expert 들이 하나로 합쳐졌다 다시 분리된다.

    흐름 (expert i 마다):
       h, gate = pre_attn_norm(x, cond)        # [B,Ti,width_i] -> [B,Ti,width_i], gate[B,1,width_i]
       a       = attention(h ... 합쳐서)        # [B,Ti,width_i]
       x       = x + a * gate                   # gate 는 adaRMS 일 때만, 아니면 None=일반 합
       h, gate = pre_ffw_norm(x, cond)
       m       = mlp(h)                         # [B,Ti,width_i]
       x       = x + m * gate

    openpi gemma.Block 와 동일.
    """

    def __init__(self, configs: list[GemmaConfig], use_adarms: list[bool]):
        super().__init__()
        self.configs = configs
        self.attn = ExpertAttention(configs)
        # expert 별 norm: adaRMS 쓰는 expert 는 cond_dim=width 로 modulation 생성.
        self.pre_attn_norm = nn.ModuleList(
            [RMSNorm(c.width, c.width if use_adarms[i] else None) for i, c in enumerate(configs)]
        )
        self.pre_ffw_norm = nn.ModuleList(
            [RMSNorm(c.width, c.width if use_adarms[i] else None) for i, c in enumerate(configs)]
        )
        self.mlp = nn.ModuleList([GemmaMLP(c.width, c.mlp_dim) for c in configs])

    @staticmethod
    def _gated_residual(x, y, gate):
        # x,y: [B,Ti,width_i] ,  gate: [B,1,width_i] 또는 None
        if x is None:
            return None
        if gate is None:
            return x + y  # 일반 residual
        return x + y * gate  # adaRMS gated residual

    def forward(self, xs, positions, attn_mask, adarms_cond, kv_cache=None):
        # xs: list of [B, Ti, width_i] 또는 None
        # --- attention ---
        pre, gates = [], []
        for i, x in enumerate(xs):
            if x is None:
                pre.append(None)
                gates.append(None)
                continue
            h, gate = self.pre_attn_norm[i](x, adarms_cond[i])  # [B,Ti,width_i], gate[B,1,width_i]|None
            pre.append(h)
            gates.append(gate)
        attn_out, new_kv = self.attn(pre, positions, attn_mask, kv_cache)  # attn_out: list of [B,Ti,width_i]
        xs = [self._gated_residual(x, y, g) for x, y, g in zip(xs, attn_out, gates)]  # [B,Ti,width_i]

        # --- feed forward ---
        out, gates = [], []
        for i, x in enumerate(xs):
            if x is None:
                out.append(None)
                gates.append(None)
                continue
            h, gate = self.pre_ffw_norm[i](x, adarms_cond[i])  # [B,Ti,width_i]
            out.append(self.mlp[i](h))  # [B,Ti,width_i]
            gates.append(gate)
        xs = [self._gated_residual(x, y, g) for x, y, g in zip(xs, out, gates)]  # [B,Ti,width_i]
        return xs, new_kv


class GemmaMixture(nn.Module):
    """여러 expert 를 묶은 Gemma 트랜스포머 (= Pi0 의 LLM backbone).

    shape 기호:  B=batch, Ti=expert i 토큰 수, T=sum(Ti)=전체 query 길이, S=key 길이,
                 width_i=expert i width, depth=레이어 수.

    - expert 0 (PaliGemma) 만 token embedding table 을 가진다 (언어 토큰 임베딩용).
    - 모든 expert 는 depth 가 같고, 레이어마다 attention 을 공유한다.
    openpi gemma.Module 와 동일.
    """

    def __init__(self, configs: list[GemmaConfig], use_adarms: list[bool]):
        super().__init__()
        assert all(c.depth == configs[0].depth for c in configs)
        self.configs = configs
        self.depth = configs[0].depth

        # 언어 토큰 임베딩 (expert 0 = PaliGemma 의 width = pg_width). vocab -> pg_width
        self.embedder = nn.Embedding(PALIGEMMA_VOCAB_SIZE, configs[0].width)
        self.layers = nn.ModuleList([GemmaBlock(configs, use_adarms) for _ in range(self.depth)])
        self.final_norm = nn.ModuleList(
            [RMSNorm(c.width, c.width if use_adarms[i] else None) for i, c in enumerate(configs)]
        )

    def embed_language(self, tokens: Tensor) -> Tensor:
        """언어 토큰 id -> 임베딩. Gemma 관례로 sqrt(width) 스케일.

        tokens [B, L] (정수 id) -> [B, L, pg_width] (실수 임베딩).
        """
        emb = self.embedder(tokens)  # [B, L] -> [B, L, pg_width]
        return emb * math.sqrt(self.configs[0].width)  # [B, L, pg_width]

    def forward(self, embedded, positions, attn_mask, adarms_cond=None, kv_cache=None):
        """
        embedded    : list[Tensor|None]  expert 별 입력 토큰 [B, Ti, width_i]
        positions   : [B, T]
        attn_mask   : bool[B, T, S]
        adarms_cond : list[Tensor|None]  expert 별 adaRMS 조건 (timestep emb [B, width_i])
        kv_cache    : list(길이 depth) of (k, v)  또는 None
        반환        : (outputs list[Tensor|None] 각 [B, Ti, width_i],
                       new_kv_cache list(길이 depth) of (k,v))
        """
        if adarms_cond is None:
            adarms_cond = [None] * len(self.configs)
        new_kv_cache = []
        xs = list(embedded)  # list of [B, Ti, width_i]
        for layer_idx, layer in enumerate(self.layers):
            layer_cache = kv_cache[layer_idx] if kv_cache is not None else None  # 이 레이어의 (k,v)
            xs, kv = layer(xs, positions, attn_mask, adarms_cond, layer_cache)  # [B,Ti,width_i] 유지
            new_kv_cache.append(kv)  # 레이어별 (k,v) 누적
        # 마지막 norm (차원 불변)
        outputs = [self.final_norm[i](x, adarms_cond[i])[0] if x is not None else None for i, x in enumerate(xs)]
        return outputs, new_kv_cache  # outputs: list of [B, Ti, width_i]


# =====================================================================================
# 4. SigLIP Vision Transformer (이미지 -> 토큰)
# =====================================================================================


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


class SiglipVisionModel(nn.Module):
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

    def __init__(self, config: Pi0Config):
        super().__init__()
        self.config = config
        w = config.vit_width  # vit_width
        h_patches = IMAGE_RESOLUTION[0] // config.vit_patch_size  # 224//14 = 16
        num_patches = h_patches * h_patches  # P = 256

        self.patch_embed = nn.Conv2d(3, w, kernel_size=config.vit_patch_size, stride=config.vit_patch_size)  # 3 -> vit_width
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, w) * (1.0 / math.sqrt(w)))  # [1, P, vit_width]
        self.blocks = nn.ModuleList(
            [VitEncoderBlock(w, config.vit_num_heads, config.vit_mlp_dim) for _ in range(config.vit_depth)]
        )
        self.norm = nn.LayerNorm(w)
        # vit width -> paligemma width 로 투영 (LLM 토큰 차원에 맞춤). vit_width -> pg_width
        self.head = nn.Linear(w, config.paligemma().width)

    def forward(self, image: Tensor) -> Tensor:
        """image: [B, 3, 224, 224] in [-1,1]  ->  tokens: [B, P, pg_width]"""
        x = self.patch_embed(image)  # [B, 3, 224, 224] -> [B, vit_width, 16, 16]
        x = x.flatten(2).transpose(1, 2)  # [B, vit_width, 16, 16] -> [B, vit_width, 256] -> [B, P=256, vit_width]
        x = x + self.pos_embed  # [B, P, vit_width]  (위치 임베딩 더하기)
        for block in self.blocks:
            x = block(x)  # [B, P, vit_width] (불변)
        x = self.norm(x)  # [B, P, vit_width]
        return self.head(x)  # head (vit_width->pg_width): [B, P, vit_width] -> [B, P, pg_width]


# =====================================================================================
# 5. Pi0 모델 (flow matching VLA)
# =====================================================================================


class Pi0(nn.Module):
    """Pi0 Vision-Language-Action 모델. openpi Pi0 / PI0Pytorch 와 동일한 알고리즘."""

    def __init__(self, config: Pi0Config):
        super().__init__()
        self.config = config
        self.pi05 = config.pi05
        pg_cfg = config.paligemma()  # width = pg_width
        ax_cfg = config.action_expert()  # width = ax_width
        self.action_horizon = config.action_horizon  # Ah
        self.action_dim = config.action_dim  # Ad

        # 비전 인코더 + LLM mixture (PaliGemma + Action Expert).
        self.vision = SiglipVisionModel(config)
        # pi05 면 action expert(=index 1)만 adaRMS 사용.
        use_adarms = [False, True] if self.pi05 else [False, False]
        self.llm = GemmaMixture([pg_cfg, ax_cfg], use_adarms=use_adarms)

        # action <-> expert width projection.
        self.action_in_proj = nn.Linear(config.action_dim, ax_cfg.width)  # Ad -> ax_width
        self.action_out_proj = nn.Linear(ax_cfg.width, config.action_dim)  # ax_width -> Ad

        if self.pi05:
            # timestep emb -> adaRMS 조건 으로 변환하는 MLP. ax_width -> ax_width
            self.time_mlp_in = nn.Linear(ax_cfg.width, ax_cfg.width)
            self.time_mlp_out = nn.Linear(ax_cfg.width, ax_cfg.width)
        else:
            # state 토큰 투영 + (action, time) 융합 MLP.
            self.state_proj = nn.Linear(config.action_dim, ax_cfg.width)  # Ad -> ax_width
            self.action_time_mlp_in = nn.Linear(2 * ax_cfg.width, ax_cfg.width)  # 2*ax_width -> ax_width
            self.action_time_mlp_out = nn.Linear(ax_cfg.width, ax_cfg.width)  # ax_width -> ax_width

    # ---------------------------------------------------------------------------------
    # prefix(이미지+언어) / suffix(state+action+time) 토큰 임베딩
    # ---------------------------------------------------------------------------------

    def embed_prefix(self, obs: Observation):
        """이미지(SigLIP) + 언어 토큰을 임베딩. 모두 서로 full attention.

        shape 기호:  B=batch, P=카메라당 패치 수(256), ncam=카메라 수, L=언어 토큰 수,
                     Sp=prefix 길이=P*ncam+L, pg_width=PaliGemma width(2048).
        반환:
          tokens   : [B, Sp, pg_width]   (이미지 토큰들 + 언어 토큰들을 이어붙임)
          pad_mask : [B, Sp]             유효 토큰 마스크
          ar_mask  : [Sp]                블록 경계 (prefix 는 전부 0 = 서로 full attention)
        """
        tokens, pad_masks, ar_mask = [], [], []
        for name in obs.images:
            img_tokens = self.vision(obs.images[name])  # [B, 3, 224, 224] -> [B, P, pg_width]
            B, P = img_tokens.shape[:2]
            tokens.append(img_tokens)  # [B, P, pg_width]
            pad_masks.append(obs.image_masks[name][:, None].expand(B, P))  # [B] --[:,None]--> [B,1] --expand--> [B,P]
            ar_mask += [0] * P  # 이미지 토큰끼리 full attention (블록 경계 아님)

        if obs.tokenized_prompt is not None:
            lang = self.llm.embed_language(obs.tokenized_prompt)  # [B, L] -> [B, L, pg_width]
            tokens.append(lang)
            pad_masks.append(obs.tokenized_prompt_mask)  # [B, L]
            ar_mask += [0] * lang.shape[1]  # 이미지+언어 사이 full attention

        tokens = torch.cat(tokens, dim=1)  # [B, Sp, pg_width]   (Sp = P*ncam + L)
        pad_masks = torch.cat(pad_masks, dim=1)  # [B, Sp]
        ar_mask = torch.tensor(ar_mask, dtype=torch.bool, device=tokens.device)  # [Sp]
        return tokens, pad_masks, ar_mask

    def embed_suffix(self, state: Tensor, noisy_actions: Tensor, timestep: Tensor):
        """state(옵션) + noisy action + timestep 을 action expert 토큰으로 임베딩.

        shape 기호:  B=batch, Ah=action_horizon, Ad=action_dim, ax_width=Action Expert width(1024),
                     Ss=suffix 길이 = (pi0: 1+Ah) / (pi05: Ah).
        입력:
          state         : [B, Ad]
          noisy_actions : [B, Ah, Ad]
          timestep      : [B]
        반환:
          tokens      : [B, Ss, ax_width]
          pad_mask    : [B, Ss]
          ar_mask     : [Ss]                  블록 경계
          adarms_cond : [B, ax_width] | None  (pi05 면 timestep emb, 아니면 None)
        """
        tokens, pad_masks, ar_mask = [], [], []
        B = noisy_actions.shape[0]
        device = noisy_actions.device

        if not self.pi05:
            # state 를 하나의 연속 토큰으로. (pi05 는 state 를 prefix 언어토큰으로 넣으므로 생략)
            state_tok = self.state_proj(state)[:, None, :]  # [B, Ad] -> [B, ax_width] -> [B, 1, ax_width]
            tokens.append(state_tok)
            pad_masks.append(torch.ones(B, 1, dtype=torch.bool, device=device))  # [B, 1]
            ar_mask += [1]  # prefix/state 는 action 을 못 본다 = 블록 경계

        # timestep -> sine-cosine 임베딩 (민감 구간 [0,1]). [B] -> [B, ax_width]
        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        action_emb = self.action_in_proj(noisy_actions)  # [B, Ah, Ad] -> [B, Ah, ax_width]

        if self.pi05:
            # timestep 은 토큰이 아니라 adaRMS 조건으로 주입. [B, ax_width] -> [B, ax_width]
            te = F.silu(self.time_mlp_in(time_emb))
            te = F.silu(self.time_mlp_out(te))
            action_time_tokens = action_emb  # [B, Ah, ax_width]
            adarms_cond = te  # [B, ax_width]
        else:
            # (action, time) 을 concat 후 MLP 로 융합 (adaRMS 없음).
            time_tok = time_emb[:, None, :].expand_as(action_emb)  # [B,ax_width] --[:,None,:]--> [B,1,ax_width] --expand--> [B,Ah,ax_width]
            x = torch.cat([action_emb, time_tok], dim=-1)  # [B,Ah,ax_width] ⊕ [B,Ah,ax_width] -> [B, Ah, 2*ax_width]
            x = F.silu(self.action_time_mlp_in(x))  # action_time_mlp_in (2*ax_width->ax_width): [B,Ah,2*ax_width] -> [B,Ah,ax_width]
            action_time_tokens = self.action_time_mlp_out(x)  # ax_width->ax_width: [B,Ah,ax_width] -> [B,Ah,ax_width]
            adarms_cond = None

        tokens.append(action_time_tokens)  # [B, Ah, ax_width]
        pad_masks.append(torch.ones(B, action_time_tokens.shape[1], dtype=torch.bool, device=device))  # [B, Ah]
        # action chunk 의 첫 토큰만 블록 경계(=이전 것들은 action 을 못 봄), 나머지는 서로 full.
        ar_mask += [1] + [0] * (self.action_horizon - 1)

        tokens = torch.cat(tokens, dim=1)  # [B, Ss, ax_width]   (Ss = 1+Ah 또는 Ah)
        pad_masks = torch.cat(pad_masks, dim=1)  # [B, Ss]
        ar_mask = torch.tensor(ar_mask, dtype=torch.bool, device=device)  # [Ss]
        return tokens, pad_masks, ar_mask, adarms_cond

    # ---------------------------------------------------------------------------------
    # 학습: flow matching loss
    # ---------------------------------------------------------------------------------

    def compute_loss(self, obs: Observation, actions: Tensor, noise=None, time=None) -> Tensor:
        """flow matching MSE loss.

        shape 기호:  B=batch, Ah=action_horizon, Ad=action_dim, Sp/Ss/St=prefix/suffix/전체 길이.
        입력 actions: [B, Ah, Ad].  반환 loss: [B, Ah, Ad] (reduction 없음).

        x_t = t*noise + (1-t)*actions ,  u_t = noise - actions ,  v_t = model(...)
        loss = ||v_t - u_t||^2
        """
        obs = preprocess_observation(obs)
        B = actions.shape[0]
        device = actions.device

        if noise is None:
            noise = torch.randn_like(actions)  # [B, Ah, Ad]
        if time is None:
            # Beta(1.5,1) 분포에서 timestep 샘플 (작은 t 에 더 집중). [B], 범위 [0.001, 1.0]
            time = torch.distributions.Beta(1.5, 1.0).sample((B,)).to(device) * 0.999 + 0.001

        t = time[:, None, None]  # [B] -> [B, 1, 1]  (broadcast 용)
        x_t = t * noise + (1.0 - t) * actions  # [B, Ah, Ad]  (노이즈 섞인 action)
        u_t = noise - actions  # [B, Ah, Ad]  (정답 velocity)

        # prefix + suffix 를 한 번에 forward.
        prefix_tok, prefix_mask, prefix_ar = self.embed_prefix(obs)  # [B,Sp,pg_width], [B,Sp], [Sp]
        suffix_tok, suffix_mask, suffix_ar, adarms_cond = self.embed_suffix(obs.state, x_t, time)  # [B,Ss,ax_width], ...

        pad_mask = torch.cat([prefix_mask, suffix_mask], dim=1)  # [B, St]   (St = Sp + Ss)
        ar_mask = torch.cat([prefix_ar, suffix_ar], dim=0)  # [St]
        attn_mask = make_attn_mask(pad_mask, ar_mask)  # [B, St, St]
        positions = torch.cumsum(pad_mask.int(), dim=1) - 1  # [B, St]  각 토큰 절대 위치

        # 두 expert 입력을 같이 넘김 (prefix=expert0, suffix=expert1). 출력도 expert 별 list.
        outputs, _ = self.llm(
            [prefix_tok, suffix_tok], positions, attn_mask, adarms_cond=[None, adarms_cond]
        )
        suffix_out = outputs[1]  # action expert 출력 [B, Ss, ax_width]
        # 뒤쪽 Ah 개(=action 토큰)만 골라 Ad 로 투영 -> 예측 velocity. (한 줄에 슬라이스+투영 2번 변형)
        #   suffix_out[B,Ss,ax_width] --슬라이스 [:,-Ah:] (Ss->Ah)--> [B,Ah,ax_width]
        #                             --action_out_proj (ax_width->Ad)--> [B,Ah,Ad]
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])
        return F.mse_loss(v_t, u_t, reduction="none")  # (v_t[B,Ah,Ad], u_t[B,Ah,Ad]) -> [B, Ah, Ad]

    # ---------------------------------------------------------------------------------
    # 추론: KV cache + Euler 적분으로 action 생성
    # ---------------------------------------------------------------------------------

    @torch.no_grad()
    def sample_actions(self, obs: Observation, num_steps: int = 10, noise=None) -> Tensor:
        """noise 에서 시작해 flow 를 따라 적분하여 action chunk 생성.

        shape 기호:  B=batch, Ah=action_horizon, Ad=action_dim, Sp=prefix 길이.
        반환: [B, Ah, Ad]  (생성된 action chunk).
        """
        obs = preprocess_observation(obs)
        B = obs.state.shape[0]
        device = obs.state.device
        if noise is None:
            noise = torch.randn(B, self.action_horizon, self.action_dim, device=device)  # [B, Ah, Ad]

        # 1) prefix(이미지+언어)를 한 번 forward 하여 KV cache 채움. (suffix=None)
        prefix_tok, prefix_mask, prefix_ar = self.embed_prefix(obs)  # [B,Sp,pg_width], [B,Sp], [Sp]
        prefix_attn = make_attn_mask(prefix_mask, prefix_ar)  # [B, Sp, Sp]
        prefix_pos = torch.cumsum(prefix_mask.int(), dim=1) - 1  # [B, Sp]
        # kv_cache: list(길이 depth) of (k,v) 각 [B, Sp, K, Hd]
        _, kv_cache = self.llm([prefix_tok, None], prefix_pos, prefix_attn)

        # 2) Euler 적분: t=1 (noise) -> t=0 (action).  num_steps 번 반복.
        dt = -1.0 / num_steps
        x_t = noise  # [B, Ah, Ad]
        t = 1.0
        while t >= -dt / 2:  # floating point 안전 마진
            time = torch.full((B,), t, dtype=torch.float32, device=device)  # [B]
            v_t = self._denoise_step(obs.state, prefix_mask, kv_cache, x_t, time)  # [B, Ah, Ad]
            x_t = x_t + dt * v_t  # [B, Ah, Ad]  (flow 따라 한 스텝 이동)
            t += dt
        return x_t  # [B, Ah, Ad]

    def _denoise_step(self, state, prefix_mask, kv_cache, x_t, time) -> Tensor:
        """한 번의 denoising step. suffix 만 forward (prefix 는 cache 재사용).

        shape 기호:  B=batch, Sp=prefix 길이, Ss=suffix 길이, Ah=action_horizon, Ad=action_dim.
        입력 x_t: [B, Ah, Ad], time: [B].  반환 velocity: [B, Ah, Ad].
        """
        suffix_tok, suffix_mask, suffix_ar, adarms_cond = self.embed_suffix(state, x_t, time)  # [B,Ss,ax_width], ...
        B = prefix_mask.shape[0]
        prefix_len = prefix_mask.shape[1]  # Sp
        suffix_len = suffix_mask.shape[1]  # Ss

        # suffix query 가 prefix(전체) + suffix(블록) 를 보는 mask.
        prefix_2d = prefix_mask[:, None, :].expand(B, suffix_len, prefix_len)  # [B, Ss, Sp]  (suffix->prefix 전부 허용)
        suffix_2d = make_attn_mask(suffix_mask, suffix_ar)  # [B, Ss, Ss]  (suffix 내부 블록 마스크)
        full_mask = torch.cat([prefix_2d, suffix_2d], dim=2)  # [B, Ss, Sp+Ss]

        # suffix 토큰의 절대 위치 = prefix 길이 + suffix 내 누적. (cache 된 prefix 와 위치 안 겹치게)
        prefix_offset = prefix_mask.int().sum(dim=-1)[:, None]  # [B,Sp] --sum(dim=-1)--> [B] --[:,None]--> [B,1] (각 배치 prefix 길이=Sp)
        positions = prefix_offset + torch.cumsum(suffix_mask.int(), dim=1) - 1  # [B,1] + [B,Ss] -> [B, Ss]

        # suffix(expert1)만 실행, prefix(expert0)=None. cache 로 prefix K/V 를 attention 에 붙임.
        outputs, _ = self.llm(
            [None, suffix_tok], positions, full_mask, adarms_cond=[None, adarms_cond], kv_cache=kv_cache
        )
        # outputs[1][B,Ss,ax_width] --슬라이스 [:,-Ah:] (Ss->Ah)--> [B,Ah,ax_width]
        suffix_out = outputs[1][:, -self.action_horizon :]
        return self.action_out_proj(suffix_out)  # action_out_proj (ax_width->Ad): [B,Ah,ax_width] -> [B,Ah,Ad]


# =====================================================================================
# 6. 스모크 테스트 (작은 dummy config 로 forward/sample 동작 확인)
# =====================================================================================

if __name__ == "__main__":
    torch.manual_seed(0)

    # 빠른 테스트용 아주 작은 설정. (실제 Pi0 는 gemma_2b + gemma_300m + So400m/14)
    cfg = Pi0Config(
        action_dim=8,
        action_horizon=4,
        max_token_len=6,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        vit_width=64,
        vit_depth=2,
        vit_mlp_dim=128,
        vit_num_heads=4,
        vit_patch_size=14,
        pi05=False,
    )

    model = Pi0(cfg)
    model.eval()

    B = 2
    obs = Observation(
        images={
            "base_0_rgb": torch.randn(B, 3, 224, 224),
            "left_wrist_0_rgb": torch.randn(B, 3, 224, 224),
        },
        image_masks={
            "base_0_rgb": torch.ones(B, dtype=torch.bool),
            "left_wrist_0_rgb": torch.ones(B, dtype=torch.bool),
        },
        state=torch.randn(B, cfg.action_dim),
        tokenized_prompt=torch.randint(0, 1000, (B, cfg.max_token_len)),
        tokenized_prompt_mask=torch.ones(B, cfg.max_token_len, dtype=torch.bool),
    )
    actions = torch.randn(B, cfg.action_horizon, cfg.action_dim)

    loss = model.compute_loss(obs, actions)
    print("loss shape:", tuple(loss.shape), "| mean:", loss.mean().item())

    sampled = model.sample_actions(obs, num_steps=5)
    print("sampled actions shape:", tuple(sampled.shape))

    # pi05 변형도 확인.
    cfg05 = dataclasses.replace(cfg, pi05=True, max_token_len=6)
    model05 = Pi0(cfg05).eval()
    loss05 = model05.compute_loss(obs, actions)
    sampled05 = model05.sample_actions(obs, num_steps=5)
    print("[pi05] loss:", tuple(loss05.shape), "| sampled:", tuple(sampled05.shape))

    print("OK: pi0-base forward/sample 동작 확인 완료")
