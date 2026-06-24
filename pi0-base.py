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

    width: int  # d_model (토큰 임베딩 차원)
    depth: int  # 트랜스포머 레이어 수 (모든 expert 동일해야 함)
    mlp_dim: int  # FeedForward hidden 차원
    num_heads: int  # query head 수
    num_kv_heads: int  # key/value head 수 (GQA; 1 이면 multi-query)
    head_dim: int  # head 당 차원


# openpi gemma.py 의 variant 들.
GEMMA_VARIANTS = {
    "dummy": GemmaConfig(width=64, depth=4, mlp_dim=128, num_heads=8, num_kv_heads=1, head_dim=16),
    "gemma_300m": GemmaConfig(width=1024, depth=18, mlp_dim=4096, num_heads=8, num_kv_heads=1, head_dim=256),
    "gemma_2b": GemmaConfig(width=2048, depth=18, mlp_dim=16_384, num_heads=8, num_kv_heads=1, head_dim=256),
}

PALIGEMMA_VOCAB_SIZE = 257_152
IMAGE_RESOLUTION = (224, 224)


@dataclasses.dataclass
class Pi0Config:
    """Pi0 모델 설정. (openpi Pi0Config 의 핵심 필드만)"""

    action_dim: int = 32  # 로봇 action 차원
    action_horizon: int = 50  # 한 번에 예측하는 action step 수 (chunk 길이)
    max_token_len: int = 48  # 언어 프롬프트 최대 토큰 수 (pi05 면 200)

    paligemma_variant: str = "gemma_2b"  # prefix expert (VLM)
    action_expert_variant: str = "gemma_300m"  # suffix expert (action)

    # SigLIP (So400m/14) 비전 인코더 설정.
    vit_width: int = 1152
    vit_depth: int = 27
    vit_mlp_dim: int = 4304
    vit_num_heads: int = 16
    vit_patch_size: int = 14

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

    images       : {camera_name: float[B, 3, H, W]} , 값 범위 [-1, 1]
    image_masks  : {camera_name: bool[B]} , 해당 카메라가 유효한지
    state        : float[B, action_dim] , 로봇 proprioception
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

    원본 openpi 는 train 시 RandomCrop/Rotate/ColorJitter augmentation 을 추가하지만,
    베이스 파일에서는 핵심 흐름에 집중하기 위해 resize 와 마스크만 처리한다.
    """
    out_images, out_masks = {}, {}
    for key, image in obs.images.items():
        if image.shape[-2:] != tuple(image_resolution):
            image = F.interpolate(image, size=image_resolution, mode="bilinear", align_corners=False)
        out_images[key] = image
        if key in obs.image_masks:
            out_masks[key] = obs.image_masks[key]
        else:
            out_masks[key] = torch.ones(image.shape[0], dtype=torch.bool, device=image.device)
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
    mask_ar = mask_ar.to(torch.int32).broadcast_to(input_mask.shape)
    cumsum = torch.cumsum(mask_ar, dim=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]  # [B, query, key]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return attn_mask & valid_mask.bool()


def posemb_sincos(pos: Tensor, embedding_dim: int, min_period: float, max_period: float) -> Tensor:
    """스칼라 위치(여기선 flow matching timestep)에 대한 sine-cosine 임베딩.

    openpi posemb_sincos / create_sinusoidal_pos_embedding 와 동일.
    pos: float[B] -> [B, embedding_dim]
    """
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")
    fraction = torch.linspace(0.0, 1.0, embedding_dim // 2, dtype=torch.float32, device=pos.device)
    period = min_period * (max_period / min_period) ** fraction
    scaling = 1.0 / period * 2 * math.pi  # [embedding_dim//2]
    sinusoid = scaling[None, :] * pos[:, None].float()  # [B, embedding_dim//2]
    return torch.cat([torch.sin(sinusoid), torch.cos(sinusoid)], dim=-1)


def apply_rope(x: Tensor, positions: Tensor, max_wavelength: int = 10_000) -> Tensor:
    """RoPE (Rotary Position Embedding). openpi _apply_rope 와 동일 (half-split 방식).

    x         : [B, L, H, head_dim]   (query 또는 key)
    positions : [B, L]                절대 위치
    """
    head_dim = x.shape[-1]
    freq_exponents = (2.0 / head_dim) * torch.arange(head_dim // 2, dtype=torch.float32, device=x.device)
    timescale = max_wavelength**freq_exponents  # [head_dim//2]
    radians = positions[..., None].float() / timescale[None, None, :]  # [B, L, head_dim//2]
    radians = radians[..., None, :]  # [B, L, 1, head_dim//2]  (head 축으로 broadcast)
    sin, cos = torch.sin(radians), torch.cos(radians)
    x1, x2 = torch.split(x.float(), head_dim // 2, dim=-1)
    res = torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
    return res.to(x.dtype)


# =====================================================================================
# 3. Gemma 빌딩 블록: RMSNorm(adaRMS), MLP, Attention(mixture), Block
# =====================================================================================


class RMSNorm(nn.Module):
    """RMSNorm. cond 가 주어지면 adaptive RMSNorm (adaRMS) 으로 동작.

    - 일반 RMSNorm : normed * (1 + scale)        (scale 은 learnable, zeros 초기화)
    - adaRMS       : cond(=timestep emb) 로부터 scale/shift/gate 를 생성
                     normed * (1 + scale) + shift , 그리고 residual 용 gate 반환

    openpi gemma.RMSNorm 과 동일. variance 는 float32 로 계산.
    """

    def __init__(self, dim: int, adarms_cond_dim: Optional[int] = None):
        super().__init__()
        self.dim = dim
        self.use_adarms = adarms_cond_dim is not None
        if self.use_adarms:
            # cond -> [scale, shift, gate] (각 dim). zeros 초기화 => 처음엔 identity.
            self.modulation = nn.Linear(adarms_cond_dim, dim * 3)
            nn.init.zeros_(self.modulation.weight)
            nn.init.zeros_(self.modulation.bias)
        else:
            self.scale = nn.Parameter(torch.zeros(dim))  # (1 + scale) 형태로 사용

    def forward(self, x: Tensor, cond: Optional[Tensor] = None):
        dtype = x.dtype
        var = x.float().pow(2).mean(dim=-1, keepdim=True)
        normed = x.float() * torch.rsqrt(var + 1e-6)
        if not self.use_adarms:
            normed = normed * (1.0 + self.scale.float())
            return normed.to(dtype), None
        # adaRMS
        mod = self.modulation(cond.to(self.modulation.weight.dtype))  # [B, 3*dim]
        scale, shift, gate = torch.chunk(mod[:, None, :], 3, dim=-1)  # 각 [B, 1, dim]
        normed = normed * (1.0 + scale.float()) + shift.float()
        return normed.to(dtype), gate


class GemmaMLP(nn.Module):
    """Gemma FeedForward: GeGLU.  out = (gelu(x W_gate) * (x W_up)) W_down

    openpi gemma.FeedForward 와 동일 구조 (bias 없음).
    """

    def __init__(self, width: int, hidden_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(width, hidden_dim, bias=False)
        self.up_proj = nn.Linear(width, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, width, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(F.gelu(self.gate_proj(x), approximate="tanh") * self.up_proj(x))


class ExpertAttention(nn.Module):
    """여러 expert 의 토큰을 한 시퀀스로 합쳐서 self-attention 하는 모듈. (Pi0 의 핵심)

    각 expert(=stream)는 자신만의 q/k/v/out projection 을 갖는다 (width 가 다르므로).
    하지만 head_dim, num_heads, num_kv_heads 는 공유하므로 attention logit 은 하나의
    시퀀스로 합쳐서 계산할 수 있다 -> 토큰들이 expert 경계를 넘어 서로를 본다.

    openpi gemma.Attention 와 동일.
    """

    def __init__(self, configs: list[GemmaConfig]):
        super().__init__()
        c0 = configs[0]
        assert all(c.head_dim == c0.head_dim for c in configs)
        assert all(c.num_heads == c0.num_heads for c in configs)
        assert all(c.num_kv_heads == c0.num_kv_heads for c in configs)
        self.configs = configs
        self.num_heads = c0.num_heads
        self.num_kv_heads = c0.num_kv_heads
        self.head_dim = c0.head_dim

        # expert 별 projection (bias 없음, Gemma 관례).
        self.q_proj = nn.ModuleList([nn.Linear(c.width, c.num_heads * c.head_dim, bias=False) for c in configs])
        self.k_proj = nn.ModuleList([nn.Linear(c.width, c.num_kv_heads * c.head_dim, bias=False) for c in configs])
        self.v_proj = nn.ModuleList([nn.Linear(c.width, c.num_kv_heads * c.head_dim, bias=False) for c in configs])
        self.o_proj = nn.ModuleList([nn.Linear(c.num_heads * c.head_dim, c.width, bias=False) for c in configs])

    def forward(
        self,
        xs: list[Optional[Tensor]],  # expert 별 입력 [B, Ti, width_i] 또는 None
        positions: Tensor,  # [B, T_total]  (실행되는 expert 들의 토큰을 이어붙인 전체 위치)
        attn_mask: Tensor,  # bool[B, T_total, S_total]
        kv_cache: Optional[tuple[Tensor, Tensor]] = None,
    ):
        H, Hd = self.num_heads, self.head_dim
        K = self.num_kv_heads

        # 1) 각 expert 의 q/k/v 를 계산하고 시퀀스 축으로 이어붙인다.
        qs, ks, vs = [], [], []
        for i, x in enumerate(xs):
            if x is None:
                continue
            B, T, _ = x.shape
            qs.append(self.q_proj[i](x).view(B, T, H, Hd))
            ks.append(self.k_proj[i](x).view(B, T, K, Hd))
            vs.append(self.v_proj[i](x).view(B, T, K, Hd))
        q = torch.cat(qs, dim=1)  # [B, T, H, Hd]
        k = torch.cat(ks, dim=1)  # [B, T, K, Hd]
        v = torch.cat(vs, dim=1)

        # 2) RoPE 적용 후 query scaling.
        q = apply_rope(q, positions)
        k = apply_rope(k, positions)
        q = q * (Hd**-0.5)

        # 3) KV cache (inference 의 prefix 재사용): 이전 k/v 를 앞에 붙인다.
        if kv_cache is not None:
            cache_k, cache_v = kv_cache
            k = torch.cat([cache_k, k], dim=1)
            v = torch.cat([cache_v, v], dim=1)
        new_kv_cache = (k, v)

        # 4) GQA: kv head 를 query head 수에 맞게 확장.
        if K != H:
            g = H // K
            k = k.repeat_interleave(g, dim=2)  # [B, S, H, Hd]
            v = v.repeat_interleave(g, dim=2)

        # 5) attention logits (float32 로 안정적으로). b=batch, t/s=seq, h=head, d=head_dim
        logits = torch.einsum("bthd,bshd->bhts", q.float(), k.float())  # [B, H, T, S]

        big_neg = -2.3819763e38
        mask = attn_mask[:, None, :, :]  # [B, 1, T, S]
        logits = torch.where(mask, logits, torch.full_like(logits, big_neg))
        probs = torch.softmax(logits, dim=-1).to(v.dtype)  # [B, H, T, S]

        encoded = torch.einsum("bhts,bshd->bthd", probs, v)  # [B, T, H, Hd]
        B, T = encoded.shape[:2]
        encoded = encoded.reshape(B, T, H * Hd)

        # 6) expert 별 out projection 으로 다시 분리.
        out, start = [], 0
        for i, x in enumerate(xs):
            if x is None:
                out.append(None)
                continue
            end = start + x.shape[1]
            out.append(self.o_proj[i](encoded[:, start:end]))
            start = end
        return out, new_kv_cache


class GemmaBlock(nn.Module):
    """트랜스포머 블록 하나. 각 expert 에 대해 pre-norm + gated residual.

    흐름 (expert i 마다):
       h, gate = pre_attn_norm(x, cond)
       a       = attention(h ... 합쳐서)
       x       = x + a * gate                  (gate 는 adaRMS 일 때만, 아니면 None=일반 합)
       h, gate = pre_ffw_norm(x, cond)
       m       = mlp(h)
       x       = x + m * gate

    openpi gemma.Block 와 동일.
    """

    def __init__(self, configs: list[GemmaConfig], use_adarms: list[bool]):
        super().__init__()
        self.configs = configs
        self.attn = ExpertAttention(configs)
        self.pre_attn_norm = nn.ModuleList(
            [RMSNorm(c.width, c.width if use_adarms[i] else None) for i, c in enumerate(configs)]
        )
        self.pre_ffw_norm = nn.ModuleList(
            [RMSNorm(c.width, c.width if use_adarms[i] else None) for i, c in enumerate(configs)]
        )
        self.mlp = nn.ModuleList([GemmaMLP(c.width, c.mlp_dim) for c in configs])

    @staticmethod
    def _gated_residual(x, y, gate):
        if x is None:
            return None
        if gate is None:
            return x + y
        return x + y * gate

    def forward(self, xs, positions, attn_mask, adarms_cond, kv_cache=None):
        # --- attention ---
        pre, gates = [], []
        for i, x in enumerate(xs):
            if x is None:
                pre.append(None)
                gates.append(None)
                continue
            h, gate = self.pre_attn_norm[i](x, adarms_cond[i])
            pre.append(h)
            gates.append(gate)
        attn_out, new_kv = self.attn(pre, positions, attn_mask, kv_cache)
        xs = [self._gated_residual(x, y, g) for x, y, g in zip(xs, attn_out, gates)]

        # --- feed forward ---
        out, gates = [], []
        for i, x in enumerate(xs):
            if x is None:
                out.append(None)
                gates.append(None)
                continue
            h, gate = self.pre_ffw_norm[i](x, adarms_cond[i])
            out.append(self.mlp[i](h))
            gates.append(gate)
        xs = [self._gated_residual(x, y, g) for x, y, g in zip(xs, out, gates)]
        return xs, new_kv


class GemmaMixture(nn.Module):
    """여러 expert 를 묶은 Gemma 트랜스포머 (= Pi0 의 LLM backbone).

    - expert 0 (PaliGemma) 만 token embedding table 을 가진다 (언어 토큰 임베딩용).
    - 모든 expert 는 depth 가 같고, 레이어마다 attention 을 공유한다.

    openpi gemma.Module 와 동일.
    """

    def __init__(self, configs: list[GemmaConfig], use_adarms: list[bool]):
        super().__init__()
        assert all(c.depth == configs[0].depth for c in configs)
        self.configs = configs
        self.depth = configs[0].depth

        # 언어 토큰 임베딩 (expert 0 = PaliGemma 의 width).
        self.embedder = nn.Embedding(PALIGEMMA_VOCAB_SIZE, configs[0].width)
        self.layers = nn.ModuleList([GemmaBlock(configs, use_adarms) for _ in range(self.depth)])
        self.final_norm = nn.ModuleList(
            [RMSNorm(c.width, c.width if use_adarms[i] else None) for i, c in enumerate(configs)]
        )

    def embed_language(self, tokens: Tensor) -> Tensor:
        """언어 토큰 id -> 임베딩. Gemma 관례로 sqrt(width) 스케일."""
        emb = self.embedder(tokens)
        return emb * math.sqrt(self.configs[0].width)

    def forward(self, embedded, positions, attn_mask, adarms_cond=None, kv_cache=None):
        """
        embedded    : list[Tensor|None]  expert 별 입력 토큰 [B, Ti, width_i]
        positions   : [B, T_total]
        attn_mask   : bool[B, T_total, S_total]
        adarms_cond : list[Tensor|None]  expert 별 adaRMS 조건 (timestep emb)
        kv_cache    : list per-layer (k, v) | None
        반환        : (outputs list[Tensor|None], new_kv_cache list per-layer (k,v))
        """
        if adarms_cond is None:
            adarms_cond = [None] * len(self.configs)
        new_kv_cache = []
        xs = list(embedded)
        for layer_idx, layer in enumerate(self.layers):
            layer_cache = kv_cache[layer_idx] if kv_cache is not None else None
            xs, kv = layer(xs, positions, attn_mask, adarms_cond, layer_cache)
            new_kv_cache.append(kv)
        outputs = [self.final_norm[i](x, adarms_cond[i])[0] if x is not None else None for i, x in enumerate(xs)]
        return outputs, new_kv_cache


# =====================================================================================
# 4. SigLIP Vision Transformer (이미지 -> 토큰)
# =====================================================================================


class VitMLP(nn.Module):
    def __init__(self, width: int, mlp_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(width, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, width)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(F.gelu(self.fc1(x), approximate="tanh"))


class VitEncoderBlock(nn.Module):
    """표준 pre-LN 트랜스포머 인코더 블록 (MHSA + MLP). (openpi siglip.Encoder1DBlock)"""

    def __init__(self, width: int, num_heads: int, mlp_dim: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(width)
        self.attn = nn.MultiheadAttention(width, num_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(width)
        self.mlp = VitMLP(width, mlp_dim)

    def forward(self, x: Tensor) -> Tensor:
        y = self.ln1(x)
        y, _ = self.attn(y, y, y, need_weights=False)
        x = x + y
        y = self.ln2(x)
        return x + self.mlp(y)


class SiglipVisionModel(nn.Module):
    """SigLIP ViT (So400m/14). 이미지를 patch 토큰으로 인코딩한 뒤 PaliGemma width 로 투영.

    openpi siglip._Module 의 pool_type="none" 경로:
      patch conv -> + learned posemb -> encoder(depth) -> LayerNorm -> head(Linear)
    head 가 vit_width(1152) -> num_classes(=paligemma width) 로 투영하여 LLM 입력 토큰을 만든다.
    """

    def __init__(self, config: Pi0Config):
        super().__init__()
        self.config = config
        w = config.vit_width
        h_patches = IMAGE_RESOLUTION[0] // config.vit_patch_size
        num_patches = h_patches * h_patches

        self.patch_embed = nn.Conv2d(3, w, kernel_size=config.vit_patch_size, stride=config.vit_patch_size)
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, w) * (1.0 / math.sqrt(w)))
        self.blocks = nn.ModuleList(
            [VitEncoderBlock(w, config.vit_num_heads, config.vit_mlp_dim) for _ in range(config.vit_depth)]
        )
        self.norm = nn.LayerNorm(w)
        # vit width -> paligemma width 로 투영 (LLM 토큰 차원에 맞춤).
        self.head = nn.Linear(w, config.paligemma().width)

    def forward(self, image: Tensor) -> Tensor:
        """image: [B, 3, H, W] in [-1,1]  ->  tokens: [B, num_patches, paligemma_width]"""
        x = self.patch_embed(image)  # [B, w, h', w']
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, w]
        x = x + self.pos_embed
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return self.head(x)


# =====================================================================================
# 5. Pi0 모델 (flow matching VLA)
# =====================================================================================


class Pi0(nn.Module):
    """Pi0 Vision-Language-Action 모델. openpi Pi0 / PI0Pytorch 와 동일한 알고리즘."""

    def __init__(self, config: Pi0Config):
        super().__init__()
        self.config = config
        self.pi05 = config.pi05
        pg_cfg = config.paligemma()
        ax_cfg = config.action_expert()
        self.action_horizon = config.action_horizon
        self.action_dim = config.action_dim

        # 비전 인코더 + LLM mixture (PaliGemma + Action Expert).
        self.vision = SiglipVisionModel(config)
        # pi05 면 action expert(=index 1)만 adaRMS 사용.
        use_adarms = [False, True] if self.pi05 else [False, False]
        self.llm = GemmaMixture([pg_cfg, ax_cfg], use_adarms=use_adarms)

        # action <-> expert width projection.
        self.action_in_proj = nn.Linear(config.action_dim, ax_cfg.width)
        self.action_out_proj = nn.Linear(ax_cfg.width, config.action_dim)

        if self.pi05:
            # timestep -> adaRMS 조건 으로 변환하는 MLP.
            self.time_mlp_in = nn.Linear(ax_cfg.width, ax_cfg.width)
            self.time_mlp_out = nn.Linear(ax_cfg.width, ax_cfg.width)
        else:
            # state 토큰 투영 + (action, time) 융합 MLP.
            self.state_proj = nn.Linear(config.action_dim, ax_cfg.width)
            self.action_time_mlp_in = nn.Linear(2 * ax_cfg.width, ax_cfg.width)
            self.action_time_mlp_out = nn.Linear(ax_cfg.width, ax_cfg.width)

    # ---------------------------------------------------------------------------------
    # prefix(이미지+언어) / suffix(state+action+time) 토큰 임베딩
    # ---------------------------------------------------------------------------------

    def embed_prefix(self, obs: Observation):
        """이미지(SigLIP) + 언어 토큰을 임베딩. 모두 서로 full attention.

        반환: tokens [B, S, pg_width], pad_mask [B, S], ar_mask [S]
        """
        tokens, pad_masks, ar_mask = [], [], []
        for name in obs.images:
            img_tokens = self.vision(obs.images[name])  # [B, P, pg_width]
            B, P = img_tokens.shape[:2]
            tokens.append(img_tokens)
            pad_masks.append(obs.image_masks[name][:, None].expand(B, P))
            ar_mask += [0] * P  # 이미지 토큰끼리 full attention (블록 경계 아님)

        if obs.tokenized_prompt is not None:
            lang = self.llm.embed_language(obs.tokenized_prompt)  # [B, L, pg_width]
            tokens.append(lang)
            pad_masks.append(obs.tokenized_prompt_mask)
            ar_mask += [0] * lang.shape[1]  # 이미지+언어 사이 full attention

        tokens = torch.cat(tokens, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        ar_mask = torch.tensor(ar_mask, dtype=torch.bool, device=tokens.device)
        return tokens, pad_masks, ar_mask

    def embed_suffix(self, state: Tensor, noisy_actions: Tensor, timestep: Tensor):
        """state(옵션) + noisy action + timestep 을 action expert 토큰으로 임베딩.

        반환: tokens [B, S, ax_width], pad_mask [B, S], ar_mask [S], adarms_cond [B, ax_width]|None
        """
        tokens, pad_masks, ar_mask = [], [], []
        B = noisy_actions.shape[0]
        device = noisy_actions.device

        if not self.pi05:
            # state 를 하나의 연속 토큰으로. (pi05 는 state 를 prefix 언어토큰으로 넣으므로 생략)
            state_tok = self.state_proj(state)[:, None, :]
            tokens.append(state_tok)
            pad_masks.append(torch.ones(B, 1, dtype=torch.bool, device=device))
            ar_mask += [1]  # prefix/state 는 action 을 못 본다 = 블록 경계

        # timestep -> sine-cosine 임베딩 (민감 구간 [0,1]).
        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        action_emb = self.action_in_proj(noisy_actions)  # [B, H, ax_width]

        if self.pi05:
            # timestep 은 토큰이 아니라 adaRMS 조건으로 주입.
            te = F.silu(self.time_mlp_in(time_emb))
            te = F.silu(self.time_mlp_out(te))
            action_time_tokens = action_emb
            adarms_cond = te
        else:
            # (action, time) 을 concat 후 MLP 로 융합 (adaRMS 없음).
            time_tok = time_emb[:, None, :].expand_as(action_emb)
            x = torch.cat([action_emb, time_tok], dim=-1)
            x = F.silu(self.action_time_mlp_in(x))
            action_time_tokens = self.action_time_mlp_out(x)
            adarms_cond = None

        tokens.append(action_time_tokens)
        pad_masks.append(torch.ones(B, action_time_tokens.shape[1], dtype=torch.bool, device=device))
        # action chunk 의 첫 토큰만 블록 경계(=이전 것들은 action 을 못 봄), 나머지는 서로 full.
        ar_mask += [1] + [0] * (self.action_horizon - 1)

        tokens = torch.cat(tokens, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        ar_mask = torch.tensor(ar_mask, dtype=torch.bool, device=device)
        return tokens, pad_masks, ar_mask, adarms_cond

    # ---------------------------------------------------------------------------------
    # 학습: flow matching loss
    # ---------------------------------------------------------------------------------

    def compute_loss(self, obs: Observation, actions: Tensor, noise=None, time=None) -> Tensor:
        """flow matching MSE loss. 반환: [B, action_horizon, action_dim] (reduction 없음)

        x_t = t*noise + (1-t)*actions ,  u_t = noise - actions ,  v_t = model(...)
        loss = ||v_t - u_t||^2
        """
        obs = preprocess_observation(obs)
        B = actions.shape[0]
        device = actions.device

        if noise is None:
            noise = torch.randn_like(actions)
        if time is None:
            # Beta(1.5,1) 분포에서 timestep 샘플 (작은 t 에 더 집중). [0.001, 1.0]
            time = torch.distributions.Beta(1.5, 1.0).sample((B,)).to(device) * 0.999 + 0.001

        t = time[:, None, None]
        x_t = t * noise + (1.0 - t) * actions
        u_t = noise - actions

        # prefix + suffix 를 한 번에 forward.
        prefix_tok, prefix_mask, prefix_ar = self.embed_prefix(obs)
        suffix_tok, suffix_mask, suffix_ar, adarms_cond = self.embed_suffix(obs.state, x_t, time)

        pad_mask = torch.cat([prefix_mask, suffix_mask], dim=1)
        ar_mask = torch.cat([prefix_ar, suffix_ar], dim=0)
        attn_mask = make_attn_mask(pad_mask, ar_mask)
        positions = torch.cumsum(pad_mask.int(), dim=1) - 1

        outputs, _ = self.llm(
            [prefix_tok, suffix_tok], positions, attn_mask, adarms_cond=[None, adarms_cond]
        )
        suffix_out = outputs[1]  # action expert 출력
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])
        return F.mse_loss(v_t, u_t, reduction="none")

    # ---------------------------------------------------------------------------------
    # 추론: KV cache + Euler 적분으로 action 생성
    # ---------------------------------------------------------------------------------

    @torch.no_grad()
    def sample_actions(self, obs: Observation, num_steps: int = 10, noise=None) -> Tensor:
        """noise 에서 시작해 flow 를 따라 적분하여 action chunk 생성. 반환: [B, H, action_dim]"""
        obs = preprocess_observation(obs)
        B = obs.state.shape[0]
        device = obs.state.device
        if noise is None:
            noise = torch.randn(B, self.action_horizon, self.action_dim, device=device)

        # 1) prefix(이미지+언어)를 한 번 forward 하여 KV cache 채움.
        prefix_tok, prefix_mask, prefix_ar = self.embed_prefix(obs)
        prefix_attn = make_attn_mask(prefix_mask, prefix_ar)
        prefix_pos = torch.cumsum(prefix_mask.int(), dim=1) - 1
        _, kv_cache = self.llm([prefix_tok, None], prefix_pos, prefix_attn)

        # 2) Euler 적분: t=1 (noise) -> t=0 (action).
        dt = -1.0 / num_steps
        x_t = noise
        t = 1.0
        while t >= -dt / 2:  # floating point 안전 마진
            time = torch.full((B,), t, dtype=torch.float32, device=device)
            v_t = self._denoise_step(obs.state, prefix_mask, kv_cache, x_t, time)
            x_t = x_t + dt * v_t
            t += dt
        return x_t

    def _denoise_step(self, state, prefix_mask, kv_cache, x_t, time) -> Tensor:
        """한 번의 denoising step. suffix 만 forward (prefix 는 cache 재사용)."""
        suffix_tok, suffix_mask, suffix_ar, adarms_cond = self.embed_suffix(state, x_t, time)
        B = prefix_mask.shape[0]
        prefix_len = prefix_mask.shape[1]
        suffix_len = suffix_mask.shape[1]

        # suffix query 가 prefix(전체) + suffix(블록) 를 보는 mask.
        prefix_2d = prefix_mask[:, None, :].expand(B, suffix_len, prefix_len)
        suffix_2d = make_attn_mask(suffix_mask, suffix_ar)
        full_mask = torch.cat([prefix_2d, suffix_2d], dim=2)  # [B, suffix_len, prefix+suffix]

        # suffix 토큰의 절대 위치 = prefix 길이 + suffix 내 누적.
        prefix_offset = prefix_mask.int().sum(dim=-1)[:, None]
        positions = prefix_offset + torch.cumsum(suffix_mask.int(), dim=1) - 1

        outputs, _ = self.llm(
            [None, suffix_tok], positions, full_mask, adarms_cond=[None, adarms_cond], kv_cache=kv_cache
        )
        suffix_out = outputs[1][:, -self.action_horizon :]
        return self.action_out_proj(suffix_out)


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
