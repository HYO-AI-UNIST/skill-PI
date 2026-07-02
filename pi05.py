"""
This is PI05 implementation with core logics and functions from scratch.
Written by Hyogi Kim.

The architecture difference from pi0 to pi05
 1. RMSNorm: Time-embedding used in each normalization with gated residual (Verified in DiT models)
    - Inject projected time-embedding vector to each RMSNorm layer
    - Caculate scale/shift/gate value everytime when forward passing happens.
    - Use gate as a residual output controller

 2. State as Prefix: Robot proprioceptions are used in prefix with 256-bin discretized value (Readable)
    - Convert original float state[-0.1, 1.2, 0.7, ..., 6.8] into discretized state[0, 10, 7, ..., 59]
    - Injected into the task prompt: "Task: Put the cup on the table, State: 0 10 7 ... 59"
    - Use Gemma Tokenizer to make VLM to read the state as a specific value

PI05 Config list

"""

from __future__ import annotations

import dataclasses
import math
from typing import Optional
import configparser

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from module import *
from VisionEncoder import SigLIP
from Tokenizer import Pi05Tokenizer, Pi05Embedder
from utils import posemb_sincos, make_attn_mask, apply_rope, gated_residual, optional_cat

# For ease debug, define DEBUG variable as global (0: no debug, 1~: debug print level)
DEBUG = 1
BIG_NEG = -2.3819763e38

class PI05(nn.Module) :
    def __init__(self, config):
        super().__init__()
        if DEBUG >= 1: self.summary_cfg(config)

        # PI05 model setup
        model_config = config["PI05"]
        vlm_config = config["VLM"]
        siglip_config = config["SigLIP"]
        action_config = config["ActionExpert"]

        self.depth = int(model_config["depth"])
        self.vlm_width = int(vlm_config["width"])
        self.action_dim = int(action_config["action_dim"])
        self.action_hz = int(action_config["action_horizontal"])
        self.action_width = int(action_config["width"])

        self.head_dim = int(model_config["head_dim"])
        self.num_heads = int(model_config["num_heads"])
        self.num_kv_heads = int(model_config["num_kv_heads"])

        # Vision Encoder (SigLIP) setup
        self.image_width, self.image_height = int(model_config["image_width"]), int(model_config["image_height"])
        self.patch_width, self.patch_height = int(siglip_config["patch_width"]), int(siglip_config["patch_height"])
        self.vision = SigLIP(
            (self.image_height, self.image_width),
            (self.patch_height, self.patch_width),
            int(siglip_config["width"]),
            int(siglip_config["depth"]),
            int(siglip_config["num_heads"]),
            int(siglip_config["mlp_dim"]),
            self.vlm_width
        )

        # Action Expert Setup
        self.action_in = nn.Linear(self.action_dim, self.action_width)
        self.action_out = nn.Linear(self.action_width, self.action_dim)
        self.time_in = nn.Linear(self.action_width, self.action_width)
        self.time_out = nn.Linear(self.action_width, self.action_width)

        # Action Expert Transformer Setup
        self.action_attn_rms = nn.ModuleList(
            [RMSNorm(self.action_width, self.action_width) for _ in range(self.depth)]
        )
        self.qkv_action = nn.ModuleDict({
            "q" : nn.ModuleList([Linear(self.action_width, self.head_dim * self.num_heads, bias=False) for _ in range(self.depth)]),
            "k" : nn.ModuleList([Linear(self.action_width, self.head_dim * self.num_kv_heads, bias=False) for _ in range(self.depth)]),
            "v" : nn.ModuleList([Linear(self.action_width, self.head_dim * self.num_kv_heads, bias=False) for _ in range(self.depth)])
        })
        self.out_action = nn.ModuleList(
            [Linear(self.head_dim * self.num_heads, self.action_width, bias=False) for _ in range(self.depth)]
        )
        self.action_mlp_rms = nn.ModuleList(
            [RMSNorm(self.action_width, self.action_width) for _ in range(self.depth)]
        )
        self.action_mlp = nn.ModuleList(
            [GemmaMLP(self.action_width, int(action_config["mlp_dim"])) for _ in range(self.depth)]
        )
        self.action_final_rms = RMSNorm(self.action_width, self.action_width)

        # VLM Setup (Tokenizer Involved)
        self.subtask_length = int(vlm_config["max_subtask_len"])
        self.embedder = Pi05Embedder(
            int(vlm_config["vocab_size"]), 
            self.vlm_width,
            str(vlm_config["backbone_model"]),
            int(vlm_config["freeze"])
        )
        self.already_tokenized = int(vlm_config["tokenized"])
        if not self.already_tokenized :
            self.tokenizer = Pi05Tokenizer(
                int(vlm_config["max_token_len"]),
                str(vlm_config["tokenizer_model"]),
                float(action_config["state_lowerbound"]), 
                float(action_config["state_upperbound"]),
                int(vlm_config["action_bin_size"])
            )
        self.lm_head = nn.Linear(self.vlm_width, int(vlm_config["vocab_size"]), bias=False)
        
        # VLM Transformer Setup
        self.vlm_attn_rms = nn.ModuleList(
            [RMSNorm(self.vlm_width) for _ in range(self.depth)]
        )
        self.qkv_vlm = nn.ModuleDict({
            "q" : nn.ModuleList([Linear(self.vlm_width, self.head_dim * self.num_heads, bias=False) for _ in range(self.depth)]),
            "k" : nn.ModuleList([Linear(self.vlm_width, self.head_dim * self.num_kv_heads, bias=False) for _ in range(self.depth)]),
            "v" : nn.ModuleList([Linear(self.vlm_width, self.head_dim * self.num_kv_heads, bias=False) for _ in range(self.depth)])
        })
        self.out_vlm = nn.ModuleList(
            [Linear(self.head_dim * self.num_heads, self.vlm_width, bias=False) for _ in range(self.depth)]
        )
        self.vlm_mlp_rms = nn.ModuleList(
            [RMSNorm(self.vlm_width) for _ in range(self.depth)]
        )
        self.vlm_mlp = nn.ModuleList(
            [GemmaMLP(self.vlm_width, int(vlm_config["mlp_dim"])) for _ in range(self.depth)]
        )
        self.vlm_final_rms = RMSNorm(self.vlm_width)

    def attention(self, query, key, value, mask, positions, kv_cache=None) :
        # Reshape tensor (B, T, head_dim * [kv]num_heads) -> (B, T, [kv]num_heads, head_dim)
        B, T, _ = query.shape
        q = query.view(B, T, self.num_heads, self.head_dim)
        k = key.view(B, T, self.num_kv_heads, self.head_dim)
        v = value.view(B, T, self.num_kv_heads, self.head_dim)

        # Apply RoPE (Rotary Positional Encoding for each token with position idx)
        q = apply_rope(q, positions) * (self.head_dim**-0.5)
        k = apply_rope(k, positions)

        # Extend given KV cache if it exist (Use both VLM & ActionExpert - Inference step)
        if kv_cache is not None:
            k = torch.cat([kv_cache[0], k], dim=1)
            v = torch.cat([kv_cache[1], v], dim=1)
        new_kv_cache = (k, v)

        # Repeat KV for GQA(Grouped Query Attention) to match num_kv_heads and num_head
        if self.num_kv_heads != self.num_heads:
            repeat_size = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(repeat_size, dim=2)
            v = v.repeat_interleave(repeat_size, dim=2)
        
        # Query, Key attention logits (float32 stabilization). b=batch, t=query, s=key, h=num_heads, d=head_dim
        logits = torch.einsum("bthd,bshd->bhts", q.float(), k.float())

        # Applying mask to attnetion logits
        mask = mask[:, None, :, :] # Expand mask to broadcast each heads
        logits = torch.where(mask, logits, torch.full_like(logits, BIG_NEG))
        probs = torch.softmax(logits, dim=-1).to(v.dtype)

        # Value matmul with attention logits to finalize calculation
        encoded = torch.einsum("bhts,bshd->bthd", probs, v)
        encoded = encoded.reshape(B, T, self.num_heads * self.head_dim)

        return encoded, new_kv_cache
    
    def forward(self, prefix_tokens, suffix_tokens, attn_mask, positions, adarms_cond_emb=None, kv_caches=None):
        # Initial setup
        new_kv_caches = []
        if not kv_caches :
            kv_caches = [kv_caches] * self.depth

        if DEBUG >= 1 : print("[INFO] Transformer forward with given Prefix/Suffix tokens")
        for i in range(self.depth) :
            # Step 1. Adaptive RMS normalization with conditional vector (pre attention)
            if DEBUG >= 2 : print("=" * 30 + f" {i + 1}-th Step " + "=" * 30)
            mod_prefix_tokens, gate_p = self.vlm_attn_rms[i](prefix_tokens)
            mod_suffix_tokens, gate_s = self.action_attn_rms[i](suffix_tokens, adarms_cond_emb)
            if DEBUG >= 2 : print(f" - Prefix/Suffix Tokens shape after RMSNorm (Pre Attention) : {mod_prefix_tokens.shape if mod_prefix_tokens is not None else None} / {mod_suffix_tokens.shape if mod_suffix_tokens is not None else None}")

            # Step 2. Shared Attention between VLM & ActionExpert
            q = optional_cat(self.qkv_vlm["q"][i](mod_prefix_tokens), self.qkv_action["q"][i](mod_suffix_tokens), dim=1)
            k = optional_cat(self.qkv_vlm["k"][i](mod_prefix_tokens), self.qkv_action["k"][i](mod_suffix_tokens), dim=1)
            v = optional_cat(self.qkv_vlm["v"][i](mod_prefix_tokens), self.qkv_action["v"][i](mod_suffix_tokens), dim=1)
            if DEBUG >= 2 : 
                print(f" - Query, Key, Value shape before attention : Q({q.shape}), K({k.shape}), V({v.shape})")
                if DEBUG >= 3 : 
                    print(f"    - Query True value : {q}")
                    print(f"    - Key True value : {k}")
                    print(f"    - Value True value : {v}")
            attn_out, new_kv_cache = self.attention(q, k ,v, attn_mask, positions, kv_caches[i])
            new_kv_caches.append(new_kv_cache)
            bifurcation = mod_prefix_tokens.shape[1] if mod_prefix_tokens is not None else 0
            mod_prefix_tokens = self.out_vlm[i](attn_out[:, :bifurcation] if mod_prefix_tokens is not None else None)
            mod_suffix_tokens = self.out_action[i](attn_out[:, bifurcation : bifurcation + mod_suffix_tokens.shape[1]] if mod_suffix_tokens is not None else None)
            if DEBUG >= 2 : 
                print(f" - Prefix/Suffix Tokens shape after Attention : {mod_prefix_tokens.shape if mod_prefix_tokens is not None else None} / {mod_suffix_tokens.shape if mod_suffix_tokens is not None else None}")
                print(f" - New KV Cache shape from Attention : Cache K({new_kv_cache[0].shape}), Cache V({new_kv_cache[1].shape})")

            # Step 3. Gated Residual for each token
            prefix_tokens = gated_residual(prefix_tokens, mod_prefix_tokens, gate_p)
            suffix_tokens = gated_residual(suffix_tokens, mod_suffix_tokens, gate_s)
            if DEBUG >= 2 : print(f" - Prefix/Suffix Tokens shape after Residual : {prefix_tokens.shape if prefix_tokens is not None else None} / {suffix_tokens.shape if suffix_tokens is not None else None}")

            # Step 4. Adaptive RMS normalization with conditional vector (pre MLP)
            mod_prefix_tokens, gate_p = self.vlm_mlp_rms[i](prefix_tokens)
            mod_suffix_tokens, gate_s = self.action_mlp_rms[i](suffix_tokens, adarms_cond_emb)
            if DEBUG >= 2 : print(f" - Prefix/Suffix Tokens shape after RMSNorm (Pre MLP) : {mod_prefix_tokens.shape if mod_prefix_tokens is not None else None} / {mod_suffix_tokens.shape if mod_suffix_tokens is not None else None}")

            # Step 5. Private MLP for each expert (VLM / ActionExpert)
            mod_prefix_tokens = self.vlm_mlp[i](mod_prefix_tokens)
            mod_suffix_tokens = self.action_mlp[i](mod_suffix_tokens)
            if DEBUG >= 2 : print(f" - Prefix/Suffix Tokens shape after MLP : {mod_prefix_tokens.shape if mod_prefix_tokens is not None else None} / {mod_suffix_tokens.shape if mod_suffix_tokens is not None else None}")

            # Step6. Gated Residual for each token
            prefix_tokens = gated_residual(prefix_tokens, mod_prefix_tokens, gate_p)
            suffix_tokens = gated_residual(suffix_tokens, mod_suffix_tokens, gate_s)
            if DEBUG >= 2 : print(f" - Prefix/Suffix Tokens shape after Residual : {prefix_tokens.shape if prefix_tokens is not None else None} / {suffix_tokens.shape if suffix_tokens is not None else None}")
        if DEBUG >= 2 : print("=" * 72)

        # Final RMS Normalization layer and cache output
        prefix_tokens = self.vlm_final_rms(prefix_tokens)[0]
        suffix_tokens = self.action_final_rms(suffix_tokens, adarms_cond_emb)[0]
        if DEBUG >= 1:
            print(f"[INFO] Transformer Finished, result summary: ")
            print(f"  - Final prefix token shape : {prefix_tokens.shape if prefix_tokens is not None else None}")
            print(f"  - Final suffix token shape : {suffix_tokens.shape if suffix_tokens is not None else None}")
            print(f"  - Newly generated KV Cache shape : {len(new_kv_caches)} x [{new_kv_caches[0][0].shape}, {new_kv_caches[0][1].shape}] ")
            print(f"  - Conditional Embedding shape : {adarms_cond_emb.shape if adarms_cond_emb is not None else None}")

        return prefix_tokens, suffix_tokens, new_kv_caches

    def embed_prefix(self, observation, device):
        tokens, pad_masks, ar_mask = [], [], []

        # Image Encoding
        if DEBUG >= 2 : print(f"Input Images : {observation["images"]}")
        for image, img_pad_mask in zip(observation["images"], observation["images_mask"]) :
            img_tokens = self.vision(image)
            tokens.append(img_tokens)
            pad_masks.append(img_pad_mask[:, None].expand(img_tokens.shape[:2]))  # img_tokens.shape[:2] = (Batch, #Patches = #Tokens)
            ar_mask += [0] * img_tokens.shape[1]

        # Task, State Encoding
        if self.already_tokenized :
            assert (observation["tokens"] != None) and (observation["tokens_mask"] != None), \
                "There is no tokens(=prompts) or related mask!, something wrong happens..."
            if DEBUG >= 2 : 
                print(f"Input Prompt : '{observation["tokens"]}'")
                print(f"Input Mask : {observation["tokens_mask"]}")
            tokens.append(self.embedder(observation["tokens"]) * math.sqrt(self.vlm_width))
            pad_masks.append(observation["tokens_mask"])
            ar_mask += [0] * tokens[-1].shape[1]
        else :
            assert (observation["task"] != None) and (observation["state"] != None), \
                "There is no task or states!, something wrong happens..."
            if DEBUG >= 2 : 
                print(f"Input Task : '{observation["task"]}'")
                print(f"Input State : {observation["state"]}")
            tokenized_prompt, prompt_pad_mask = self.tokenizer.encode(observation["task"], observation["state"])
            tokens.append(self.embedder(tokenized_prompt) * math.sqrt(self.vlm_width))
            pad_masks.append(prompt_pad_mask)
            ar_mask += [0] * tokenized_prompt.shape[1]
        task_state_length = tokens[-1].shape[1]

        # Concat for "list -> torch.Tensor" converstion
        tokens = torch.cat(tokens, dim=1).to(device)
        pad_masks = torch.cat(pad_masks, dim=1).to(device)
        ar_mask = torch.tensor(ar_mask, dtype=torch.bool).to(device)

        if DEBUG >= 1 : 
            print(f"[INFO] Final Prefix encoding result: ")
            print(f"    - tokens(images {img_tokens.shape[1]} x {len(observation["images"])}, task & state {task_state_length}) : {tokens.shape}")
            if DEBUG >= 2 : print(f"       - specification : {tokens}") 
            print(f"    - masking pad for indicating true tokens : {pad_masks.shape}")
            if DEBUG >= 2 : print(f"       - specification : {pad_masks}") 
            print(f"    - auto-regressive orthogonal mask pad : {ar_mask.shape}")
            if DEBUG >= 2 : print(f"       - specification : {ar_mask}") 
        return tokens, pad_masks, ar_mask
    
    def embed_suffix(self, actions, timestep, device) :
        tokens, pad_masks, ar_mask = [], [], []

        # Time Embedding -> Condition Vector for RMSNorm
        time_emb = posemb_sincos(timestep, self.action_width, min_period=4e-3, max_period=4.0)
        adarms_cond_emb = F.silu(self.time_in(time_emb))
        adarms_cond_emb = F.silu(self.time_out(adarms_cond_emb))
        if DEBUG >= 2 : 
            print(f"Time Embedding: {time_emb}")

        # Action Encoding
        tokens.append(self.action_in(actions))
        pad_masks.append(torch.ones(actions.shape[0], self.action_hz, dtype=torch.bool))
        ar_mask += [1] + [0] * (self.action_hz - 1)
        
        # Concat for "list -> torch.Tensor" converstion
        tokens = torch.cat(tokens, dim=1).to(device)
        pad_masks = torch.cat(pad_masks, dim=1).to(device)
        ar_mask = torch.tensor(ar_mask, dtype=torch.bool).to(device)

        if DEBUG >= 1 : 
            print(f"[INFO] Final Suffix encoding result: ")
            print(f"    - tokens(actions {self.action_hz}) : {tokens.shape}")
            if DEBUG >= 2 : print(f"       - specification : {tokens}") 
            print(f"    - masking pad for indicating true tokens : {pad_masks.shape}")
            if DEBUG >= 2 : print(f"       - specification : {pad_masks}") 
            print(f"    - auto-regressive orthogonal mask pad : {ar_mask.shape}")
            if DEBUG >= 2 : print(f"       - specification : {ar_mask}")
            print(f"    - Adaptive RMS Conditional embedding vector : {adarms_cond_emb.shape}")
            if DEBUG >= 2 : print(f"       - specification : {adarms_cond_emb}")
        return tokens, pad_masks, ar_mask, adarms_cond_emb

    def subtask_generation(self, observation) :
        # Stage 1. Observation Token Embeddings (Prefix only)
        prefix_tok, prefix_pad, prefix_ar = self.embed_prefix(observation, device)

        # Stage 2. Subtask auto-regressive generation for each token
        generated_ids = []
        kv_caches = None

        for step in range(self.subtask_length):
            if step == 0:
                tokens = prefix_tok
                pad = prefix_pad
                ar = prefix_ar
            else:
                token_ids = torch.tensor([[generated_ids[-1]]], device=device)
                token_emb = self.embedder(token_ids) * math.sqrt(self.vlm_width)

                tokens = token_emb
                pad = torch.ones(1, 1, dtype=torch.bool, device=device)
                ar = torch.ones(1, dtype=torch.bool, device=device)  # causal text step

            attn_mask = make_attn_mask(pad, ar)
            positions = torch.cumsum(pad.int(), dim=1) - 1

            prefix_out, _, kv_caches = self(tokens, None, attn_mask, positions, None, kv_caches)

            logits = model.lm_head(prefix_out[:, -1])
            next_id = int(torch.argmax(logits, dim=-1).item())

            if next_id in {1}:
                break
            generated_ids.append(next_id)

        return self.tokenizer.decode(generated_ids)
    
    def pi05_whole(self, observation, actions, only_flows=False) :
        # Initial setup
        B = observation["state"].shape[0]
        device = observation["state"].device
        assert actions != None, "Ture action is required for smoke test"
        observation = observation_preprocess(observation)

        # Training sample augmentation
        noise = torch.randn_like(actions)
        time = torch.distributions.Beta(1.5, 1.0).sample((B,)).to(device) * 0.999 + 0.001

        t = time[:, None, None] # Broadcasted current timestep
        x_t = noise * t + (1.0 - t) * actions # Noise mixed sample
        u_t = noise - actions # True flow for x_t

        # Stage 1. Token Embeddings
        prefix_tok, prefix_pad, prefix_ar = self.embed_prefix(observation, device)
        suffix_tok, suffix_pad, suffix_ar, adarms_cond_emb = self.embed_suffix(x_t, time, device)

        # Concatenate pad, auto-regressive mask and generate final Attention Mask
        pad_mask = torch.cat([prefix_pad, suffix_pad], dim=1)
        ar_mask = torch.cat([prefix_ar, suffix_ar], dim=0)
        attn_mask = make_attn_mask(pad_mask, ar_mask)
        positions = torch.cumsum(pad_mask.int(), dim=1) - 1 # token position for RoPE (Rotary Position Embedding) [0, 1, 2, ...]
        if DEBUG >= 1 : 
            print(f"[INFO] Final Attention Mask : {attn_mask.shape}")
            if DEBUG >= 2 : print(f"    - Specification : {attn_mask}")
            print(f"[INFO] Embedding positions for RoPE : {positions}")

        # Stage 2. Expert Transformer (Shared Attention)
        prefix_out, suffix_out, kv_caches = self(prefix_tok, suffix_tok, attn_mask, positions, adarms_cond_emb)
        v_t = self.action_out(suffix_out)

        if only_flows :
            return v_t, u_t
        else :
            return F.mse_loss(v_t, u_t, reduction="none")
    
    def pi05_split(self, observation, actions) :
        # Initial setup
        B = observation["state"].shape[0]
        device = observation["state"].device
        assert actions != None, "Ture action is required for smoke test"
        observation = observation_preprocess(observation)

         # Training sample augmentation
        noise = torch.randn_like(actions)
        time = torch.distributions.Beta(1.5, 1.0).sample((B,)).to(device) * 0.999 + 0.001

        t = time[:, None, None] # Broadcasted current timestep
        x_t = noise * t + (1.0 - t) * actions # Noise mixed sample
        u_t = noise - actions # True flow for x_t

         # Stage 1. Token Embeddings
        prefix_tok, prefix_pad, prefix_ar = self.embed_prefix(observation, device)
        suffix_tok, suffix_pad, suffix_ar, adarms_cond_emb = self.embed_suffix(x_t, time, device)

        # Generate each prefix, suffix tokens attention mask
        prefix_attn_mask = make_attn_mask(prefix_pad, prefix_ar)
        prefix_positions = torch.cumsum(prefix_pad.int(), dim=1) - 1

        suffix_attn_mask = make_attn_mask(suffix_pad, suffix_ar)
        prefix_extended_mask = prefix_pad[:, None, :].expand(B, suffix_pad.shape[1], prefix_pad.shape[1])
        suffix_full_mask = torch.cat([prefix_extended_mask, suffix_attn_mask], dim=2)
        suffix_positions = prefix_positions[:, -1:] + torch.cumsum(suffix_pad.int(), dim=1)

        # Stage 2. VLM Transformer (Get KV cache) and Action Expert Transformer (Denoising)
        prefix_out, _, kv_caches = self(prefix_tok, None, prefix_attn_mask, prefix_positions)
        _, suffix_out, _ = self(None, suffix_tok, suffix_full_mask, suffix_positions, adarms_cond_emb, kv_caches)

        #Summary Result
        if DEBUG >= 1 : 
            print(f"[INFO] prefix, suffix split transformer forward result")
            print(f"  - prefix output shape : {prefix_out.shape}")
            print(f"  - suffix output shape : {suffix_out.shape}")

        return prefix_out, suffix_out

    @staticmethod
    def summary_cfg(config):
        print("=" * 30 + " CONFIG " + "=" * 30)
        for category in config:
            print(f"Category '{category}'")
            for item in config[category]:
                print(f" - Item '{item}' : {config[category][item]}")
        print("=" * 68)


def observation_preprocess(observation) :
    return observation


if __name__ == "__main__":
    ### PI05 EXAMPLE TEST ###
    device = "cuda"
    state_lowerbound, state_upperbound = -1.0, 1.0
    observation = {
        "images": [torch.randn(1, 3, 224, 224).to(device), torch.randn(1, 3, 224, 224).to(device), torch.randn(1, 3, 224, 224).to(device)],
        "images_mask": [torch.ones(1, dtype=torch.bool).to(device), torch.ones(1, dtype=torch.bool).to(device), torch.ones(1, dtype=torch.bool).to(device)],
        "task": ["Put the dish on the table"],
        "state": torch.empty(1, 32).uniform_(state_lowerbound, state_upperbound).to(device)
    }
    observation_preprocessed = {
        "images": [torch.randn(1, 3, 224, 224).to(device), torch.randn(1, 3, 224, 224).to(device), torch.randn(1, 3, 224, 224).to(device)],
        "images_mask": [torch.ones(1, dtype=torch.bool).to(device), torch.ones(1, dtype=torch.bool).to(device), torch.ones(1, dtype=torch.bool).to(device)],
        "tokens": torch.randint(1, 257152, (1, 200)).to(device),
        "tokens_mask": torch.ones(1, 200, dtype=torch.bool).to(device)
    }
    true_actions = torch.empty(1, 50, 32).uniform_(state_lowerbound, state_upperbound).to(device)

    # Model setup
    config = configparser.ConfigParser()
    config.read("./config.cfg")
    model = PI05(config).to(device)

    # Model whole token test
    if DEBUG >= 1 : print("================== Whole Token Smoke Test ==================")
    model.pi05_whole(observation, true_actions)

    # Model split token test
    if DEBUG >= 1 : print("================== Split Token Smoke Test ==================")
    model.pi05_split(observation, true_actions)

    # Model Subtask generation test
    if DEBUG >= 1 : print("================== Subtask Generation Smoke Test ==================")
    subtask = model.subtask_generation(observation)
    if DEBUG >=1 : print(f"Generated Subtask : {subtask}")
    
