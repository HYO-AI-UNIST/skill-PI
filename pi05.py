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

from VisionEncoder import SigLIP
from Tokenizer import Pi05Tokenizer, Pi05Embedder
from utils import posemb_sincos, make_attn_mask

# For ease debug, define DEBUG variable as global (0: no debug, 1~: debug print level)
DEBUG = 1

class PI05(nn.Module) :
    def __init__(self, config):
        super().__init__()
        if DEBUG >= 1: self.summary_cfg(config)

        # PI05 model setup
        model_config = config["PI05"]
        vlm_config = config["VLM"]
        siglip_config = config["SigLIP"]
        action_config = config["ActionExpert"]

        self.vlm_width = int(vlm_config["width"])
        self.action_dim = int(action_config["action_dim"])
        self.action_hz = int(action_config["action_horizontal"])
        self.action_width = int(action_config["width"])

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

        # VLM Setup (Tokenizer Involved)
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

    def embed_prefix(self, observation):
        """
        Core Variables:
         - tokens : suffix tokens
         - pad_masks: padding mask related to input settings
         - ar_mask: auto-regressinve mask to make boarder for block attention

        Input conf.

        Output conf.

        """
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
            tokenized_prompt, prompt_pad_mask = self.tokenizer(observation["task"], observation["state"])
            tokens.append(self.embedder(tokenized_prompt) * math.sqrt(self.vlm_width))
            pad_masks.append(prompt_pad_mask)
            ar_mask += [0] * tokenized_prompt.shape[1]
        task_state_length = tokens[-1].shape[1]

        # Concat for "list -> torch.Tensor" converstion
        tokens = torch.cat(tokens, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        ar_mask = torch.tensor(ar_mask, dtype=torch.bool)

        if DEBUG >= 1 : 
            print(f"[INFO] Final Prefix encoding result: ")
            print(f"    - tokens(images {img_tokens.shape[1]} x {len(observation["images"])}, task & state {task_state_length}) : {tokens.shape}")
            if DEBUG >= 2 : print(f"       - specification : {tokens}") 
            print(f"    - masking pad for indicating true tokens : {pad_masks.shape}")
            if DEBUG >= 2 : print(f"       - specification : {pad_masks}") 
            print(f"    - auto-regressive orthogonal mask pad : {ar_mask.shape}")
            if DEBUG >= 2 : print(f"       - specification : {ar_mask}") 
        return tokens, pad_masks, ar_mask
    
    def embed_suffix(self, actions, timestep) :
        """
        Core Variables:
         - tokens : suffix tokens
         - pad_masks: padding mask related to input settings
         - ar_mask: auto-regressinve mask to make boarder for block attention

        Input conf.

        Output conf.

        """
        tokens, pad_masks, ar_mask = [], [], []

        # Time Embedding -> Condition Vector for RMSNorm
        time_emb = posemb_sincos(timestep, self.action_width, min_period=4e-3, max_period=4.0)
        adarms_cond_emb = F.silu(self.time_in(time_emb))
        adarms_cond_emb = F.silu(self.time_out(adarms_cond_emb))
        if DEBUG >= 2 : 
            print(f"Time Embedding: {time_emb}")
            print(f"RMS time conditional vector : {adarms_cond_emb}")

        # Action Encoding
        tokens.append(self.action_in(actions))
        pad_masks.append(torch.ones(actions.shape[0], self.action_hz, dtype=torch.bool))
        ar_mask += [1] + [0] * (self.action_hz - 1)
        
        # Concat for "list -> torch.Tensor" converstion
        tokens = torch.cat(tokens, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        ar_mask = torch.tensor(ar_mask, dtype=torch.bool)

        if DEBUG >= 1 : 
            print(f"[INFO] Final Suffix encoding result: ")
            print(f"    - tokens(actions {self.action_hz}) : {tokens.shape}")
            if DEBUG >= 2 : print(f"       - specification : {tokens}") 
            print(f"    - masking pad for indicating true tokens : {pad_masks.shape}")
            if DEBUG >= 2 : print(f"       - specification : {pad_masks}") 
            print(f"    - auto-regressive orthogonal mask pad : {ar_mask.shape}")
            if DEBUG >= 2 : print(f"       - specification : {ar_mask}")
        return tokens, pad_masks, ar_mask, adarms_cond_emb

    def forward(self, prefix_tokens, suffix_tokens, attn_mask, positions, adarms_cond_emb):
        # Initial setup
        B = observation["state"].shape[0]
        device = observation["state"].device



        return
    
    def pi05_train(self, observation, actions, only_flows=False) :
        # Initial setup
        B = observation["state"].shape[0]
        device = observation["state"].device
        assert actions != None, "Training requires ground true actions"
        observation = observation_preprocess(observation)

        # Training sample augmentation
        noise = torch.randn_like(actions)
        time = torch.distributions.Beta(1.5, 1.0).sample((B,)).to(device) * 0.999 + 0.001

        t = time[:, None, None] # Broadcasted current timestep
        x_t = noise * t + (1.0 - t) * actions # Noise mixed sample
        u_t = noise - actions # True flow for x_t

        # Stage 1. Token Embeddings
        prefix_tok, prefix_pad, prefix_ar = self.embed_prefix(observation)
        suffix_tok, suffix_pad, suffix_ar, adarms_cond_emb = self.embed_suffix(x_t, time)

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
        prefix_out, suffix_out = self(prefix_tok, suffix_tok, attn_mask, positions, adarms_cond_emb)
        v_t = self.action_out(suffix_out)

        if only_flows :
            return v_t, u_t
        else :
            return F.mse_loss(v_t, u_t, reduction="none")
    
    def pi05_inference(self, observation, actions=None) :
        # Initial setup
        B = observation["state"].shape[0]
        device = observation["state"].device
        if actions != None :
            print("[WARN] Given actions will be used as initial noise")
            if DEBUG >= 2 : print("Actions: ", actions)

        return

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
    state_lowerbound, state_upperbound = -1.0, 1.0
    observation = {
        "images": [torch.randn(1, 3, 224, 224), torch.randn(1, 3, 224, 224), torch.randn(1, 3, 224, 224)],
        "images_mask": [torch.ones(1, dtype=torch.bool), torch.ones(1, dtype=torch.bool), torch.ones(1, dtype=torch.bool)],
        "task": ["Put the dish on the table"],
        "state": torch.empty(1, 32).uniform_(state_lowerbound, state_upperbound)
    }
    observation_preprocessed = {
        "images": [torch.randn(1, 3, 224, 224), torch.randn(1, 3, 224, 224), torch.randn(1, 3, 224, 224)],
        "images_mask": [torch.ones(1, dtype=torch.bool), torch.ones(1, dtype=torch.bool), torch.ones(1, dtype=torch.bool)],
        "tokens": torch.randint(1, 257152, (1, 200)),
        "tokens_mask": torch.ones(1, 200, dtype=torch.bool)
    }
    true_actions = torch.empty(1, 50, 32).uniform_(state_lowerbound, state_upperbound)

    # Model setup
    config = configparser.ConfigParser()
    config.read("./config.cfg")
    model = PI05(config)

    # Model basic test
    # model(observation, true_actions)

    # Model training test
    model.pi05_train(observation, true_actions)

    # Model inference test
