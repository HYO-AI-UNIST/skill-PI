from __future__ import annotations

import dataclasses
import hashlib
import math
from typing import Optional
import configparser

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

import sentencepiece
from safetensors import safe_open

class Pi05Embedder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, backbone_llm_path = None, freeze = 1) :
        super().__init__()

        # Define default embedding table (Randomly Initialized)
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # If backbone llm is given, change it's weight with pretrained one
        if backbone_llm_path :
            print(f"[INFO] Embedding table in pretrained model({backbone_llm_path}) will be used")
            self.load_pretrained(backbone_llm_path)
        self.embedding.weight.requires_grad_(not freeze)

    def forward(self, tokens) :
        return self.embedding(tokens)
    
    def load_pretrained(self, path):
        if path.endswith(".npz"):
            data = np.load(path)
            key = "params/llm/embedder/input_embedding"
            if key not in data:
                key = "llm/embedder/input_embedding"
            weight = torch.from_numpy(data[key])
        elif path.endswith(".safetensors"):
            key = "paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"
            fallback_key = "embedder.weight"
            with safe_open(path, framework="pt", device="cpu") as f:
                weight = f.get_tensor(key if key in f.keys() else fallback_key)
        else:
            state = torch.load(path, map_location="cpu")
            key = "paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"
            weight = state[key] if key in state else state["embedder.weight"]

        if tuple(weight.shape) != tuple(self.embedding.weight.shape):
            raise ValueError(
                f"Embedding shape mismatch: checkpoint {tuple(weight.shape)} "
                f"vs model {tuple(self.embedding.weight.shape)}"
            )

        with torch.no_grad():
            self.embedding.weight.copy_(weight.to(self.embedding.weight.dtype))


class Pi05Tokenizer(nn.Module):
    def __init__(self, prompt_length, sp_model_path, state_low = -1.0, state_up = 1.0, bin_size = 256) :
        super().__init__()
        self.max_len = prompt_length
        self.discrete_bins = np.linspace(state_low, state_up, bin_size + 1)[:-1]

        # Loading PaliGemma Tokenizer
        sp = sentencepiece.SentencePieceProcessor(model_file=sp_model_path)
        self._encode = lambda s, add_bos=False: sp.encode(s, add_bos=add_bos)
    
    def forward(self, raw_task : list[str], raw_state: torch.Tensor) :
        tokenized_prompt, prompt_pad_mask = [], []

        for i in range(raw_state.shape[0]) :
            # Continuous value to discritize value state conversion
            state = raw_state[i].detach().cpu().numpy()
            discretized_state = np.digitize(state, self.discrete_bins) - 1
            state_str = " ".join(map(str, discretized_state.tolist()))

            # polish task prompt and generate final prompt
            task_str = raw_task[i].strip().replace("_", " ").replace("\n", " ")
            full_prompt = f"Task: {task_str}, State: {state_str};\nAction: "
            tokens = self._encode(full_prompt, add_bos=True)

            # Build mask for prompt based on max_len
            assert len(tokens) <= self.max_len, f"{i} sample with prompt '{full_prompt}' overs the max length of input ({self.max_len})"
            if len(tokens) < self.max_len :
                mask = [True] * len(tokens) + [False] * (self.max_len - len(tokens))
                tokens += [0] * (self.max_len - len(tokens))
            else :
                mask = [True] * self.max_len

            # Add current mask, tokens to the final output
            tokenized_prompt.append(np.asarray(tokens, dtype=np.int64))
            prompt_pad_mask.append(np.asarray(mask, dtype=bool))
        
        # Convert Numpy to Pytorch tensor and return
        # As you notice, this process becomes more efficient if it is done before model training/inferrence
        # For practical purpose, please preprocess the dataset to use it as a tokenized one
        return torch.as_tensor(np.stack(tokenized_prompt), device=raw_state.device), torch.as_tensor(np.stack(prompt_pad_mask), device=raw_state.device)