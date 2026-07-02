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
    def __init__(self, prompt_length, sp_model_path, state_low=-1.0, state_up=1.0, bin_size=256):
        super().__init__()
        self.max_len = prompt_length
        self.state_low = state_low
        self.state_up = state_up
        self.bin_size = bin_size
        self.discrete_bins = np.linspace(state_low, state_up, bin_size + 1)[:-1]

        # Keep the SentencePiece processor so generation can access ids/decode.
        self.sp = sentencepiece.SentencePieceProcessor(model_file=sp_model_path)
        self.pad_id = self._safe_special_id(self.sp.pad_id(), 0)
        self.eos_id = self._safe_special_id(self.sp.eos_id(), 1)
        self.bos_id = self._safe_special_id(self.sp.bos_id(), 2)
        self.unk_id = self._safe_special_id(self.sp.unk_id(), 3)

    @staticmethod
    def _safe_special_id(value: int, fallback: int) -> int:
        return fallback if value is None or value < 0 else int(value)

    def encode_text(self, text: str, *, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        return list(self.sp.encode(text, add_bos=add_bos, add_eos=add_eos))

    def decode(self, generated_tokens: list[int] | torch.Tensor, *, skip_special_tokens: bool = True) -> str:
        if isinstance(generated_tokens, torch.Tensor):
            token_ids = generated_tokens.detach().cpu().view(-1).tolist()
        else:
            token_ids = [int(token) for token in generated_tokens]

        if skip_special_tokens:
            special_ids = {self.pad_id, self.bos_id, self.eos_id}
            token_ids = [token for token in token_ids if token not in special_ids]
        return self.sp.decode(token_ids)

    def _state_to_text(self, state: np.ndarray) -> str:
        state = np.clip(state, self.state_low, self.state_up)
        discretized_state = np.digitize(state, self.discrete_bins) - 1
        discretized_state = np.clip(discretized_state, 0, self.bin_size - 1)
        return " ".join(map(str, discretized_state.tolist()))

    @staticmethod
    def _clean_task(task: str) -> str:
        return task.strip().replace("_", " ").replace("\n", " ")

    def build_prompt(self, raw_task: str, raw_state: torch.Tensor | np.ndarray, *, response_key: str = "Action") -> str:
        if isinstance(raw_state, torch.Tensor):
            state = raw_state.detach().cpu().numpy()
        else:
            state = np.asarray(raw_state)
        task_str = self._clean_task(raw_task)
        state_str = self._state_to_text(state)
        return f"Task: {task_str}, State: {state_str};\n{response_key}: "

    def encode(
        self,
        raw_task: list[str] | str,
        raw_state: torch.Tensor | None = None,
        *,
        response_key: str = "Action",
        add_bos: bool = True,
        add_eos: bool = False,
        max_len: int | None = None,
        pad: bool = True,
        device: torch.device | str | None = None,
    ):
        """Encode either PI05 prompts or plain text.

        If raw_state is provided, raw_task is interpreted as task text and a full
        PI05 prompt is built: "Task: ..., State: ...;\n<Response>: ". This is the
        path used by PI05.embed_prefix.

        If raw_state is None, raw_task is encoded as plain text. This is useful
        for subtask targets, usually with add_eos=True.
        """
        if raw_state is None:
            text = raw_task if isinstance(raw_task, str) else raw_task[0]
            return self.encode_text(text, add_bos=add_bos, add_eos=add_eos)

        tasks = [raw_task] if isinstance(raw_task, str) else list(raw_task)
        length = self.max_len if max_len is None else max_len
        out_device = raw_state.device if device is None and isinstance(raw_state, torch.Tensor) else device

        tokenized_prompt, prompt_pad_mask = [], []
        for i, task in enumerate(tasks):
            state_i = raw_state[i] if isinstance(raw_state, torch.Tensor) and raw_state.ndim > 1 else raw_state
            full_prompt = self.build_prompt(task, state_i, response_key=response_key)
            tokens = self.encode_text(full_prompt, add_bos=add_bos, add_eos=add_eos)

            if len(tokens) > length:
                raise ValueError(
                    f"{i} sample with prompt '{full_prompt}' exceeds max token length "
                    f"({len(tokens)} > {length})"
                )

            if pad:
                mask = [True] * len(tokens) + [False] * (length - len(tokens))
                tokens = tokens + [self.pad_id] * (length - len(tokens))
            else:
                mask = [True] * len(tokens)

            tokenized_prompt.append(np.asarray(tokens, dtype=np.int64))
            prompt_pad_mask.append(np.asarray(mask, dtype=bool))

        return (
            torch.as_tensor(np.stack(tokenized_prompt), device=out_device),
            torch.as_tensor(np.stack(prompt_pad_mask), device=out_device),
        )

    def encode_subtask_target(self, subtask: list[str] | str, *, device=None, max_len: int | None = None):
        subtasks = [subtask] if isinstance(subtask, str) else list(subtask)
        encoded = [self.encode_text(text, add_bos=False, add_eos=True) for text in subtasks]
        length = max(len(tokens) for tokens in encoded) if max_len is None else max_len

        tokenized, mask = [], []
        for i, tokens in enumerate(encoded):
            if len(tokens) > length:
                raise ValueError(f"{i} subtask target exceeds max length ({len(tokens)} > {length})")
            tokenized.append(np.asarray(tokens + [self.pad_id] * (length - len(tokens)), dtype=np.int64))
            mask.append(np.asarray([True] * len(tokens) + [False] * (length - len(tokens)), dtype=bool))
        return torch.as_tensor(np.stack(tokenized), device=device), torch.as_tensor(np.stack(mask), device=device)

