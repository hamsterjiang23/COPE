"""Frozen-backbone interface candidates for the registered v3 search.

KV here is synthetic prefix memory, not a reconstruction of native prompt caches.
All candidates share a teacher-side latent, with explicit student-side decoding.
"""

from __future__ import annotations

import re
from fractions import Fraction

import torch
from torch import nn
from torch.nn import functional as F


def numeric_answer(text):
    """Fixed priority: ####, balanced boxed numeric, explicit final-answer line.

    No fallback to the last arbitrary number in a reasoning chain.
    """
    candidates = re.findall(r"####\s*([^\n]+)", text)
    if not candidates:
        for match in re.finditer(r"\\boxed\s*\{", text):
            start, depth, end = match.end(), 1, match.end()
            while end < len(text) and depth:
                depth += (text[end] == "{") - (text[end] == "}")
                end += 1
            if depth == 0:
                candidates.append(text[start : end - 1])
    if not candidates:
        candidates = re.findall(r"(?:final answer|the answer is)\s*[:=]?\s*([^\n]+)", text, re.I)
    if not candidates:
        return None
    value = candidates[-1].strip().strip("$ ")
    value = re.sub(r"\\(?:d?frac)\{([+-]?[\d.]+)\}\{([+-]?[\d.]+)\}", r"\1/\2", value)
    match = re.fullmatch(
        r"([+-]?(?:\d[\d,]*(?:\.\d+)?|\.\d+)(?:\s*/\s*[+-]?(?:\d+(?:\.\d+)?|\.\d+))?)\s*[.!]?",
        value,
    )
    if not match:
        return None
    try:
        parts = match[1].replace(",", "").replace(" ", "").split("/")
        result = Fraction(parts[0])
        if len(parts) == 2:
            result /= Fraction(parts[1])
        return str(result)
    except (ValueError, ZeroDivisionError):
        return None


class SearchBridge(nn.Module):
    """One recipe: embedding prefix, additional KV, or question-replacing KV."""

    def __init__(self, teacher_size, student_config, kind, slots=32, width=512, rank=32):
        super().__init__()
        if kind not in {"embedding", "kv_add", "kv_replace"}:
            raise ValueError(kind)
        self.kind, self.slots, self.width = kind, slots, width
        self.layers = student_config.num_hidden_layers
        self.heads = student_config.num_key_value_heads
        self.head_dim = student_config.head_dim
        self.projector = nn.Sequential(
            nn.LayerNorm(teacher_size),
            nn.Linear(teacher_size, width),
            nn.GELU(),
            nn.Linear(width, width),
            nn.LayerNorm(width),
        )
        self.constant = nn.Parameter(torch.zeros(1, slots, width))
        if kind == "embedding":
            self.reader = nn.Linear(width, student_config.hidden_size)
        else:
            self.reader = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(width, rank),
                        nn.GELU(),
                        nn.Linear(rank, 2 * self.heads * self.head_dim),
                    )
                    for _ in range(self.layers)
                ]
            )

    def place(self, teacher_device, student_device):
        self.projector.to(teacher_device)
        self.constant.data = self.constant.data.to(teacher_device)
        self.reader.to(student_device)
        return self

    def encode(self, hidden, mode="correct"):
        if mode == "constant":
            return self.constant
        # Inputs are unpadded, single-example teacher states.
        reduced = F.adaptive_avg_pool1d(hidden.transpose(1, 2), self.slots).transpose(1, 2)
        return self.projector(reduced.float())

    def decode(self, z, student):
        from transformers.cache_utils import DynamicCache
        from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

        parameter = next(self.reader.parameters())
        z = z.to(parameter)
        dtype = next(student.parameters()).dtype
        if self.kind == "embedding":
            return self.reader(z).to(dtype)
        cache = DynamicCache()
        positions = torch.arange(self.slots, device=z.device).unsqueeze(0)
        cos, sin = student.model.rotary_emb(z.to(dtype), positions)
        for index, reader in enumerate(self.reader):
            kv = reader(z).to(dtype).view(1, self.slots, 2, self.heads, self.head_dim)
            key, value = kv[:, :, 0].transpose(1, 2), kv[:, :, 1].transpose(1, 2)
            # Keys obey the student's normalization and RoPE convention exactly once.
            key = student.model.layers[index].self_attn.k_norm(key)
            _, key = apply_rotary_pos_emb(key, key, cos, sin)
            cache.update(key, value, index)
        return cache


def conditioned_inputs(student, bridge, memory, prompt_ids, target_ids=None):
    """No original question tokens enter kv_replace; one fixed start token does."""
    device = next(student.parameters()).device
    prefix = prompt_ids[-1:] if bridge and bridge.kind == "kv_replace" else prompt_ids
    tokens = prefix + (target_ids or [])
    ids = torch.tensor([tokens], device=device)
    slots = bridge.slots if bridge else 0
    labels = ids.clone()
    labels[:, : len(prefix)] = -100
    if bridge and bridge.kind == "embedding":
        embeddings = torch.cat([memory, student.get_input_embeddings()(ids)], dim=1)
        labels = torch.cat([torch.full((1, slots), -100, device=device), labels], dim=1)
        inputs = {"inputs_embeds": embeddings}
        positions = torch.arange(len(tokens) + slots, device=device).unsqueeze(0)
    else:
        inputs = {"input_ids": ids}
        positions = torch.arange(slots, slots + len(tokens), device=device).unsqueeze(0)
        if bridge:
            inputs["past_key_values"] = memory
    inputs.update(
        attention_mask=torch.ones((1, slots + len(tokens)), dtype=torch.long, device=device),
        position_ids=positions,
    )
    if target_ids is not None:
        inputs["labels"] = labels
    return inputs
