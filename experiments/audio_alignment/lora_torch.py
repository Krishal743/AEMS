#!/usr/bin/env python3
"""LoRA (Hu et al. 2021) adapter modules for PyTorch nn.Linear.
LoRA.proj for zero-init and standard LoRA weight update:
    y = x @ W.T + alpha * (x @ A.T) @ B.T
B zero-initialised so the module starts as the identity transform."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, r=16, alpha=16.0, dropout=0.0):
        super().__init__()
        assert r > 0
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.A = nn.Parameter(torch.zeros(in_features, r))
        self.B = nn.Parameter(torch.zeros(r, out_features))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        nn.init.kaiming_uniform_(self.A, a=5 ** 0.5)

    def forward(self, x):
        out = self.dropout(x) @ self.A @ self.B
        return out * self.scaling


def apply_lora_to_linear(module, r=16, alpha=16.0, dropout=0.1):
    """Wrap an existing nn.Linear into an nn.Module that keeps the base frozen
    and adds a LoRA path. Mutates the parameters of `module` (base) in place and
    returns the wrapper. Assumes inputs/outputs dtype float."""
    in_f = module.in_features
    out_f = module.out_features
    lora = LoRALayer(in_f, out_f, r=r, alpha=alpha, dropout=dropout)
    # match base device/dtype so params live where the model does
    dev = module.weight.device
    dt = module.weight.dtype
    lora = lora.to(device=dev, dtype=dt)

    class _LoRALinear(nn.Module):
        def __init__(self, base, lora_mod):
            super().__init__()
            self.base = base
            self.lora = lora_mod

        def forward(self, x):
            return self.base(x) + self.lora(x)

    wrapper = _LoRALinear(module, lora)
    return wrapper


def inject_lora(clap_model, targets=None, r=16, alpha=16.0, dropout=0.1):
    """Insert LoRA adapters inside the HTSAT audio backend. Returns list of
    (name) of LoRA-injected Linear modules.

    targets: substrings matched against Linear module names, or None for the
    default set (attn.qkv, attn.proj, mlp.fc1, mlp.fc2).
    """
    if targets is None:
        targets = ["attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"]

    ab = clap_model.audio_branch
    # freeze the whole audio backend base weights (LoRA wrappers add trainable params)
    for p in ab.parameters():
        p.requires_grad_(False)
    for p in clap_model.audio_projection.parameters():
        p.requires_grad_(True)

    inserted = []

    def _recurse(parent, name_prefix, depth=0):
        for child_name, child in parent.named_children():
            full = f"{name_prefix}.{child_name}" if name_prefix else child_name
            if isinstance(child, nn.Linear):
                if any(t in full for t in targets):
                    wrapper = apply_lora_to_linear(child, r=r, alpha=alpha, dropout=dropout)
                    setattr(parent, child_name, wrapper)
                    inserted.append(full)
                continue
            if isinstance(child, nn.Module):
                _recurse(child, full, depth + 1)

    _recurse(ab, "")
    return inserted


def count_trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_total(model):
    return sum(p.numel() for p in model.parameters())
