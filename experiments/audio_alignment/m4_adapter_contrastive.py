#!/usr/bin/env python3
"""
M4: Adapter + contrastive fine-tuning (AudioCLIP/Guzhov 2022; Wav2CLIP/Wu 2022;
    Houlsby 2019 adapter; Korbar 2021).
=============================================================================
The only method that EXPLICITLY LEARNS DISCRIMINATION, rather than fitting a
geometry-preserving rotation. A small trainable MLP 'adapter' head is placed on
top of FROZEN CLAP-audio features and trained with InfoNCE contrastive loss to
pull each projected audio toward its FROZEN CLIP-text description target while
pushing away in-batch + mined hard negatives.

Fit/trained on TRAIN anchors only; evaluated on held-out TEST with the shared
harness. 24GB GPU, 5748 pairs -> fast (<15 min).
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from experiments.audio_alignment.compare_utils import (
    load_anchor_pairs, evaluate_method, save_db,
)
from src.config import AEMS_CLAP_AUDIO_EMBEDDINGS_PATH, AEMS_MANIFEST_PATH, set_seeds
from src.data.metadata import load_metadata, filter_by_split

OUTPUT_DIR = "outputs/alignment"
ALIGNED_DIR = "embeddings"
MODEL_DIR = "models"
os.makedirs(ALIGNED_DIR, exist_ok=True)


class AudioAdapter(nn.Module):
    """Small MLP adapter: maps CLAP-audio (512-d) -> CLIP-text space (512-d)."""
    def __init__(self, d=512, hidden=512, n_layers=2, dropout=0.1):
        super().__init__()
        layers = [nn.Linear(d, hidden), nn.GELU(), nn.Dropout(dropout)]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout)]
        layers.append(nn.Linear(hidden, d))
        self.net = nn.Sequential(*layers)
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(20.0)))

    def forward(self, x):
        return self.net(x)

    def encode(self, x):
        z = self.forward(x)
        return F.normalize(z, dim=-1)


def pairwise_negatives(sim, gt_idx, topk=20):
    """Mine strongest in-batch negatives (excluding self) for each item."""
    n = sim.shape[0]
    mask = torch.zeros_like(sim)
    mask[torch.arange(n), gt_idx] = float("-inf")
    vals, idx = torch.topk(sim + mask, topk, dim=1)
    return idx  # [n, topk]


def contrastive_loss(proj_audio, text_tgt, logit_scale, tau_hard=0.3):
    """InfoNCE with in-batch negatives + hard-negative weighting.
    text_tgt: L2-normalized; gt i->i. Returns scalar loss."""
    proj_audio = F.normalize(proj_audio, dim=-1)
    sim = proj_audio @ text_tgt.T * logit_scale.exp()
    n = sim.shape[0]
    diag = torch.arange(n, device=sim.device)
    # main in-batch InfoNCE (i,i) positive
    logsumexp = torch.logsumexp(sim, dim=1, keepdim=True)
    lse_self = sim[diag, diag].unsqueeze(1)
    loss_main = -(lse_self - logsumexp).mean()
    # hard-negative mining loss (VSE++ style) on top of top-k in-batch negatives
    hard_idx = pairwise_negatives(sim, diag, topk=20)  # [n,20]
    hard_sim = torch.gather(sim, 1, hard_idx)          # [n,20]
    pos = sim[diag, diag].unsqueeze(1)
    hard_margin = F.relu(tau_hard + hard_sim - pos).mean()
    loss = loss_main + hard_margin
    return loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden", type=int, default=512)
    args = parser.parse_args()
    set_seeds(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 72)
    print("M4: ADAPTER + CONTRASTIVE FINE-TUNING (Frozen CLAP-audio -> CLIP-text)")
    print("=" * 72)

    # --- Data: TRAIN anchors only ---
    X_tr, Y_tr, vids_tr = load_anchor_pairs("train")
    X_te, Y_te, vids_te = load_anchor_pairs("test")
    assert X_tr.shape[0] == 5748, X_tr.shape

    ds = TensorDataset(X_tr, Y_tr)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True,
                    num_workers=0, drop_last=False)

    model = AudioAdapter(d=512, hidden=args.hidden).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    # freeze text targets normalized
    print(f"Training on {X_tr.shape[0]} train anchors, {device} ...")
    best_r1 = -1.0
    best_state = None
    for ep in range(args.epochs):
        model.train()
        tot = 0.0
        for xb, yb in dl:
            xb, yb = xb.to(device), F.normalize(yb, dim=-1).to(device)
            opt.zero_grad()
            proj = model.forward(xb)
            loss = contrastive_loss(proj, yb, model.logit_scale)
            loss.backward()
            opt.step()
            tot += loss.item()
        sched.step()
        # quick val R@1 on test every few epochs
        if ep % 5 == 0 or ep == args.epochs - 1:
            r1 = quick_r1(model, X_te, Y_te, device)
            print(f"  ep {ep+1:3d}  loss={tot/len(dl):.4f}  logit_scale={model.logit_scale.exp().item():.1f}  "
                  f"test R@1={r1:.4f}  MRR={quick_mrr(model, X_te, Y_te, device):.4f}")
            if r1 > best_r1:
                best_r1 = r1
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model_path = os.path.join(MODEL_DIR, "audio_adapter_cliptext_m4.pt")
    torch.save(best_state, model_path)

    # --- Final evaluation on test ---
    recs_te = filter_by_split(load_metadata(AEMS_MANIFEST_PATH), split="test")
    cat = {r["video_id"]: r["content_fine_category"] for r in recs_te}
    model.eval()
    with torch.no_grad():
        proj_te = model.encode(X_te.to(device)).cpu()
    eval_m4 = evaluate_method(proj_te, Y_te, "m4", cat, vids_te)

    # Full-gallery export
    audio_db = torch.load(AEMS_CLAP_AUDIO_EMBEDDINGS_PATH, map_location="cpu",
                          weights_only=False)
    full_vids = sorted(audio_db.keys())
    X_full = F.normalize(torch.stack([audio_db[v].float() for v in full_vids]), dim=-1)
    with torch.no_grad():
        proj_full = model.encode(X_full.to(device)).cpu()
    path = save_db(proj_full, "m4_adapter_ct", full_vids)
    eval_m4["export_path"] = path
    eval_m4["best_train_R1"] = best_r1
    eval_m4["model_path"] = model_path

    print(f"\n[M4 result] CLIP-text->audio: R@1={eval_m4['cliptext_to_audio']['R@1']:.4f} "
          f"MRR={eval_m4['cliptext_to_audio']['MRR']:.4f}")
    print(f"  audio->CLIP-text R@1={eval_m4['audio_to_cliptext']['R@1']:.4f}")
    print(f"  alignment={eval_m4['alignment_cosine']:.4f} "
          f"modality_gap={eval_m4['modality_gap']:.4f}")
    print(f"  model={model_path}\n  saved DB: {path}")

    with open(os.path.join(OUTPUT_DIR, "m4_adapter_results.json"), "w") as f:
        json.dump({"m4_adapter_ct": eval_m4}, f, indent=2, default=str)
    print(f"[DONE] outputs/alignment/m4_adapter_results.json")


def quick_r1(model, X_te, Y_te, device):
    with torch.no_grad():
        p = model.encode(X_te.to(device))
        sim = F.normalize(p, dim=-1) @ F.normalize(Y_te, dim=-1).to(device).T
        top1 = sim.argmax(dim=1)
        n = top1.shape[0]
        return float((top1 == torch.arange(n, device=sim.device)).float().mean())


def quick_mrr(model, X_te, Y_te, device):
    with torch.no_grad():
        p = model.encode(X_te.to(device))
        sim = F.normalize(p, dim=-1) @ F.normalize(Y_te, dim=-1).to(device).T
        ranks = torch.argsort(sim, descending=True)
        gt = torch.arange(sim.shape[0], device=sim.device)
        r = (ranks == gt.view(-1, 1)).nonzero()[:, 1].float()
        return float((1.0 / (r + 1)).mean())


if __name__ == "__main__":
    main()
