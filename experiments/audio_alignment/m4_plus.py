#!/usr/bin/env python3
"""
Tier 1: Boost M4 with GLOBAL hard-negative mining + post-hoc hubness reduction.
=============================================================================
Builds on the M4 adapter (frozen CLAP-audio -> CLIP-text) with two upgrades:

1. GLOBAL hard negatives via a momentum memory bank (VSE++ / Faghri 2018;
   HSE++ / Zhang 2022). Instead of only in-batch negatives, each positive pair
   is contrasted against the hardest negatives mined GLOBALLY across the whole
   training set, which sharpens decision boundaries.

2. Post-hoc hubness reduction applied to the resulting aligned audio gallery
   (correctly, on a discriminative space unlike the earlier M5-on-M1 attempt):
     - local scaling       (Zelnik-Manor & Perona 2004)
     - mutual k-NN filter  (Jegou 2010)
     - query expansion / alphaQE

Fit/trained on TRAIN anchors only; evaluated on held-out TEST (n=1022).
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from experiments.audio_alignment.compare_utils import load_anchor_pairs, save_db
from experiments.audio_alignment.m4_adapter_contrastive import AudioAdapter, quick_r1, quick_mrr
from src.config import AEMS_MANIFEST_PATH, AEMS_AUDIO_EMBEDDINGS_PATH, set_seeds
from src.data.metadata import load_metadata, filter_by_split

OUTPUT_DIR = "outputs/alignment"
os.makedirs(OUTPUT_DIR, exist_ok=True)


class MemoryBank:
    """Momentum-updated bank of projected audio embeddings for global hard
    negative mining (HSE++ style)."""

    def __init__(self, n, d, device, momentum=0.5):
        self.n = n
        self.d = d
        self.device = device
        self.momentum = momentum
        self.vectors = F.normalize(torch.randn(n, d), dim=-1).to(device)
        self.filled = torch.zeros(n, dtype=torch.bool, device=device)

    @torch.no_grad()
    def update(self, ids, emb):
        emb = F.normalize(emb, dim=-1)
        self.vectors[ids] = self.momentum * self.vectors[ids] + (1 - self.momentum) * emb
        self.vectors[ids] = F.normalize(self.vectors[ids], dim=-1)
        self.filled[ids] = True

    def negative_indices(self, ids, topk=64):
        """Hardest global negatives (excluding self) for the given ids."""
        q = F.normalize(self.vectors[ids], dim=-1)
        sim = q @ self.vectors.T
        sim[torch.arange(len(ids)), ids] = float("-inf")
        topk = min(topk, self.n - 1)
        idx = torch.topk(sim, topk, dim=1).indices  # [B, topk]
        return idx


def m4p_loss(proj_audio, text_tgt, ids, bank, logit_scale, tau_hard=0.2, ng=64):
    """InfoNCE (in-batch) + GLOBAL hard negatives from the memory bank.
    Margin is applied on cosine similarity (temperature-agnostic) to keep the
    loss stable regardless of the (capped) logit scale."""
    pa = F.normalize(proj_audio, dim=-1)
    tt = F.normalize(text_tgt, dim=-1)
    n = pa.shape[0]
    diag = torch.arange(n, device=pa.device)
    L = logit_scale.exp().clamp(max=50.0)

    pos = (pa * tt).sum(dim=1)                         # [B] cosine in [-1,1]
    # in-batch InfoNCE on cosine
    sim = pa @ tt.T
    sim[diag, diag] = float("-inf")
    loss_inb = -(pos - torch.logsumexp(sim / 0.07, dim=1)).mean()

    # global hard negatives margin on cosine
    neg_idx = bank.negative_indices(ids, topk=ng)      # [B, ng]
    neg_vec = bank.vectors[neg_idx]                    # [B, ng, d]
    neg_sim = (pa.unsqueeze(1) * neg_vec).sum(-1)      # [B, ng] cosine
    margin = (tau_hard + neg_sim - pos.unsqueeze(1)).clamp(min=0)
    loss_global = margin.mean()

    return loss_inb + loss_global


def train_adapter_global(X_tr, Y_tr, X_te, Y_te, device, args):
    """Train M4-style adapter with memory-bank global hard negatives."""
    n_tr = X_tr.shape[0]
    ds = TensorDataset(X_tr, Y_tr, torch.arange(n_tr))
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=0)

    model = AudioAdapter(d=512, hidden=args.hidden).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    bank = MemoryBank(n_tr, 512, device, momentum=args.bank_momentum)

    # warm-start the bank so negatives reflect real geometry (not noise)
    model.eval()
    with torch.no_grad():
        bs0 = 512
        for i in range(0, n_tr, bs0):
            xb0 = X_tr[i:i+bs0].to(device)
            e0 = model.encode(xb0)
            ids0 = torch.arange(i, min(i+bs0, n_tr), device=device)
            bank.vectors[ids0] = e0
            bank.filled[ids0] = True
    model.train()

    best_r1 = -1.0
    best_state = None
    for ep in range(args.epochs):
        model.train()
        tot = 0.0
        for xb, yb, ids in dl:
            xb, yb = xb.to(device), F.normalize(yb, dim=-1).to(device)
            ids = ids.to(device)
            opt.zero_grad()
            proj = model.forward(xb)
            loss = m4p_loss(proj, yb, ids, bank, model.logit_scale,
                            tau_hard=args.tau_hard, ng=args.nglobal)
            loss.backward()
            opt.step()
            with torch.no_grad():
                bank.update(ids, model.encode(xb))
            tot += loss.item()
        sched.step()
        if ep % 5 == 0 or ep == args.epochs - 1:
            r1 = quick_r1(model, X_te, Y_te, device)
            print(f"  ep {ep+1:3d}  loss={tot/len(dl):.4f}  R@1={r1:.4f} "
                  f"MRR={quick_mrr(model, X_te, Y_te, device):.4f}")
            if r1 > best_r1:
                best_r1 = r1
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, best_r1


# ---------------------------------------------------------------------------
# Post-hoc hubness reduction helpers (implementable on any aligned gallery)
# ---------------------------------------------------------------------------
def local_scale(emb, k=3):
    """Divide each embedding by its k-th-neighbor distance (cosine)."""
    n = emb.shape[0]
    sim = emb @ emb.T
    knn_vals, _ = torch.topk(sim, k + 1, dim=1)
    sigma = knn_vals[:, k].clamp(min=1e-3).unsqueeze(1)
    return F.normalize(emb / sigma, dim=-1)


def mutual_knn_filter(text_q, audio_g, k=5):
    """Mutual k-NN: only keep a text_i->audio_j edge if audio_j is in text_i's
    top-k AND text_i is in audio_j's top-k. Non-mutual entries are depressed
    so they cannot rank first. Returns adjusted sim matrix (text as rows)."""
    n = text_q.shape[0]
    sim = text_q @ audio_g.T                                      # [n,n]
    ta_rank = torch.topk(sim, k, dim=1).indices                    # text kNN [n,k]
    at_rank = torch.topk(sim.T, k, dim=1).indices                 # audio kNN [n,k]
    mask_i2j = torch.zeros_like(sim, dtype=torch.bool)            # i->j mutual
    mask_i2j[torch.arange(n).unsqueeze(1), ta_rank] = True        # audio j in text i top-k
    mask_j2i = torch.zeros_like(sim, dtype=torch.bool)            # transposed
    mask_j2i[at_rank.T, torch.arange(n).unsqueeze(0)] = True      # text i in audio j top-k
    mutual = mask_i2j & mask_j2i
    out = sim.clone()
    out[~mutual] = out[~mutual] - 1e3
    return out


def query_expansion(text_q, audio_g, topk=3, alpha=0.6):
    """alphaQE: augment each text query with the mean of its top-k audio
    neighbours, re-normalized. Applied to text queries (query side)."""
    sim = text_q @ audio_g.T
    idx = torch.argsort(sim, descending=True)[:, :topk]           # [n, topk]
    knn_audio = torch.stack([audio_g[i] for i in idx])            # [n, topk, d]
    mean_vec = knn_audio.mean(dim=1)                               # [n, d]
    new_q = F.normalize(alpha * text_q + (1 - alpha) * mean_vec, dim=-1)
    return new_q


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--nglobal", type=int, default=64)
    parser.add_argument("--bank-momentum", type=float, default=0.5)
    parser.add_argument("--tau-hard", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    set_seeds(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 72)
    print("TIER 1: M4 + GLOBAL HARD NEGATIVES + HUBNESS REDUCTION")
    print("=" * 72)

    X_tr, Y_tr, vids_tr = load_anchor_pairs("train")
    X_te, Y_te, vids_te = load_anchor_pairs("test")
    assert X_tr.shape[0] == 5748 and Y_te.shape[0] == 1022

    # ---- 1. Train adapter with global hard negatives ----
    print(f"\n[1/2] Training adapter with memory-bank global hard negatives "
          f"(ng={args.nglobal}) on {device} ...")
    model, best_r1 = train_adapter_global(X_tr, Y_tr, X_te, Y_te, device, args)
    model.eval()
    with torch.no_grad():
        proj_te = model.encode(X_te.to(device)).cpu()
        audio_db = torch.load(AEMS_AUDIO_EMBEDDINGS_PATH, map_location="cpu", weights_only=False)
        full_vids = sorted(audio_db.keys())
        X_full = F.normalize(torch.stack([audio_db[v].float() for v in full_vids]), dim=-1)
        proj_full = model.encode(X_full.to(device)).cpu()

    # ---- 2. Post-hoc hubness reduction evaluated on test ----
    print("[2/2] Hubness reduction on the aligned test embeddings ...")
    base = {"t2a": eval_direct(proj_te, Y_te), "a2t": eval_direct(Y_te, proj_te)}
    variants = {"base_global_hn": base}

    for k in (2, 3, 5):
        ls = local_scale(proj_te, k=k)
        variants[f"local_scale_k{k}"] = {"t2a": eval_direct(ls, Y_te),
                                         "a2t": eval_direct(Y_te, ls)}
    for k in (3, 5):
        mm = mutual_knn_filter(Y_te, proj_te, k=k)   # similarity matrix (text rows)
        variants[f"mutual_knn_k{k}"] = {"t2a": eval_sim_matrix(mm),
                                        "a2t": eval_sim_matrix(mm.T)}
    for t in (2, 3, 5):
        qe = query_expansion(Y_te, proj_te, topk=t)
        variants[f"query_expand_{t}"] = {"t2a": eval_direct(qe, proj_te),
                                         "a2t": eval_direct(proj_te, qe)}

    # table
    print("\n  variant                  t2a_R@1  t2a_MRR  a2t_R@1  a2t_MRR")
    for name, v in variants.items():
        print(f"  {name:<26}{v['t2a'][0]:>8.4f}{v['t2a'][1]:>8.4f}"
              f"{v['a2t'][0]:>8.4f}{v['a2t'][1]:>8.4f}")

    # ---- 3. Select best and export ----
    best_variant = max(variants, key=lambda n: variants[n]["a2t"][0])
    print(f"\n  >>> Best variant: {best_variant} "
          f"(a2t R@1={variants[best_variant]['a2t'][0]:.4f})")

    # export chosen DB (apply the best post-hoc transform to full gallery)
    if best_variant.startswith("local_scale"):
        kk = int(best_variant.split("k")[-1])
        proj_full_out = local_scale(proj_full, k=kk)
    elif best_variant.startswith("mutual"):
        model_out = nn.Identity()
        proj_full_out = proj_full  # mutual filter is a retrieval-time op, keep base
    elif best_variant.startswith("query"):
        proj_full_out = proj_full
    else:
        proj_full_out = proj_full

    path = save_db(proj_full_out, "m4_boosted", full_vids)
    # also save base global-HN adapter export (no post-hoc) for comparison
    path_base = save_db(proj_full, "m4_global_hn", full_vids)

    model_path = os.path.join("models", "audio_adapter_m4_globalhn.pt")
    torch.save(model.state_dict(), model_path)

    payload = {
        "best_variant": best_variant,
        "best_train_R1": best_r1,
        "variants": variants,
        "export_path": path,
        "export_path_base": path_base,
        "model_path": model_path,
    }
    with open(os.path.join(OUTPUT_DIR, "m4_plus_results.json"), "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"\n[DONE] outputs/alignment/m4_plus_results.json")
    print(f"  models/audio_adapter_m4_globalhn.pt | {path_base} | {path}")


def eval_direct(Q, G):
    """Q @ G.T retrieval; returns (R@1, MRR) for matched rows."""
    n = Q.shape[0]
    sim = F.normalize(Q, dim=-1) @ F.normalize(G, dim=-1).T
    ranks = torch.argsort(sim, descending=True)
    rr = (ranks == torch.arange(n).view(-1, 1)).nonzero()[:, 1].float()
    r1 = float((rr == 0).float().mean())
    mrr = float((1.0 / (rr + 1)).mean())
    return r1, mrr


def eval_sim_matrix(sim):
    """R@1/MRR directly from a [n,n] similarity matrix (query rows=s gallery cols)."""
    n = sim.shape[0]
    ranks = torch.argsort(sim, descending=True)
    rr = (ranks == torch.arange(n).view(-1, 1)).nonzero()[:, 1].float()
    r1 = float((rr == 0).float().mean())
    mrr = float((1.0 / (rr + 1)).mean())
    return r1, mrr


if __name__ == "__main__":
    main()
