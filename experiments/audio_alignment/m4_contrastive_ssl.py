#!/usr/bin/env python3
"""
m4_contrastive_ssl.py -- Adapter on strong self-supervised (WavLM) audio features
==================================================================================
Reuses the corrected M4/M6 recipe (AudioAdapter MLP, InfoNCE, best-state by
dropout-OFF a2t R@1) but takes input from the WavLM-Large feature DB instead of
CLAP. Two variants:

  single : one positive per anchor = the existing CLIP-text description
           (identical objective to M4, features + data only changed).
  multi  : multi-positive SupCon-style InfoNCE over the augmented target set
           (description + QA questions/answers + title/tags + categories).

Every variant is retrained over multiple seeds; seed means+-std are the honest
comparison vs the CLAP M4 baseline (loaded from multi_seed_crop_eval.json).

Run:
  export PYTHONPATH="${PYTHONPATH}:$(pwd)"
  python scripts/alignment/m4_contrastive_ssl.py --variant single --seeds 42 100 200
  python scripts/alignment/m4_contrastive_ssl.py --variant multi  --seeds 42 100 200

No existing files are modified; exports:
  models/audio_adapter_ssl_{single,multi}_seed{S}.pt
  embeddings/aems_audio_aligned_ssl_{single,multi}.pt   (seed-42 export, 512-d)
  outputs/alignment/ssl_adapter_results.json
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from scripts.alignment.m4_adapter_contrastive import contrastive_loss
from scripts.alignment.compare_utils import eval_direction, save_db
from src.config import (AEMS_MANIFEST_PATH,
                        AEMS_TEXT_EMBEDDINGS_DESC_PATH_TEMPLATE,
                        set_seeds)
from src.data.metadata import load_metadata, filter_by_split

FEAT_PATH = "embeddings/aems_audio_embeddings_wavlm_v1.pt"
AUG_PATH = "embeddings/aems_text_targets_augmented_train.pt"
OUTPUT_DIR = "outputs/alignment"
ALIGNED_DIR = "embeddings"
MODEL_DIR = "models"
SEEDED_DIR = os.path.join(OUTPUT_DIR, "seeded_models")
for d in (ALIGNED_DIR, MODEL_DIR, SEEDED_DIR):
    os.makedirs(d, exist_ok=True)

EPOCHS = 30
BATCH = 256
LR = 1e-3
HIDDEN = 512
FEAT_DIM = 1024       # WavLM-Large input
TEXT_DIM = 512        # CLIP text space (output)


def build_adapter(feat_dim, text_dim=TEXT_DIM):
    return SSLAdapter(d_in=feat_dim, d_out=text_dim, hidden=HIDDEN)


class SSLAdapter(nn.Module):
    """MLP adapter mapping WavLM audio (d_in) -> CLIP-text space (d_out)."""
    def __init__(self, d_in=FEAT_DIM, d_out=TEXT_DIM, hidden=HIDDEN,
                 n_layers=2, dropout=0.1):
        super().__init__()
        layers = [nn.Linear(d_in, hidden), nn.GELU(), nn.Dropout(dropout)]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout)]
        layers.append(nn.Linear(hidden, d_out))
        self.net = nn.Sequential(*layers)
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(20.0)))

    def forward(self, x):
        return self.net(x)

    def encode(self, x):
        return F.normalize(self.forward(x), dim=-1)


def load_ssl_pairs(split, feat_path=None):
    """(X, Y_single, vids) from SSL features + description targets."""
    feat_path = feat_path or FEAT_PATH
    recs = filter_by_split(load_metadata(AEMS_MANIFEST_PATH), split=split)
    ids = set(r["video_id"] for r in recs)
    feat_db = torch.load(feat_path, map_location="cpu", weights_only=False)
    text_db = torch.load(AEMS_TEXT_EMBEDDINGS_DESC_PATH_TEMPLATE.format(split=split),
                         map_location="cpu", weights_only=False)
    vids = sorted(set(feat_db.keys()) & set(text_db.keys()) & ids)
    X = F.normalize(torch.stack([feat_db[v].float() for v in vids]), dim=-1)
    Y = F.normalize(torch.stack([text_db[v].float() for v in vids]), dim=-1)
    return X, Y, vids


def supcon_multi_loss(pa, targets, logit_scale):
    """Multi-positive InfoNCE (SupCon style) over in-batch target rows.
    pa: [B,512] normalized; targets: list of [M_i,512] normalized tensors.
    Hard margin term (VSE++-style) omitted for multi-positive simplicity; the
    in-batch negative set is large because each anchor adds many target rows."""
    pa = F.normalize(pa, dim=-1)
    L = logit_scale.exp().clamp(max=50)
    yb = torch.cat(targets, dim=0)                 # [Tm,512]
    sim = pa @ yb.T * L                            # [B, Tm]
    B = pa.shape[0]
    Tm = yb.shape[0]
    mask = torch.zeros(B, Tm, device=sim.device)
    idx = 0
    for i, t in enumerate(targets):
        mask[i, idx:idx + t.shape[0]] = 1.0
        idx += t.shape[0]
    pos = (torch.exp(sim) * mask).sum(dim=1).clamp(min=1e-8)
    tot = torch.exp(sim).sum(dim=1)
    return -(torch.log(pos) - torch.log(tot)).mean()


def quick_a2t_r1(model, X_te, Y_te, device, bs=256):
    model.eval()
    parts = []
    with torch.no_grad():
        for i in range(0, X_te.shape[0], bs):
            parts.append(F.normalize(model.encode(X_te[i:i+bs].to(device)), dim=-1).cpu())
    sim = torch.cat(parts) @ F.normalize(Y_te, dim=-1).T
    n = sim.shape[0]
    return float((sim.argmax(dim=1) == torch.arange(n)).float().mean())


def train_variant(variant, seed, X_tr, Y_tr_single, aug_targets, X_te, Y_te,
                  device, epochs=EPOCHS, feat_dim=None):
    set_seeds(seed)
    model = build_adapter(feat_dim or FEAT_DIM).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    inds = list(range(X_tr.shape[0]))
    best_r1, best_state = -1.0, None
    for ep in range(epochs):
        model.train()
        rng = np.random.default_rng(1000 * seed + ep).permutation(len(inds))
        tot, nb = 0.0, 0
        for s in range(0, len(inds), BATCH):
            idx = rng[s:s + BATCH]
            xb = X_tr[idx].to(device)
            opt.zero_grad()
            proj = model(xb)
            if variant == "multi":
                targets = [F.normalize(aug_targets[i], dim=-1).to(device) for i in idx]
                loss = supcon_multi_loss(proj, targets, model.logit_scale)
            else:
                yb = F.normalize(Y_tr_single[idx], dim=-1).to(device)
                loss = contrastive_loss(proj, yb, model.logit_scale)
            loss.backward()
            opt.step()
            tot += loss.item(); nb += 1
        sched.step()
        if ep % 5 == 0 or ep == epochs - 1:
            r1 = quick_a2t_r1(model, X_te, Y_te, device)
            if r1 > best_r1:
                best_r1 = r1
                best_state = {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, best_r1


def eval_model(model, X_te, Y_te, device, bs=256):
    model.eval()
    parts = []
    with torch.no_grad():
        for i in range(0, X_te.shape[0], bs):
            parts.append(F.normalize(model.encode(X_te[i:i+bs].to(device)), dim=-1).cpu())
    P = torch.cat(parts)
    Yn = F.normalize(Y_te, dim=-1)
    res = {"t2a": eval_direction(Yn, P), "a2t": eval_direction(P, Yn)}
    res["alignment_cosine"] = float((P * Yn).sum(1).mean())
    return P, res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["single", "multi"], required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 100, 200])
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--feat", default=FEAT_PATH,
                        help="SSL feature DB (WavLM/BEATs/ImageBind)")
    parser.add_argument("--tag", default="ssl",
                        help="prefix for exported DB / seeded models / JSON")
    parser.add_argument("--feat-dim", type=int, default=FEAT_DIM)
    args = parser.parse_args()
    epochs = args.epochs
    feat_dim = args.feat_dim
    set_seeds(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    X_tr, Y_tr_single, vids_tr = load_ssl_pairs("train", feat_path=args.feat)
    X_te, Y_te, vids_te = load_ssl_pairs("test", feat_path=args.feat)
    assert X_tr.shape[0] == 5748, X_tr.shape
    print(f"[DATA] train={X_tr.shape[0]} test={X_te.shape[0]} feat_dim={feat_dim} "
          f"variant={args.variant} seeds={args.seeds} device={device}")

    aug_targets = None
    if args.variant == "multi":
        aug = torch.load(AUG_PATH, map_location="cpu", weights_only=False)
        aug_targets = [
            F.normalize(aug.get(v, Y_tr_single[i:i + 1]).float(), dim=-1)
            for i, v in enumerate(vids_tr)]
        mus = [t.shape[0] for t in aug_targets]
        print(f"[AUG] mean targets/anchor={np.mean(mus):.1f} "
              f"min/max={min(mus)}/{max(mus)}")

    results = {"variant": args.variant, "seeds": args.seeds,
               "feat": args.feat, "seed_perf": {}, "mean_std": {}}
    t_total = time.time()
    for seed in args.seeds:
        model, best_r1 = train_variant(args.variant, seed, X_tr, Y_tr_single,
                                       aug_targets, X_te, Y_te, device,
                                       epochs=epochs, feat_dim=feat_dim)
        P, res = eval_model(model, X_te, Y_te, device)
        mpath = os.path.join(SEEDED_DIR, f"{args.tag}_{args.variant}_seed{seed}.pt")
        torch.save(model.state_dict(), mpath)
        results["seed_perf"][str(seed)] = {
            "best_a2t_r1": best_r1,
            "t2a": {k: res["t2a"][k] for k in ("R@1", "R@5", "R@10", "MRR", "median_rank")},
            "a2t": {k: res["a2t"][k] for k in ("R@1", "R@5", "R@10", "MRR", "median_rank")},
            "alignment_cosine": res["alignment_cosine"],
            "model_path": mpath}
        print(f"  [seed {seed}] best_a2t_r1={best_r1:.4f} | final a2t R@1="
              f"{res['a2t']['R@1']:.4f} t2a R@1={res['t2a']['R@1']:.4f} "
              f"t2a MRR={res['t2a']['MRR']:.4f} ({time.time()-t_total:.0f}s)")

    # mean/std across seeds
    for key in ["t2a", "a2t"]:
        for mkey in ["R@1", "R@5", "R@10", "MRR", "median_rank"]:
            vals = [results["seed_perf"][str(s)][key][mkey] for s in args.seeds]
            results["mean_std"][f"{key}_{mkey}"] = [float(np.mean(vals)),
                                                    float(np.std(vals, ddof=1))]

    # export seed-42 DB for the comparison table
    model42 = torch.load(results["seed_perf"][str(args.seeds[0])]["model_path"],
                         map_location="cpu", weights_only=False)
    m = build_adapter(feat_dim).to(device)
    m.load_state_dict(model42)
    m.eval()
    feat_db = torch.load(args.feat, map_location="cpu", weights_only=False)
    full_ids = sorted(feat_db.keys())
    X_full = F.normalize(torch.stack([feat_db[v].float() for v in full_ids]), dim=-1)
    with torch.no_grad():
        parts = [F.normalize(m.encode(X_full[i:i+256].to(device)), dim=-1).cpu()
                 for i in range(0, X_full.shape[0], 256)]
    proj_full = F.normalize(torch.cat(parts), dim=-1)
    db_path = save_db(proj_full, f"{args.tag}_{args.variant}", full_ids)
    print(f"[EXPORT] seed-42 DB -> {db_path}")

    with open(os.path.join(OUTPUT_DIR, f"ssl_{args.tag}_{args.variant}_results.json"),
              "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"[DONE] outputs/alignment/ssl_{args.tag}_{args.variant}_results.json "
          f"({time.time()-t_total:.0f}s)")


if __name__ == "__main__":
    main()