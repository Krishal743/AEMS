#!/usr/bin/env python3
"""
Validate the linear projection on held-out TEST anchors.
=========================================================
The projection W is fit on TRAIN videos only. This script evaluates it on the
disjoint TEST videos, so there is no leakage. It verifies whether projecting
CLAP-audio into CLIP-text space actually:
  - raises positive-pair alignment (Wang & Isola 2020)
  - shrinks the modality gap (Liang et al. 2022)
  - reduces hubness (Radovanović et al. 2010)
  - enables cross-modal CLIP-text-query -> audio retrieval (previously ~random)

Run:
  export PYTHONPATH="${PYTHONPATH}:$(pwd)"
  python experiments/audio_alignment/validate_projection.py
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from src.config import (
    AEMS_AUDIO_EMBEDDINGS_PATH,
    AEMS_TEXT_EMBEDDINGS_DESC_PATH_TEMPLATE,
    AEMS_MANIFEST_PATH,
    set_seeds,
)
from src.data.metadata import load_metadata, filter_by_split

MODEL_PATH = "models/audio_to_cliptext_projection.pt"
OUTPUT_DIR = "outputs/alignment"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=MODEL_PATH)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    set_seeds(args.seed)

    print("=" * 72)
    print("VALIDATE PROJECTION (held-out TEST set)")
    print("=" * 72)

    ckpt = torch.load(args.model, map_location="cpu", weights_only=False)
    W = ckpt["W"].float()
    b = ckpt["b"].float()
    print(f"[MODEL] method='{ckpt['method']}' (fit on "
          f"{ckpt['n_train_anchors']} train anchors)")

    # ---- Load test anchors ----
    records = filter_by_split(load_metadata(AEMS_MANIFEST_PATH), split="test")
    test_ids = set(r["video_id"] for r in records)
    audio_db = torch.load(AEMS_AUDIO_EMBEDDINGS_PATH, map_location="cpu",
                          weights_only=False)
    text_db = torch.load(AEMS_TEXT_EMBEDDINGS_DESC_PATH_TEMPLATE.format(split="test"),
                         map_location="cpu", weights_only=False)
    anchors = sorted(set(audio_db.keys()) & set(text_db.keys()) & test_ids)
    n = len(anchors)
    print(f"[DATA] Test anchor pairs: {n}")

    X = F.normalize(torch.stack([audio_db[v].float() for v in anchors]), dim=-1)
    Y = F.normalize(torch.stack([text_db[v].float() for v in anchors]), dim=-1)

    # ---- Project audio ----
    with torch.no_grad():
        X_proj = F.normalize(X @ W.T + b.view(1, -1), dim=-1)

    report = {"framework": ["Wang & Isola 2020", "Liang et al. 2022",
                            "Radovanović 2010", "Chen 2021"],
              "method": ckpt["method"], "n_test_anchors": int(n)}

    # ---- 1. Alignment (Wang & Isola) ----
    print("\n[1] ALIGNMENT (Wang & Isola 2020)")
    pos_before = float((X * Y).sum(dim=1).mean().item())
    pos_after = float((X_proj * Y).sum(dim=1).mean().item())
    print(f"  Positive-pair cosine: BEFORE={pos_before:.4f}  "
          f"AFTER projection={pos_after:.4f}")
    report["alignment"] = {"before_cosine": pos_before, "after_cosine": pos_after}

    # ---- 2. Modality gap (Liang) ----
    print("\n[2] MODALITY GAP (Liang et al. 2022)")
    before_centroid = X.mean(dim=0)
    after_centroid = X_proj.mean(dim=0)
    text_centroid = Y.mean(dim=0)
    gap_before = 1.0 - float((F.normalize(before_centroid, dim=0) *
                              F.normalize(text_centroid, dim=0)).sum())
    gap_after = 1.0 - float((F.normalize(after_centroid, dim=0) *
                             F.normalize(text_centroid, dim=0)).sum())
    print(f"  Centroid cosine-dist to text: BEFORE={gap_before:.4f}  "
          f"AFTER={gap_after:.4f}")
    report["modality_gap"] = {"before": gap_before, "after": gap_after}

    # ---- 3. Hubness + cross-modal retrieval (Radovanović; Chen) ----
    print("\n[3] HUBNESS & CROSS-MODAL RETRIEVAL (CLIP-text -> audio)")
    ks = [1, 5, 10]
    hubness = {}
    retrieval = {}
    for tag, gallery in [("before_raw_clap", X), ("after_projected", X_proj)]:
        sim = Y @ gallery.T
        # R@K
        line = f"  {tag:<18}: "
        rk = {}
        for k in ks:
            topk = torch.argsort(sim, descending=True)[:, :k]
            hit = (topk == torch.arange(n).view(-1, 1)).any(dim=1).float().mean()
            rk[f"R@{k}"] = float(hit)
            line += f"R@{k}={hit:.4f}  "
        # hubness k-occ skewness (K=5)
        K = 5
        idx = torch.argsort(sim, descending=True)[:, :K]
        occ = torch.zeros(n)
        occ.index_add_(0, idx.flatten(), torch.ones(n * K))
        skew = float((((occ - occ.mean()) / (occ.std() + 1e-8)) ** 3).mean())
        line += f"skew(K=5)={skew:.2f}"
        print(line)
        retrieval[tag] = rk
        hubness[tag] = skew
    report["retrieval"] = retrieval
    report["hubness_skew"] = hubness

    out_path = os.path.join(OUTPUT_DIR, "projection_validation.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[DONE] Validation report: {out_path}")


if __name__ == "__main__":
    main()
