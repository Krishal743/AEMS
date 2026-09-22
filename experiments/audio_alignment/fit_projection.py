#!/usr/bin/env python3
"""
Fit a linear projection mapping CLAP-audio embeddings into CLIP-text space.
============================================================================
Frameworks:
  - Wang et al. (2019), "Learning a Shared Embedding Space for Cross-Modal
    Retrieval" (least-squares / ridge regression between matched pairs)
  - Zhang & Saligrama (2016), "Zero-Shot Learning via Joint Latent Similarity
    Embedding" (orthogonal Procrustes alignment)
  - Grave et al. (2019), "Wasserstein Procrustes" (orthogonal map)
  - Rasiwasia et al. (2010) (CCA baseline)

Goal: learn W so that W @ CLAP_audio ≈ CLIP_text(description) for matched
pairs, using TRAIN videos only. Test videos are held out for validation.

Run:
  export PYTHONPATH="${PYTHONPATH}:$(pwd)"
  python experiments/audio_alignment/fit_projection.py
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from src.config import (
    AEMS_CLAP_AUDIO_EMBEDDINGS_PATH,
    AEMS_TEXT_EMBEDDINGS_DESC_PATH_TEMPLATE,
    AEMS_MANIFEST_PATH,
    set_seeds,
)
from src.data.metadata import load_metadata, filter_by_split

OUTPUT_DIR = "outputs/alignment"
MODEL_PATH = "models/audio_to_cliptext_projection.pt"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs("models", exist_ok=True)


def ridge_fit(X, Y, lam=1e-3):
    """Solve min_W ||W X - Y||^2 + lam||W||^2  -> W = (XX^T + lam I)^-1 X Y^T.
    X: [d, n], Y: [d, n]. Returns W [d, d]."""
    Xt = X @ X.T + lam * torch.eye(X.shape[0], device=X.device)
    W = torch.linalg.solve(Xt, X @ Y.T)
    return W


def procrustes(X, Y):
    """Orthogonal Procrustes: W = U V^T from SVD of X Y^T (centered inputs)."""
    U, _, Vt = torch.linalg.svd(X @ Y.T)
    W = U @ Vt
    return W


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ridge-lam", type=float, default=1e-3)
    parser.add_argument("--use-bias", action="store_true",
                        help="Fit an affine (bias) least-squares map")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    set_seeds(args.seed)

    print("=" * 72)
    print("FIT LINEAR PROJECTION: CLAP-audio -> CLIP-text(description)")
    print("  Frameworks: Wang 2019; Zhang & Saligrama 2016; Grave 2019; "
          "Rasiwasia 2010")
    print("=" * 72)

    # ---- Load paired anchors (TRAIN only) ----
    print("[DATA] Loading train split metadata...")
    records = filter_by_split(load_metadata(AEMS_MANIFEST_PATH), split="train")
    train_ids = set(r["video_id"] for r in records)

    print("[LOAD] Loading embedding DBs...")
    audio_db = torch.load(AEMS_CLAP_AUDIO_EMBEDDINGS_PATH, weights_only=False)
    text_db = torch.load(AEMS_TEXT_EMBEDDINGS_DESC_PATH_TEMPLATE.format(split="train"),
                         weights_only=False)

    anchors = sorted(set(audio_db.keys()) & set(text_db.keys()) & train_ids)
    print(f"[DATA] Train anchor pairs (audio ∩ CLIP-text ∩ train): {len(anchors)}")

    X = torch.stack([audio_db[v].float() for v in anchors])  # [n, 512] CLAP-audio
    Y = torch.stack([text_db[v].float() for v in anchors])   # [n, 512] CLIP-text
    X = F.normalize(X, dim=-1)
    Y = F.normalize(Y, dim=-1)
    n, d = X.shape
    print(f"  Audio (X): {X.shape}, Text (Y): {Y.shape}")

    # ---- Baseline alignment before projection ----
    pos_cos_before = float((X * Y).sum(dim=1).mean().item())
    print(f"\n[1] BASELINE (before projection): positive-pair cosine = "
          f"{pos_cos_before:.4f}")

    # ---- Method 1: Ridge regression (Wang 2019) ----
    print("\n[2] RIDGE REGRESSION (Wang et al. 2019)")
    Xt, Yt = X.T, Y.T  # both [d, n]
    if args.use_bias:
        one = torch.ones(1, n)
        Xa = torch.cat([Xt, one], dim=0)         # [d+1, n]
        Wa = ridge_fit(Xa, Yt, args.ridge_lam)   # [d, d+1]
        W_ridge = Wa[:, :d]                      # [d, d]
        b_ridge = Wa[:, d].contiguous()          # [d]
    else:
        W_ridge = ridge_fit(Xt, Yt, args.ridge_lam)  # [d, d]
        b_ridge = None
    if b_ridge is not None:
        Y_pred_ridge = X @ W_ridge.T + b_ridge.view(1, -1)   # [n, d]
    else:
        Y_pred_ridge = X @ W_ridge.T
    proj_ridge = F.normalize(Y_pred_ridge, dim=-1)
    pos_cos_ridge = float((proj_ridge * Y).sum(dim=1).mean().item())
    print(f"  Ridge (lam={args.ridge_lam}, bias={args.use_bias}): "
          f"projected positive-pair cosine = {pos_cos_ridge:.4f} "
          f"(was {pos_cos_before:.4f})")

    # ---- Method 2: Mean-centered Procrustes (Zhang & Saligrama 2016) ----
    print("\n[3] ORTHOGONAL PROCRUSTES (Zhang & Saligrama 2016)")
    mean_X = X.mean(dim=0, keepdim=True)
    mean_Y = Y.mean(dim=0, keepdim=True)
    Xc = X - mean_X
    Yc = Y - mean_Y
    W_proc = procrustes(Xc.T, Yc.T)              # [d, d]
    b_proc = (mean_Y - mean_X @ W_proc.T).squeeze(0)  # [d]
    Y_pred_proc = X @ W_proc.T + b_proc.view(1, -1)
    proj_proc = F.normalize(Y_pred_proc, dim=-1)
    pos_cos_proc = float((proj_proc * Y).sum(dim=1).mean().item())
    print(f"  Procrustes: projected positive-pair cosine = {pos_cos_proc:.4f}")

    # Also report residual distances (recall: L2 on centered procrustes)
    resid_proc = ((Yc - Xc @ W_proc.T).norm(dim=1).mean().item())
    print(f"  Procrustes residual ||Y-X W|| mean = {resid_proc:.4f}")

    # ---- Method 3: CCA reference (Rasiwasia 2010) ----
    print("\n[4] CCA REFERENCE (Rasiwasia et al. 2010)")
    # Canonical correlations = singular values of the cross-covariance
    C_xy = (Xc.T @ Yc) / (n - 1)                 # [d, d]
    # standardized cross-covariance:
    sxx = (Xc * Xc).mean(dim=0).sqrt() + 1e-8
    syy = (Yc * Yc).mean(dim=0).sqrt() + 1e-8
    C_std = C_xy / (sxx.unsqueeze(1) * syy.unsqueeze(0) + 1e-8)
    S = torch.linalg.svdvals(C_std)
    print(f"  Top-3 canonical correlation coefficients: "
          f"{np.round(S[:3].cpu().numpy(), 4).tolist()}")
    print("  (CCA reported as a reference baseline; retrieval uses the")
    print("   Procrustes/ridge maps, which output d=512 vectors.)")

    # ---- Select best linear map: prefer Procrustes (distance-preserving) ----
    print("\n[5] SELECTING MAP")
    best = "procrustes" if pos_cos_proc >= pos_cos_ridge else "ridge"
    W = W_proc if best == "procrustes" else W_ridge
    if best == "procrustes":
        b = b_proc
    elif b_ridge is not None:
        b = b_ridge
    else:
        b = torch.zeros(d)
    mean_audio = mean_X.squeeze(0)
    print(f"  Chosen: {best} (projected train cosine = "
          f"{max(pos_cos_proc, pos_cos_ridge):.4f})")

    # ---- Save ----
    torch.save({
        "W": W.detach().cpu(),
        "b": b.detach().cpu(),
        "mean_audio": mean_audio.detach().cpu(),
        "method": best,
        "ridge_lam": args.ridge_lam,
        "use_bias": args.use_bias,
        "n_train_anchors": int(n),
        "d": int(d),
        "baseline_cosine": pos_cos_before,
        "ridge_cosine": pos_cos_ridge,
        "procrustes_cosine": pos_cos_proc,
        "procrustes_residual": resid_proc,
        "cca_top3": np.round(S[:3].cpu().numpy(), 4).tolist(),
    }, MODEL_PATH)
    print(f"[SAVE] Projection saved: {MODEL_PATH}")

    report = {
        "framework": ["Wang 2019", "Zhang & Saligrama 2016", "Grave 2019",
                      "Rasiwasia 2010"],
        "target": "CLIP-text(description)",
        "n_train_anchors": int(n),
        "d": int(d),
        "ridge_lam": args.ridge_lam,
        "use_bias": args.use_bias,
        "baseline_positive_cosine": pos_cos_before,
        "ridge_projected_cosine": pos_cos_ridge,
        "procrustes_projected_cosine": pos_cos_proc,
        "procrustes_residual": resid_proc,
        "cca_top3_canonical_corr": np.round(S[:3].cpu().numpy(), 4).tolist(),
        "chosen_method": best,
        "model_path": MODEL_PATH,
    }
    out_path = os.path.join(OUTPUT_DIR, "projection_fit.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[DONE] Fit report: {out_path}")


if __name__ == "__main__":
    main()
