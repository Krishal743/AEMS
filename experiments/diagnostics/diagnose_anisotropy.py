#!/usr/bin/env python3
"""
Anisotropy Diagnostics for AEMS Embeddings
===========================================
Framework: Ethayarajh (2019), "How Contextual are Contextualized Word
Representations? Comparing the Geometry of BERT, ELMo, and GPT-2 Embeddings"

Anisotropy measures how far the embedding distribution deviates from a uniform
sphere. High anisotropy (mean pairwise cosine similarity near 1, low effective
dimensionality) means all embeddings occupy a narrow cone, which destroys
retrieval discrimination. Whitening / mean-centering + re-normalization can
restore spread.

Run:
  export PYTHONPATH="${PYTHONPATH}:$(pwd)"
  python scripts/diagnostic/diagnose_anisotropy.py [--text-variant fused]
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from src.config import (
    AEMS_MANIFEST_PATH,
    AEMS_AUDIO_EMBEDDINGS_PATH,
    AEMS_VID_EMBEDDINGS_PATH,
    AEMS_TEXT_EMBEDDINGS_DESC_PATH_TEMPLATE,
    AEMS_TEXT_EMBEDDINGS_TRANS_PATH_TEMPLATE,
    AEMS_TEXT_EMBEDDINGS_FUSED_PATH_TEMPLATE,
    set_seeds,
)
from src.data.metadata import load_metadata, filter_by_split

OUTPUT_DIR = "outputs/diagnostics"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TEXT_PATHS = {
    "description": AEMS_TEXT_EMBEDDINGS_DESC_PATH_TEMPLATE,
    "transcript": AEMS_TEXT_EMBEDDINGS_TRANS_PATH_TEMPLATE,
    "fused": AEMS_TEXT_EMBEDDINGS_FUSED_PATH_TEMPLATE,
}


def average_cosine(emb, sample_size=1500, seed=42):
    """Mean pairwise cosine similarity (anisotropy metric, Ethayarajh 2019).
    For unit vectors this equals the mean dot product. Computed on a random
    subsample to avoid O(N^2) memory."""
    rng = np.random.RandomState(seed)
    n = emb.shape[0]
    m = min(n, sample_size)
    idx = rng.choice(n, size=m, replace=False)
    sub = emb[idx]
    sims = sub @ sub.T
    # exclude diagonal
    mask = ~torch.eye(m, dtype=torch.bool)
    vals = sims[mask].detach().cpu().numpy()
    return float(vals.mean()), float(vals.std())


def effective_dimensionality(emb):
    """Effective dimensionality from eigenvalue distribution of covariance.
    H(eig) = exp(-sum p_i log p_i) where p_i = eig_i / sum(eig)."""
    cent = emb - emb.mean(dim=0, keepdim=True)
    cov = (cent.T @ cent) / cent.shape[0]
    eig = torch.linalg.eigvalsh(cov).clamp(min=0)
    total = eig.sum()
    if total <= 1e-12:
        return 1.0
    p = eig / total
    p = p[p > 0]
    entropy = -(p * p.log()).sum()
    return float(torch.exp(entropy).item())


def pca_variance_explained(emb, topk=10):
    cent = emb - emb.mean(dim=0, keepdim=True)
    cov = (cent.T @ cent) / cent.shape[0]
    eig = torch.linalg.eigvalsh(cov).clamp(min=0)
    total = eig.sum()
    if total <= 1e-12:
        return [0.0] * topk
    eig_desc = torch.sort(eig, descending=True).values
    cumulative = torch.cumsum(eig_desc, dim=0) / total
    return cumulative[:topk].cpu().numpy().tolist()


def zca_whiten(emb, eps=1e-4):
    """ZCA whitening (mean removal + decorrelation + renormalize)."""
    cent = emb - emb.mean(dim=0, keepdim=True)
    cov = (cent.T @ cent) / cent.shape[0]
    eigval, eigvec = torch.linalg.eigh(cov)
    eigval = eigval.clamp(min=0)
    # ZCA: S = V D^-1/2 V^T
    d_inv_sqrt = torch.diag(1.0 / (eigval + eps).sqrt())
    whitening = eigvec @ d_inv_sqrt @ eigvec.T
    whitened = cent @ whitening
    return F.normalize(whitened, dim=-1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text-variant", default="fused",
                        choices=["description", "transcript", "fused"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    set_seeds(args.seed)

    print("=" * 72)
    print("ANISOTROPY DIAGNOSTICS (AEMS)")
    print("  Framework: Ethayarajh et al. 2019")
    print("=" * 72)

    records = filter_by_split(load_metadata(AEMS_MANIFEST_PATH), split="test")
    test_ids = set(rec["video_id"] for rec in records)

    audio_db = torch.load(AEMS_AUDIO_EMBEDDINGS_PATH, weights_only=False)
    video_db = torch.load(AEMS_VID_EMBEDDINGS_PATH, weights_only=False)
    text_db = torch.load(TEXT_PATHS[args.text_variant].format(split="test"),
                         weights_only=False)

    common = sorted(
        set(audio_db.keys()) & set(video_db.keys()) & set(text_db.keys()) & test_ids
    )
    print(f"[DATA] Common test videos: {len(common)}")

    audio = F.normalize(torch.stack([audio_db[v].float() for v in common]), dim=-1)
    video = F.normalize(torch.stack([video_db[v].float() for v in common]), dim=-1)
    text = F.normalize(torch.stack([text_db[v].float() for v in common]), dim=-1)
    print(f"  audio: {audio.shape}, video: {video.shape}, text: {text.shape}")

    report = {"framework": "Ethayarajh et al. 2019 (anisotropy)",
              "text_variant": args.text_variant, "n_videos": len(common)}

    # ---------- 1. Average cosine similarity (anisotropy) ------------------
    print("\n[1] AVERAGE PAIRWISE COSINE SIMILARITY (ANISOTROPY)")
    print("    High value (~>0.5) => narrow cone => poor discrimination.")
    anisotropy = {}
    for name, mat in [("audio", audio), ("video", video), ("text", text)]:
        m, sd = average_cosine(mat, seed=args.seed)
        anisotropy[name] = {"mean": m, "std": sd}
        print(f"  {name:<6}: mean_cos={m:.4f} std={sd:.4f}")
    report["average_cosine"] = anisotropy

    # ---------- 2. Effective dimensionality --------------------------------
    print("\n[2] EFFECTIVE DIMENSIONALITY (from eigenvalue distribution)")
    print("    If << 512, the embeddings occupy only a low-rank subspace.")
    eff_dim = {}
    for name, mat in [("audio", audio), ("video", video), ("text", text)]:
        e = effective_dimensionality(mat)
        eff_dim[name] = e
        print(f"  {name:<6}: effective_dim={e:.1f} / 512")
    report["effective_dimensionality"] = eff_dim

    # ---------- 3. PCA variance explained by top components ----------------
    print("\n[3] PCA VARIANCE EXPLAINED (top 10 components)")
    pca = {}
    for name, mat in [("audio", audio), ("video", video), ("text", text)]:
        cum = pca_variance_explained(mat, topk=10)
        pca[name] = {"top10_cumulative": cum,
                     "top1": cum[0], "top5": cum[4], "top10": cum[9]}
        print(f"  {name:<6}: top1={cum[0]:.3f} top5={cum[4]:.3f} "
              f"top10={cum[9]:.3f} (cumulative explained variance)")
    report["pca_variance_explained"] = pca

    # ---------- 4. Whitening diagnostic ------------------------------------
    print("\n[4] WHITENING DIAGNOSTIC (would decorrelation restore spread?)")
    whitened = {}
    for name, mat in [("audio", audio), ("video", video), ("text", text)]:
        w = zca_whiten(mat, eps=1e-4)
        m_before = anisotropy[name]["mean"]
        m_after, _ = average_cosine(w, seed=args.seed)
        e_before = eff_dim[name]
        e_after = effective_dimensionality(w)
        whitened[name] = {"cos_before": m_before, "cos_after": m_after,
                          "eff_dim_before": e_before, "eff_dim_after": e_after}
        print(f"  {name:<6}: cos {m_before:.4f} -> {m_after:.4f} | "
              f"eff_dim {e_before:.1f} -> {e_after:.1f}")
    report["whitening_diagnostic"] = whitened

    out_path = os.path.join(OUTPUT_DIR, "anisotropy_report.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[DONE] Report saved: {out_path}")


if __name__ == "__main__":
    main()
