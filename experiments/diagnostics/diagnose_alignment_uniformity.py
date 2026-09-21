#!/usr/bin/env python3
"""
Alignment & Uniformity Diagnostics for AEMS Audio-Text Embeddings
=================================================================
Framework:
  - Wang & Isola (2020), "Understanding Contrastive Representation Learning
    through Alignment and Uniformity on the Hypersphere"
  - Zhai et al. (2023), "SigLIP: Sigmoid Loss for Language Image Pre-Training"
    (temperature sensitivity / similarity distribution overlap analysis)

Metrics computed:
  1. ALIGNMENT  (Wang & Isola): mean squared L2 distance between positive pairs.
     Low alignment (high distance) => positive pairs too far apart => low recall.
  2. UNIFORMITY (Wang & Isola): log of the mean Gaussian potential over all
     pairs on the hypersphere (t=2). Low uniformity => embeddings collapsed
     into a narrow region => low recall.
  3. Positive vs negative cosine-similarity distributions (SigLIP):
     histogram + overlap area. High overlap => insufficient separation.
  4. Temperature-sensitivity sweep: R@1 under varying softmax temperature tau
     over CLAP audio-text similarities.

Run:
  export PYTHONPATH="${PYTHONPATH}:$(pwd)"
  python experiments/diagnostics/diagnose_alignment_uniformity.py [--text-variant fused|description|transcript]
"""

import argparse
import gc
import json
import os
import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

from src.config import (
    AEMS_MANIFEST_PATH,
    AEMS_AUDIO_EMBEDDINGS_PATH,
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

# Wang & Isola uniformity temperature (t); t=2 is standard
UNIFORMITY_T = 2.0


def compute_alignment(pos_dists):
    """Alignment = E_{(x,y)~p_pos} ||x - y||^2  (Wang & Isola eq. 2)
    For L2-normalized embeddings, ||x-y||^2 = 2 - 2*cos_sim.
    Lower is better (positive pairs close together)."""
    return float(pos_dists.mean().item())


def compute_uniformity(embeddings, t=UNIFORMITY_T):
    """Uniformity = log E_{pairs} exp(-t * ||x - y||^2)  (Wang & Isola eq. 4)
    For L2-normalized embeddings with t=2, exp(-2*||x-y||^2) = exp(-4*(1-cos)).
    Higher (less negative) is better (more uniform spread)."""
    # Batch compute pairwise squared distances on GPU to avoid O(N^2) memory
    device = embeddings.device
    n = embeddings.shape[0]
    total = 0.0
    B = 512  # blocks to cap memory
    with torch.no_grad():
        for i in range(0, n, B):
            block = embeddings[i : i + B].to(device)
            # ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a.b ; a,b unit norm => = 2 - 2 a.b
            sims = block @ embeddings.to(device).T
            sq_dist = 2.0 - 2.0 * sims
            total += float(torch.exp(-t * sq_dist).sum().item())
    mean_potential = total / (n * n)
    uniformity = np.log(mean_potential)
    return float(uniformity)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text-variant", default="fused",
                        choices=["description", "transcript", "fused"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-samples", type=int, default=None,
                        help="Limit number of queries (for debugging)")
    args = parser.parse_args()
    set_seeds(args.seed)

    print("=" * 72)
    print("ALIGNMENT & UNIFORMITY DIAGNOSTICS (AEMS audio-text)")
    print("  Frameworks: Wang & Isola 2020; SigLIP / Zhai et al. 2023")
    print("=" * 72)

    records = filter_by_split(load_metadata(AEMS_MANIFEST_PATH), split="test")
    print(f"[DATA] Test records: {len(records)}")

    # Build positive pairs: (audio embedding, text embedding) per test video
    print("[LOAD] Loading audio + text embedding DBs...")
    audio_db = torch.load(AEMS_AUDIO_EMBEDDINGS_PATH, weights_only=False)
    text_db = torch.load(TEXT_PATHS[args.text_variant].format(split="test"),
                         weights_only=False)

    test_video_ids = set(rec["video_id"] for rec in records)
    common = sorted(set(audio_db.keys()) & set(text_db.keys()) & test_video_ids)
    print(f"[DATA] Common test videos: {len(common)}")

    if args.num_samples:
        common = common[: args.num_samples]

    if args.num_samples:
        common = common[: args.num_samples]

    print("[BUILD] Building audio & text matrices (float32, CPU)...")
    audio_mat = torch.stack([audio_db[v].float() for v in common])
    text_mat = torch.stack([text_db[v].float() for v in common])
    audio_mat = F.normalize(audio_mat, dim=-1)
    text_mat = F.normalize(text_mat, dim=-1)
    n = len(common)
    print(f"  Audio matrix: {audio_mat.shape}, Text matrix: {text_mat.shape}")

    # ---------- 1. ALIGNMENT ----------
    print("\n[1] ALIGNMENT (Wang & Isola 2020)")
    pos_sims = (audio_mat * text_mat).sum(dim=1)  # diagonal = positive pairs
    pos_cos = pos_sims.detach().cpu()
    pos_dists = torch.sqrt(2.0 - 2.0 * pos_cos)  # ||a-b||^2 = 2-2cos => ||a-b||
    alignment_l2 = float(pos_dists.mean().item())
    alignment_l2sq = compute_alignment(pos_dists**2)

    print(f"  Positive-pair cosine sim: mean={pos_cos.mean():.4f} "
          f"std={pos_cos.std():.4f} median={pos_cos.median():.4f}")
    print(f"  Positive-pair L2 distance: mean={alignment_l2:.4f}")
    print(f"  ALIGNMENT (E||x-y||^2): {alignment_l2sq:.4f}  "
          f"(lower is better, ideal ~0)")
    quantiles = [0.05, 0.25, 0.5, 0.75, 0.95]
    print(f"  Positive cosine quantiles {quantiles}: "
          f"{np.quantile(pos_cos.numpy(), quantiles).round(4)}")

    # ---------- 2. UNIFORMITY ----------
    print("\n[2] UNIFORMITY (Wang & Isola 2020)")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    audio_uni = compute_uniformity(audio_mat.to(device), UNIFORMITY_T)
    text_uni = compute_uniformity(text_mat.to(device), UNIFORMITY_T)
    print(f"  Audio uniformity (t={UNIFORMITY_T}): {audio_uni:.4f}  "
          f"(higher = more uniform = better)")
    print(f"  Text  uniformity (t={UNIFORMITY_T}): {text_uni:.4f}  "
          f"(higher = more uniform = better)")
    print(f"  Note: typical well-trained CLIP spaces have uniformity around "
          f"-2..-4; much lower (< -6) indicates collapse/anisotropy.")

    # ---------- 3. POSITIVE vs NEGATIVE SIMILARITY DISTRIBUTIONS (SigLIP) ----
    print("\n[3] POSITIVE vs NEGATIVE DISTRIBUTION OVERLAP (SigLIP / Zhai 2023)")
    # Build full audio-text similarity matrix on CPU (moderate N=1022)
    sim = audio_mat @ text_mat.T  # [n, n], diag = positives
    neg_sims = sim[~torch.eye(n, dtype=torch.bool).view(n, n)].view(n, n - 1)
    all_neg = neg_sims.flatten().detach().cpu().numpy()
    pos_all = pos_cos.numpy()

    # Overlap area between pos and neg distributions (histogram-based)
    lo = min(all_neg.min(), pos_all.min())
    hi = max(all_neg.max(), pos_all.max())
    bins = np.linspace(lo, hi, 101)
    pos_hist, _ = np.histogram(pos_all, bins=bins, density=True)
    neg_hist, _ = np.histogram(all_neg, bins=bins, density=True)
    overlap = np.minimum(pos_hist, neg_hist).sum() * (bins[1] - bins[0])

    sep = float(pos_all.mean() - all_neg.mean())
    print(f"  Positive cosine: mean={pos_all.mean():.4f} std={pos_all.std():.4f}")
    print(f"  Negative cosine: mean={all_neg.mean():.4f} std={all_neg.std():.4f}")
    print(f"  Mean separation (pos_mean - neg_mean): {sep:.4f}")
    print(f"  Distribution OVERLAP area: {overlap:.4f}  "
          f"(high overlap => poor separability; low => well separated)")
    print(f"  Separate-ability ratio (sep / (pos_std+neg_std)): "
          f"{sep/(pos_all.std()+all_neg.std()):.4f}")

    # Per-video separation: positive sim minus max-negative sim
    max_neg = neg_sims.max(dim=1).values.detach().cpu()
    margin = pos_cos - max_neg
    margin_ge0 = (margin >= 0).float().mean().item()
    print(f"  Margin (pos - max_neg): mean={margin.mean():.4f}, "
          f"frac>=0={margin_ge0:.4f} (fraction where audio correctly ranks "
          f"its own text first)")

    # ---------- 4. TEMPERATURE SENSITIVITY SWEEP (SigLIP) -------------------
    print("\n[4] TEMPERATURE SENSITIVITY SWEEP")
    # For retrieval under softmax temperature tau, scores scaled by tau.
    # Evaluate R@1 vs tau on the CLAP audio-text similarity.
    recall_at_tau = {}
    for tau in [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]:
        scaled = sim * tau
        ranks = torch.argsort(scaled, descending=True)
        gt_pos = torch.arange(n).view(-1, 1).expand(n, n)
        r1 = (ranks[:, 0] == gt_pos[:, 0]).float().mean().item()
        recall_at_tau[str(tau)] = r1
    print(f"  Audio-text self-retrieval R@1 vs temperature tau:")
    for tau, r1 in recall_at_tau.items():
        print(f"    tau={tau:>5}:  R@1={r1:.4f}")

    # ---------- 5. Embedding collapse metrics -------------------------------
    print("\n[5] EMBEDDING SPREAD / COLLAPSE CHECK")
    for name, mat in [("audio", audio_mat), ("text", text_mat)]:
        mean_vec = mat.mean(dim=0)
        row_ss = (mat - mean_vec.unsqueeze(0)).norm(dim=1).mean().item()
        print(f"  {name:>5}: centroid norm={mean_vec.norm():.4f}, "
              f"mean stddev-from-centroid={row_ss:.4f}")

    # ---------- Save report ------------------------------------------------
    report = {
        "framework": ["Wang & Isola 2020 (alignment/uniformity)",
                      "SigLIP/Zhai et al. 2023 (distribution separation & temp)"],
        "text_variant": args.text_variant,
        "n_videos": n,
        "alignment": {
            "positive_cosine_mean": float(pos_cos.mean()),
            "positive_cosine_std": float(pos_cos.std()),
            "positive_l2_mean": alignment_l2,
            "alignment_E_llx_y_ll2": alignment_l2sq,
            "positive_cosine_quantiles": dict(zip(map(str, quantiles),
                                                  np.quantile(pos_cos.numpy(), quantiles).round(4).tolist())),
        },
        "uniformity": {
            "audio": audio_uni,
            "text": text_uni,
            "t": UNIFORMITY_T,
        },
        "distribution_overlap": {
            "pos_mean": float(pos_all.mean()),
            "neg_mean": float(all_neg.mean()),
            "separation": sep,
            "overlap_area": overlap,
            "separability_ratio": float(sep / (pos_all.std() + all_neg.std())),
            "margin_mean": float(margin.mean()),
            "margin_frac_ge0": margin_ge0,
        },
        "temperature_sweep": recall_at_tau,
    }
    out_path = os.path.join(OUTPUT_DIR, "alignment_uniformity_report.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[DONE] Report saved: {out_path}")


if __name__ == "__main__":
    main()
