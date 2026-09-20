#!/usr/bin/env python3
"""
Data Quality & Failure Categorization Diagnostics for AEMS Audio-Text
=====================================================================
Frameworks:
  - Li et al. (2022), "BLIP" (data filtering / noisy-pair detection)
  - Faghri et al. (2018), "VSE++" (hard negative mining / embedding separation)
  - Hoiem et al. (2012), "Diagnosing Error in Object Detectors"
    (error categorization: poor embedding vs missing semantics vs evaluation)

This script answers:
  1. Are the audio-text positive pairs actually aligned? (BLIP noisy-pair)
  2. Are negatives too easy / too hard for training? (VSE++)
  3. When retrieval fails, is it a poor-embedding problem, a missing-semantics
     problem, or an evaluation-protocol problem? (Hoiem error categories)

Run:
  export PYTHONPATH="${PYTHONPATH}:$(pwd)"
  python scripts/diagnostic/diagnose_data_quality.py [--text-variant fused]
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text-variant", default="fused",
                        choices=["description", "transcript", "fused"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--noise-quantile", type=float, default=0.10,
                        help="Bottom fraction considered 'noisy pairs'")
    args = parser.parse_args()
    set_seeds(args.seed)

    print("=" * 72)
    print("DATA QUALITY & FAILURE CATEGORIZATION (AEMS audio-text)")
    print("  Frameworks: BLIP / Li 2022; VSE++ / Faghri 2018; Hoiem 2012")
    print("=" * 72)

    records = filter_by_split(load_metadata(AEMS_MANIFEST_PATH), split="test")
    test_ids = set(rec["video_id"] for rec in records)
    rec_by_id = {r["video_id"]: r for r in records}

    audio_db = torch.load(AEMS_AUDIO_EMBEDDINGS_PATH, weights_only=False)
    text_db = torch.load(TEXT_PATHS[args.text_variant].format(split="test"),
                         weights_only=False)

    common = sorted(set(audio_db.keys()) & set(text_db.keys()) & test_ids)
    print(f"[DATA] Common test videos: {len(common)}")

    audio_mat = F.normalize(torch.stack([audio_db[v].float() for v in common]), dim=-1)
    text_mat = F.normalize(torch.stack([text_db[v].float() for v in common]), dim=-1)
    n = len(common)

    sim = audio_mat @ text_mat.T  # [n audio, n text]; row i vs all texts
    diag = torch.diag(sim).detach().cpu()

    # =====================
    # 1. NOISY PAIR DETECTION (BLIP)
    # =====================
    print("\n[1] POSITIVE-PAIR ALIGNMENT & NOISY PAIR DETECTION (BLIP / Li 2022)")
    print(f"  Positive-pair audio-text cosine: mean={diag.mean():.4f} "
          f"std={diag.std():.4f} median={diag.median():.4f}")
    q = args.noise_quantile
    threshold = float(np.quantile(diag.numpy(), q))
    print(f"  Noisy-pair threshold (bottom {100*q:.0f}%): cos={threshold:.4f}")

    noisy_idx = (diag <= threshold).nonzero().flatten().tolist()
    print(f"  Detected {len(noisy_idx)} potentially NOISY pairs (lowest "
          f"{100*q:.0f}% alignment):")
    noisy_examples = []
    for i in noisy_idx[:10]:
        vid = common[i]
        rec = rec_by_id.get(vid, {})
        desc = (rec.get("text_description") or "")[:80]
        noisy_examples.append({
            "video_id": vid,
            "positive_cosine": float(diag[i]),
            "max_neg_cosine": float(sim[i].max().item()),
            "category": rec.get("content_fine_category"),
            "description": desc,
        })
        print(f"    video {vid:<8} pos_cos={diag[i]:.3f} "
              f"max_neg={sim[i].max():.3f}  | {desc}")
    report = {"framework": "BLIP/Li 2022 (noisy-pair) etc.",
              "text_variant": args.text_variant, "n_videos": n,
              "noise_quantile": q, "noise_threshold": threshold,
              "n_noisy_pairs": len(noisy_idx),
              "noisy_examples": noisy_examples}

    # =====================
    # 2. HARD vs EASY NEGATIVES (VSE++)
    # =====================
    print("\n[2] NEGATIVE DIFFICULTY DISTRIBUTION (VSE++ / Faghri 2018)")
    # Exclude diagonal from negatives
    mask = ~torch.eye(n, dtype=torch.bool)
    neg_sims = sim[mask].view(n, n - 1)
    max_neg = neg_sims.max(dim=1).values
    mean_neg = neg_sims.mean(dim=1)
    min_neg = neg_sims.min(dim=1).values
    print(f"  Per-positive max negative:  mean={max_neg.mean():.4f} "
          f"median={max_neg.median():.4f}")
    print(f"  Per-positive mean negative: mean={mean_neg.mean():.4f}")
    print(f"  Per-positive min negative:  mean={min_neg.mean():.4f}")
    margin_pos_maxneg = diag - max_neg
    print(f"  Margin (pos - max_neg): mean={margin_pos_maxneg.mean():.4f}, "
          f"{100*(margin_pos_maxneg>=0).float().mean():.1f}% positive pairs beat "
          f"their hardest negative")
    print(f"  => If margin is low/negative, negatives are too hard / positives "
          f"not discriminative (Faghri 2018).")

    # Hard-negative ratio: fraction of max-neg that exceed the positive
    hard_ratio = float((max_neg > diag).float().mean().item())
    print(f"  Fraction where hardest negative OUTSCORES the positive: "
          f"{100*hard_ratio:.1f}% (high => severe alignment problem)")
    report["vsepp_negatives"] = {
        "max_neg_mean": float(max_neg.mean()),
        "mean_neg_mean": float(mean_neg.mean()),
        "min_neg_mean": float(min_neg.mean()),
        "margin_mean": float(margin_pos_maxneg.mean()),
        "hard_negative_ratio": hard_ratio,
    }

    # =====================
    # 3. FAILURE CATEGORIZATION (Hoiem et al. 2012 adapted)
    # =====================
    print("\n[3] RETRIEVAL FAILURE CATEGORIZATION (Hoiem et al. 2012)")
    # Adapted categories for the audio-text self retrieval task:
    #  A. CORRECT: audio retrieves its own text at rank 1
    #  B. NEAR MISS: rank 2-10 (embedding close, just not top)
    #  C. SEMANTIC CONFUSION: rank>10 but the retrieved text belongs to the
    #     same category (audio semantics exist but not discriminative)
    #  D. MISSING SEMANTICS: rank>10 and retrieved text in a different category
    #     (audio content may not encode the query semantics at all)
    #  E. MISALIGNED / LEAK: audio-text pair appears noise-aligned (below
    #     noise threshold) — data quality issue.
    ranks = torch.argsort(sim, descending=True)
    rank_of_gt = (ranks == torch.arange(n).view(-1, 1)).nonzero()[:, 1].detach().cpu().numpy()

    cats = defaultdict(lambda: 0)
    cat_details = defaultdict(list)
    for i in range(n):
        rank = rank_of_gt[i]
        vid = common[i]
        cat = rec_by_id.get(vid, {}).get("content_fine_category", "?")
        is_noisy = i in set(noisy_idx)
        if rank == 0:
            key = "A_correct"
        elif rank < 10:
            key = "B_near_miss"
        elif cat == rec_by_id.get(common[ranks[i, 0].item()], {}).get(
                "content_fine_category"):
            key = "C_semantic_confusion"
        elif is_noisy:
            key = "E_noisy_pair"
        else:
            key = "D_missing_semantics"
        cats[key] += 1
        cat_details[key].append(vid)

    total = n
    print(f"  A. Correct (rank1):            {cats['A_correct']:>5} "
          f"({100*cats['A_correct']/total:.1f}%)")
    print(f"  B. Near miss (rank 2-10):      {cats['B_near_miss']:>5} "
          f"({100*cats['B_near_miss']/total:.1f}%)")
    print(f"  C. Semantic confusion (r>10, same category): "
          f"{cats['C_semantic_confusion']:>5} "
          f"({100*cats['C_semantic_confusion']/total:.1f}%)")
    print(f"  D. Missing semantics (r>10, diff category): "
          f"{cats['D_missing_semantics']:>5} "
          f"({100*cats['D_missing_semantics']/total:.1f}%)")
    print(f"  E. Noisy pair (aligned but bad):{cats['E_noisy_pair']:>5} "
          f"({100*cats['E_noisy_pair']/total:.1f}%)")

    # Mean rank by category (diagnostic signal)
    print("\n  Mean rank by category:")
    for key in sorted(cats):
        if cats[key] > 0:
            mr = np.mean([rank_of_gt[common.index(v)] for v in cat_details[key]])
            print(f"    {key:<25} n={cats[key]:>5}  mean_rank={mr:.1f}")
    report["failure_categories"] = {k: int(v) for k, v in cats.items()}

    out_path = os.path.join(OUTPUT_DIR, "data_quality_report.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[DONE] Report saved: {out_path}")


from collections import defaultdict

if __name__ == "__main__":
    main()
