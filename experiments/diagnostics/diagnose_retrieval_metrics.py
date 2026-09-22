#!/usr/bin/env python3
"""
Comprehensive Retrieval Metrics & Evaluation Protocol Audit (AEMS audio-text)
=============================================================================
Frameworks:
  - Chen et al. (2021), "A Hitchhiker's Guide to Cross-Modal Retrieval"
    (correct evaluation protocol: candidate set size, MRR, NDCG, CMC)
  - Wu et al. (2023), "CLAP" (zero-shot retrieval analysis on AudioCaps/Clotho)
  - Hoiem et al. (2012) rank/miss breakdown

Beyond the single R@K number, this computes a full battery of information
retrieval metrics to give a more faithful picture of audio-text retrieval
quality (which R@K alone can under-report at tiny candidate sets).

Run:
  export PYTHONPATH="${PYTHONPATH}:$(pwd)"
  python experiments/diagnostics/diagnose_retrieval_metrics.py [--text-variant fused]
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from src.config import (
    AEMS_MANIFEST_PATH,
    AEMS_CLAP_AUDIO_EMBEDDINGS_PATH,
    AEMS_TEXT_EMBEDDINGS_DESC_PATH_TEMPLATE,
    AEMS_TEXT_EMBEDDINGS_TRANS_PATH_TEMPLATE,
    AEMS_TEXT_EMBEDDINGS_FUSED_PATH_TEMPLATE,
    AEMS_VID_EMBEDDINGS_PATH,
    set_seeds,
)
from src.data.metadata import load_metadata, filter_by_split
from src.evaluation.evaluate_retrieval import evaluate_retrieval

OUTPUT_DIR = "outputs/diagnostics"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TEXT_PATHS = {
    "description": AEMS_TEXT_EMBEDDINGS_DESC_PATH_TEMPLATE,
    "transcript": AEMS_TEXT_EMBEDDINGS_TRANS_PATH_TEMPLATE,
    "fused": AEMS_TEXT_EMBEDDINGS_FUSED_PATH_TEMPLATE,
}


def mrr(ranks):
    """Mean Reciprocal Rank."""
    ranks = np.asarray(ranks, dtype=np.float64) + 1.0
    return float((1.0 / ranks).mean())


def mean_median_rank(ranks):
    ranks = np.asarray(ranks, dtype=np.float64)
    return float(ranks.mean()), float(np.median(ranks))


def ndcg_at_k(ranks_list, k):
    """Binary relevance NDCG@K. Correct video = relevance 1, else 0."""
    results = []
    for ranks in ranks_list:
        # ranks: list of gallery indices in descending-similarity order
        dcgs = []
        for kk in range(1, k + 1):
            topk = ranks[:kk]
            rel = 1.0 if 0 in topk else 0.0  # position 0 = correct video
            if rel == 0:
                dcgs.append(0.0)
            else:
                dcg = 1.0 / np.log2(kk + 1)
                idcg = 1.0  # ideal: correct at rank 1
                dcgs.append(dcg / idcg)
        results.append(dcgs[-1] if dcgs else 0.0)
    return results


def ap_at_k(ranks_list, k):
    """Average precision @ K (graded: correct at rank1 is best)."""
    aps = []
    for ranks in ranks_list:
        topk = ranks[:k]
        # correct video is gallery position 0
        pos = -1
        for pi, gi in enumerate(topk):
            if gi == 0:
                pos = pi
                break
        if pos < 0:
            aps.append(0.0)
        else:
            aps.append(1.0 / (pos + 1))
    return float(np.mean(aps))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text-variant", default="fused",
                        choices=["description", "transcript", "fused"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    set_seeds(args.seed)

    print("=" * 72)
    print("COMPREHENSIVE RETRIEVAL METRICS (AEMS audio-text)")
    print("  Frameworks: Hitchhiker's Guide / Chen 2021; CLAP / Wu 2023")
    print("=" * 72)

    records = filter_by_split(load_metadata(AEMS_MANIFEST_PATH), split="test")
    test_ids = set(rec["video_id"] for rec in records)
    rec_by_id = {r["video_id"]: r for r in records}

    audio_db = torch.load(AEMS_CLAP_AUDIO_EMBEDDINGS_PATH, weights_only=False)
    text_db = torch.load(TEXT_PATHS[args.text_variant].format(split="test"),
                         weights_only=False)
    common = sorted(set(audio_db.keys()) & set(text_db.keys()) & test_ids)
    n = len(common)
    print(f"[DATA] Test videos (candidate gallery size): {n}")

    audio_mat = F.normalize(torch.stack([audio_db[v].float() for v in common]), dim=-1)
    text_mat = F.normalize(torch.stack([text_db[v].float() for v in common]), dim=-1)

    # Similarity in both directions:
    sim_t2a = text_mat @ audio_mat.T   # text query -> audio gallery
    sim_a2t = audio_mat @ text_mat.T   # audio query -> text gallery

    systems = {
        "text2audio": sim_t2a,
        "audio2text": sim_a2t,
    }

    report = {"framework": ["Hitchhiker's Guide / Chen 2021",
                            "CLAP / Wu 2023", "Hoiem 2012"],
              "text_variant": args.text_variant, "n_videos": n}

    for direction, sim in systems.items():
        print("\n" + "-" * 72)
        print(f"SYSEM: {direction}  (query -> {direction.split('2')[-1]} gallery)")
        gt = torch.arange(n)

        # Loved/ranked gallery indices for each query
        ranked = torch.argsort(sim, descending=True)  # [n, n] gallery idx
        ranks_of_gt = (ranked == gt.view(-1, 1)).nonzero()[:, 1].tolist()

        # Standard R@K
        rk = evaluate_retrieval(sim, list(gt.tolist()), common, ks=[1, 5, 10, 50])
        print(f"  R@1={rk['R@1']:.4f} R@5={rk['R@5']:.4f} R@10={rk['R@10']:.4f} "
              f"R@50={rk['R@50']:.4f}")
        mq = mrr(ranks_of_gt)
        mean_r, median_r = mean_median_rank(ranks_of_gt)
        print(f"  MRR={mq:.4f}  MeanRank={mean_r:.1f}  MedianRank={median_r:.0f}")

        # NDCG and AP (compute on rank-lists)
        ndcg1 = np.mean(ndcg_at_k(ranked.tolist(), 1))
        ndcg3 = np.mean(ndcg_at_k(ranked.tolist(), 3))
        ndcg5 = np.mean(ndcg_at_k(ranked.tolist(), 5))
        ap3 = ap_at_k(ranked.tolist(), 3)
        ap5 = ap_at_k(ranked.tolist(), 5)
        print(f"  NDCG@1={ndcg1:.4f} NDCG@3={ndcg3:.4f} NDCG@5={ndcg5:.4f}")
        print(f"  AP@3={ap3:.4f} AP@5={ap5:.4f}")

        # CMC curve (R@K for K=1..max)
        cmc = []
        for kk in [1, 2, 5, 10, 20, 50, 100, n]:
            kk_eff = min(kk, n)
            hit = 0
            for q in range(n):
                if q in ranked[q, :kk_eff]:
                    hit += 1
            cmc.append({"k": kk, "recall": hit / n})
        cmc_str = " ".join(f"R@{c['k']}={c['recall']:.3f}" for c in cmc[:8])
        print(f"  CMC: {cmc_str}")

        # Category-stratified R@1
        cat_r1 = {}
        for ccat in sorted(set(r["content_fine_category"] for r in records)):
            idx = [i for i in range(n)
                   if rec_by_id.get(common[i], {}).get("content_fine_category") == ccat]
            if not idx:
                continue
            hits = sum(1 for q in idx if q == ranked[q, 0].item())
            cat_r1[ccat] = {"n": len(idx), "r1": hits / len(idx)}
        print(f"  Category-stratified R@1 (audio->text direction):")
        for c, d in cat_r1.items():
            print(f"    {c:<28} n={d['n']:>4}  R@1={d['r1']:.4f}")

        # ---- Protocol audit (Hitchhiker's Guide) ----
        print(f"\n  [PROTOCOL AUDIT]")
        print(f"    Candidate gallery size: {n} (fixed, all shared test videos)")
        print(f"    Query count: {n} (one per video)")
        print(f"    Direction: {direction}")
        # random-chance R@1 for reference
        chance_r1 = 1.0 / n
        ratio = rk["R@1"] / chance_r1 if chance_r1 > 0 else float("inf")
        print(f"    Random-chance R@1: {chance_r1:.6f}")
        print(f"    Observed R@1 / chance: {ratio:.1f}x")
        print(f"    => R@1={rk['R@1']:.4f} at candidate set {n}. Compare to the")
        print(f"       CLAP paper's AudioCaps/Clotho (0-20% typical for 1k")
        print(f"       scale retrieval) to judge if this is unusually low.")

        report[direction] = {
            "R@1": rk["R@1"], "R@5": rk["R@5"], "R@10": rk["R@10"],
            "R@50": rk["R@50"], "MRR": mq,
            "mean_rank": mean_r, "median_rank": median_r,
            "NDCG@1": ndcg1, "NDCG@3": ndcg3, "NDCG@5": ndcg5,
            "AP@3": ap3, "AP@5": ap5, "CMC": cmc,
            "category_stratified": cat_r1,
            "chance_r1": chance_r1, "r1_over_chance": ratio,
        }

    out_path = os.path.join(OUTPUT_DIR, "retrieval_metrics_report.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[DONE] Report saved: {out_path}")


if __name__ == "__main__":
    main()
