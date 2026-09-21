#!/usr/bin/env python3
"""
Hubness Diagnostics for AEMS Audio-Text Retrieval
=================================================
Framework: Radovanović et al. (2010), "Hubs in Space: Popular Nearest Neighbors
in High-Dimensional Data"

Hubness = some points become the nearest neighbor of a disproportionately large
number of queries. This causes certain (often "generic" / silent / background)
audio embeddings to be retrieved incorrectly for many different audio queries,
which directly lowers recall.

Run:
  export PYTHONPATH="${PYTHONPATH}:$(pwd)"
  python experiments/diagnostics/diagnose_hubness.py [--text-variant fused]
"""

import argparse
import json
import os
from collections import Counter

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


def skewness(x):
    """Skewness of a distribution. >0 => heavy right tail (hubness signature)."""
    x = np.asarray(x, dtype=np.float64)
    mu = x.mean()
    std = x.std()
    if std == 0:
        return 0.0
    n = len(x)
    return float((((x - mu) / std) ** 3).mean())


def local_scaling(emb, k=5):
    """Local scaling (Zelnik-Manor & Perona 2004): normalize each embedding by
    its distance to its k-th nearest neighbor. Used for hubness reduction."""
    n = emb.shape[0]
    sim = emb @ emb.T
    # k-th nearest neighbor similarity (excluding self)
    kk = k + 1
    knn_vals, _ = torch.topk(sim, kk, dim=1)
    sigma = knn_vals[:, k]  # similarity to k-th neighbor
    sigma = sigma.clamp(min=1e-3).unsqueeze(1)
    scaled = emb / sigma
    return F.normalize(scaled, dim=-1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text-variant", default="fused",
                        choices=["description", "transcript", "fused"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--topk", type=int, default=5)
    args = parser.parse_args()
    set_seeds(args.seed)

    print("=" * 72)
    print("HUBNESS DIAGNOSTICS (AEMS audio-text)")
    print("  Framework: Radovanović et al. 2010")
    print("=" * 72)

    records = filter_by_split(load_metadata(AEMS_MANIFEST_PATH), split="test")
    test_ids = set(rec["video_id"] for rec in records)

    audio_db = torch.load(AEMS_AUDIO_EMBEDDINGS_PATH, weights_only=False)
    text_db = torch.load(TEXT_PATHS[args.text_variant].format(split="test"),
                         weights_only=False)

    common = sorted(set(audio_db.keys()) & set(text_db.keys()) & test_ids)
    print(f"[DATA] Common test videos: {len(common)}")

    # Use audio as the gallery and text as queries (text->audio retrieval),
    # which is the direction that maps queries into the audio embedding space.
    audio_mat = F.normalize(torch.stack([audio_db[v].float() for v in common]), dim=-1)
    text_mat = F.normalize(torch.stack([text_db[v].float() for v in common]), dim=-1)
    n = len(common)
    print(f"  gallery (audio): {audio_mat.shape}, queries (text): {text_mat.shape}")

    print("\n[1] HUBNESS: K-OCCURRENCE COUNTS (text -> audio)")
    K = args.topk
    sim = text_mat @ audio_mat.T  # [n queries, n gallery]
    # For each query, rank audio gallery and record top-K neighbors
    topk_indices = torch.argsort(sim, descending=True)[:, :K]
    occ_counter = Counter()
    for row in topk_indices:
        for idx in row.tolist():
            occ_counter[idx] += 1
    occ = np.array([occ_counter.get(i, 0) for i in range(n)])
    occ_sorted = np.sort(occ)[::-1]
    mean_occ = occ.mean()
    print(f"  Mean k-occurrence: {mean_occ:.2f} (theoretical {K})")
    print(f"  Max k-occurrence:  {occ.max()}  (a point retrieved this often "
          f"for many different queries is a HUB)")
    print(f"  Skewness of k-occurrence distribution: {skewness(occ):.4f}  "
          f"(>0 => hubness present)")

    # Identify hubs
    hub_threshold = max(10, int(n * 0.10 * mean_occ))
    n_k = np.sum(occ > hub_threshold)
    top_hubs_idx = occ_sorted[:min(10, n)]
    print(f"  Hubs (k-occ > {hub_threshold}): {n_k} points "
          f"({100*n_k/n:.2f}% of gallery)")
    top_occ = occ_counter.most_common(10)
    print(f"  Top hubs (video_id, k-occ):")
    hub_list = []
    for vid_idx, cnt in top_occ:
        vid = common[vid_idx]
        hub_list.append({"video_id": vid, "k_occurrence": cnt})
        print(f"    video {vid:<8}: {cnt}")
    report = {"framework": "Radovanović et al. 2010 (hubness)",
              "text_variant": args.text_variant, "n_videos": n,
              "topk": K, "mean_k_occurrence": float(mean_occ),
              "max_k_occurrence": int(occ.max()),
              "skewness": skewness(occ), "hub_threshold": int(hub_threshold),
              "n_hubs": int(n_k), "top_hubs": hub_list}

    print("\n[2] HUBNESS SKEWNESS vs K")
    skew_dict = {}
    for kk in [1, 2, 5, 10, 20]:
        idx = torch.argsort(sim, descending=True)[:, :kk]
        c = Counter(r.item() for row in idx for r in row)
        o = np.array([c.get(i, 0) for i in range(n)])
        s = skewness(o)
        skew_dict[str(kk)] = s
        print(f"  K={kk:>2}: skewness={s:.4f}")
    report["skewness_vs_k"] = skew_dict

    print("\n[3] LOCAL SCALING HUBNESS REDUCTION DIAGNOSTIC")
    # Does local scaling reduce hubness and improve text->audio R@1?
    gt = torch.arange(n)
    r1_before = (torch.argsort(sim, descending=True)[:, 0] == gt).float().mean().item()
    audio_ls = local_scaling(audio_mat, k=5)
    text_ls = local_scaling(text_mat, k=5)
    sim_ls = text_ls @ audio_ls.T
    r1_after = (torch.argsort(sim_ls, descending=True)[:, 0] == gt).float().mean().item()
    # hubness after
    idx_ls = torch.argsort(sim_ls, descending=True)[:, :K]
    c = Counter(r.item() for row in idx_ls for r in row)
    o_ls = np.array([c.get(i, 0) for i in range(n)])
    skew_after = skewness(o_ls)
    print(f"  Text->audio self R@1:  before={r1_before:.4f}  "
          f"after-local-scaling={r1_after:.4f}")
    print(f"  Hubness skewness:      before={skewness(occ):.4f}  "
          f"after-local-scaling={skew_after:.4f}")
    report["local_scaling"] = {
        "r1_before": r1_before, "r1_after": r1_after,
        "skew_before": skewness(occ), "skew_after": skew_after,
    }

    print("\n[4] OVERLAP BETWEEN QUERY GROUND-TRUTH AND HUBS")
    # For queries whose GT is a hub, how often does a hub crowd out correct answer?
    hub_set = set(i for i, v in occ_counter.items() if v > hub_threshold)
    sim_local = sim.clone()
    # For each query, does the top1 ever come from a different hub than GT?
    top1_idx = torch.argsort(sim_local, descending=True)[:, 0]
    n_gt_is_hub = sum(1 for i in range(n) if i in hub_set)
    n_top1_hub = sum(1 for i in range(n) if top1_idx[i].item() in hub_set)
    n_both = sum(1 for i in range(n) if top1_idx[i].item() in hub_set
                 and i in hub_set)
    print(f"  Queries whose GT video is a hub: {n_gt_is_hub} ({100*n_gt_is_hub/n:.2f}%)")
    print(f"  Queries whose top-1 result is a hub: {n_top1_hub} ({100*n_top1_hub/n:.2f}%)")
    report["hub_overlap"] = {"gt_is_hub": int(n_gt_is_hub),
                             "top1_is_hub": int(n_top1_hub),
                             "n": n}

    out_path = os.path.join(OUTPUT_DIR, "hubness_report.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[DONE] Report saved: {out_path}")


if __name__ == "__main__":
    main()
