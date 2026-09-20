#!/usr/bin/env python3
"""
Modality Gap Diagnostics for AEMS Multi-modal Embeddings
=========================================================
Framework: Liang et al. (2022), "Mind the Gap: Understanding the Modality Gap
in Multi-modal Contrastive Representation Learning"

A "modality gap" is the distance between the mean embeddings of different
modalities (e.g., mean audio vs mean text vs mean visual). A large gap can
make cross-modal matching (audio-to-text retrieval) harder, because audio and
text points lie in distant, nearly-disjoint regions of the shared space.

Run:
  export PYTHONPATH="${PYTHONPATH}:$(pwd)"
  python scripts/diagnostic/diagnose_modality_gap.py [--text-variant fused]
"""

import argparse
import gc
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


def centroid(emb_matrix, device="cpu"):
    """Mean embedding of a set of L2-normalized embeddings (may not be unit norm)."""
    return emb_matrix.mean(dim=0).to(device)


def dist_metrics(a, b):
    """Euclidean + cosine distance between two (possibly non-unit) vectors."""
    a = a / a.norm() if a.norm() > 1e-12 else a
    b = b / b.norm() if b.norm() > 1e-12 else b
    eucl = float((a - b).norm().item())
    cos_sim = float((a * b).sum().item())
    return {"euclidean": eucl, "cosine_sim": cos_sim, "cosine_dist": 1.0 - cos_sim}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text-variant", default="fused",
                        choices=["description", "transcript", "fused"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    set_seeds(args.seed)

    print("=" * 72)
    print("MODALITY GAP DIAGNOSTICS (AEMS)")
    print("  Framework: Liang et al. 2022, 'Mind the Gap'")
    print("=" * 72)

    records = filter_by_split(load_metadata(AEMS_MANIFEST_PATH), split="test")
    test_ids = set(rec["video_id"] for rec in records)
    print(f"[DATA] Test records: {len(records)}")

    # Load all four modality DBs
    print("[LOAD] Loading embedding DBs...")
    audio_db = torch.load(AEMS_AUDIO_EMBEDDINGS_PATH, weights_only=False)
    video_db = torch.load(AEMS_VID_EMBEDDINGS_PATH, weights_only=False)
    text_db = torch.load(TEXT_PATHS[args.text_variant].format(split="test"),
                         weights_only=False)

    common = sorted(
        set(audio_db.keys()) & set(video_db.keys()) & set(text_db.keys()) & test_ids
    )
    print(f"[DATA] Common test videos: {len(common)}")

    print("[BUILD] Building matrices (float32)...")
    audio = torch.stack([audio_db[v].float() for v in common])
    video = torch.stack([video_db[v].float() for v in common]).cpu()
    text = torch.stack([text_db[v].float() for v in common]).cpu()
    audio = F.normalize(audio, dim=-1)
    video = F.normalize(video, dim=-1)
    text = F.normalize(text, dim=-1)
    print(f"  audio: {audio.shape}, video: {video.shape}, text: {text.shape}")

    # ---------- 1. Modality centroids & gap distances ----------------------
    print("\n[1] MODALITY CENTROIDS (mean embedding per modality)")
    device = "cpu"
    c_audio = centroid(audio, device)
    c_video = centroid(video, device)
    c_text = centroid(text, device)
    print(f"  audio centroid norm:  {c_audio.norm():.4f}")
    print(f"  video centroid norm:  {c_video.norm():.4f}")
    print(f"  text  centroid norm:  {c_text.norm():.4f}")
    print("  (norms near 0 => embeddings spread evenly across sphere and cancel;")
    print("   norms near 1 => embeddings collapsed into a narrow cone = anisotropy)")

    print("\n[2] MODALITY GAP DISTANCES (inter-centroid)")
    pairs = {
        "audio-vs-text": (c_audio, c_text),
        "audio-vs-video": (c_audio, c_video),
        "text-vs-video": (c_text, c_video),
    }
    gap_matrix = {}
    for name, (a, b) in pairs.items():
        m = dist_metrics(a, b)
        gap_matrix[name] = m
        print(f"  {name:<14}: euclidean={m['euclidean']:.4f}  "
              f"cos_sim={m['cosine_sim']:.4f}  cos_dist={m['cosine_dist']:.4f}")

    # ---------- 2. Intra-modality spread (scale reference) ------------------
    print("\n[3] INTRA-MODALITY SPREAD (how tight each cloud is)")
    spread = {}
    for name, mat in [("audio", audio), ("video", video), ("text", text)]:
        s = float(mat.std(dim=0).mean().item())
        spread[name] = s
        print(f"  {name:<6}: mean component-std={s:.4f}")

    # ---------- 3. Gap normalized by inner spread ---------------------------
    print("\n[4] NORMALIZED MODALITY GAP (gap / intra-modality spread)")
    # Liang et al.: the "gap ratio" = inter-modality centroid distance relative
    # to the average intra-modality radius (measured via mean pairwise dist).
    avg_spread = np.mean([spread["audio"], spread["text"]])
    for name, (a, b) in pairs.items():
        m = gap_matrix[name]
        ratio = m["euclidean"] / (avg_spread + 1e-8)
        print(f"  {name:<14}: gap_ratio (eucl/spread) = {ratio:.4f}")

    # ---------- 4. Distribution of cross-modal pairwise distances ----------
    print("\n[5] CROSS-MODAL vs INTRA-MODAL DISTRIBUTIONS")
    # Subsample for speed
    rng = np.random.RandomState(args.seed)
    n = len(audio)
    idx = rng.choice(n, size=min(n, 500), replace=False)
    audio_s = audio[idx]
    text_s = text[idx]
    video_s = video[idx]

    with torch.no_grad():
        a_text_cos = (audio_s @ text_s.T).flatten()   # cross-modal
        audio_intra_cos = (audio_s @ audio_s.T).flatten()  # intra audio
        text_intra_cos = (text_s @ text_s.T).flatten()     # intra text
    print(f"  audio-text cross cosine: mean={a_text_cos.mean():.4f} "
          f"std={a_text_cos.std():.4f}")
    print(f"  audio intra      cosine: mean={audio_intra_cos.mean():.4f} "
          f"std={audio_intra_cos.std():.4f}")
    print(f"  text  intra      cosine: mean={text_intra_cos.mean():.4f} "
          f"std={text_intra_cos.std():.4f}")
    print(f"  Cross-modal cosine << intra-modal cosine often indicates a "
          f"large modality gap (Liang et al.).")

    # ---------- Save report ------------------------------------------------
    report = {
        "framework": "Liang et al. 2022 (Mind the Gap)",
        "text_variant": args.text_variant,
        "n_videos": n,
        "centroids": {
            "audio_norm": float(c_audio.norm()),
            "video_norm": float(c_video.norm()),
            "text_norm": float(c_text.norm()),
        },
        "modality_gap": gap_matrix,
        "intra_modality_spread": spread,
        "normalized_gap_ratio": {
            name: float(gap_matrix[name]["euclidean"] / (avg_spread + 1e-8))
            for name in pairs
        },
        "cross_vs_intra_cosine": {
            "audio_text_cross_mean": float(a_text_cos.mean()),
            "audio_intra_mean": float(audio_intra_cos.mean()),
            "text_intra_mean": float(text_intra_cos.mean()),
        },
    }
    out_path = os.path.join(OUTPUT_DIR, "modality_gap_report.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[DONE] Report saved: {out_path}")


if __name__ == "__main__":
    main()
