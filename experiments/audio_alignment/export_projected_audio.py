#!/usr/bin/env python3
"""
Export projected CLAP-audio embeddings into CLIP-text space.
==============================================================
Applies the linear projection learned by fit_projection.py to the FULL audio
gallery (all splits) and saves a new embedding DB keyed by video_id, L2-
normalized, drop-in compatible with eval_aems_retrieval.py and
train_aems_gating.py.

Run:
  export PYTHONPATH="${PYTHONPATH}:$(pwd)"
  python scripts/alignment/export_projected_audio.py
"""

import argparse
import json
import os

import torch
import torch.nn.functional as F

from src.config import (
    AEMS_AUDIO_EMBEDDINGS_PATH,
    set_seeds,
)

MODEL_PATH = "models/audio_to_cliptext_projection.pt"
OUTPUT_PATH = "embeddings/aems_audio_projected_cliptext.pt"
OUTPUT_DIR = "outputs/alignment"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def normalize_row(x):
    return F.normalize(x, dim=-1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=MODEL_PATH)
    parser.add_argument("--audio-embeds", type=str, default=AEMS_AUDIO_EMBEDDINGS_PATH)
    parser.add_argument("--out", type=str, default=OUTPUT_PATH)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    set_seeds(args.seed)

    print("=" * 72)
    print("EXPORT PROJECTED CLAP-AUDIO -> CLIP-TEXT EMBEDDINGS")
    print("=" * 72)

    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Projection not found: {args.model}. "
                                f"Run fit_projection.py first.")

    ckpt = torch.load(args.model, map_location="cpu", weights_only=False)
    W = ckpt["W"].float()          # [d, d]
    b = ckpt["b"].float()          # [d]
    d = W.shape[0]
    print(f"[MODEL] Loaded projection method='{ckpt['method']}', d={d}")
    print(f"        (fit on {ckpt['n_train_anchors']} train anchors)")

    audio_db = torch.load(args.audio_embeds, map_location="cpu", weights_only=False)
    print(f"[LOAD] Audio DB: {len(audio_db)} videos")

    # Batch in blocks to respect memory
    keys = sorted(audio_db.keys())
    out = {}
    B = 512
    with torch.no_grad():
        # Build full matrix (6770 x 512 is small, but keep block discipline)
        for i in range(0, len(keys), B):
            chunk_keys = keys[i:i + B]
            X = torch.stack([audio_db[v].float() for v in chunk_keys])  # [b, d]
            X = normalize_row(X)
            proj = X @ W.T + b.view(1, -1)        # [b, d]
            proj = normalize_row(proj)            # L2-normalize output
            for v, emb in zip(chunk_keys, proj):
                out[v] = emb.cpu()

    torch.save(out, args.out)
    print(f"[SAVE] Projected audio embeddings: {args.out} "
          f"({len(out)} videos)")
    # Sanity: norms
    sample = torch.stack(list(out.values())[:500])
    norms = sample.norm(dim=1)
    print(f"[CHECK] Output L2 norms: mean={norms.mean():.4f} std={norms.std():.4f}")

    report = {
        "model_path": args.model,
        "method": ckpt["method"],
        "n_output_videos": len(out),
        "output_path": args.out,
        "output_l2_norm_mean": float(norms.mean()),
    }
    with open(os.path.join(OUTPUT_DIR, "export_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"[DONE] Export report: {os.path.join(OUTPUT_DIR, 'export_report.json')}")


if __name__ == "__main__":
    main()
