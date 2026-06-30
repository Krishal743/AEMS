#!/usr/bin/env python3
"""
Flexible multimodal retrieval evaluation.

Supports swapping the VIDEO branch (mean-pool CLIP vs Temporal Transformer)
while keeping the same audio + caption + gating machinery.

Usage examples (after transformer training finishes):
    python scripts/eval_multimodal.py
    python scripts/eval_multimodal.py --video-embeds embeddings/video_embeddings.pt
    python scripts/eval_multimodal.py --video-embeds embeddings/video_embeddings_transformer.pt --use-gate

This script works on the official test split and respects the existing
intersection logic used by run_query_routing.py and run_three_branch.py.
"""

import os
import sys
import json
import argparse
import torch
import clip
import gc
from tqdm import tqdm

from src.evaluation.evaluate_retrieval import evaluate_retrieval
from src.encoders.clap_encode import CLAPEncoder

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
METADATA = "data/processed/metadata/msrvtt_metadata.json"

DEFAULT_VIDEO = "embeddings/video_embeddings_transformer.pt"
DEFAULT_AUDIO = "embeddings/audio_embeddings.pt"
DEFAULT_CAPTION = "embeddings/caption_embeddings.pt"
DEFAULT_GATE = "models/gating_weights.pth"


def normalize(x):
    return x / x.norm(dim=-1, keepdim=True)


def load_embeddings(path):
    print(f"[LOAD] {path}")
    db = torch.load(path, weights_only=False)
    print(f"       -> {len(db)} entries")
    return db


def main():
    parser = argparse.ArgumentParser(description="Evaluate multimodal video retrieval (swappable video branch)")
    parser.add_argument("--video-embeds", default=DEFAULT_VIDEO,
                        help="Path to video embeddings dict (transformer or old mean-pool)")
    parser.add_argument("--audio-embeds", default=DEFAULT_AUDIO)
    parser.add_argument("--caption-embeds", default=DEFAULT_CAPTION)
    parser.add_argument("--gate-weights", default=DEFAULT_GATE)
    parser.add_argument("--use-gate", action="store_true",
                        help="Also run gated fusion using the learned gating network")
    parser.add_argument("--max-videos", type=int, default=None,
                        help="Limit for quick debugging (like the old MAX_VIDEOS)")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    video_path = args.video_embeds
    print("=" * 70)
    print("MULTIMODAL RETRIEVAL EVALUATION")
    print("=" * 70)
    print(f"Video branch : {video_path}")
    print(f"Use gating   : {args.use_gate}")
    print(f"Device       : {DEVICE}")

    if not os.path.exists(video_path):
        print(f"\n[ERROR] Video embeddings not found: {video_path}")
        print("Run the training to completion first (it exports at the end).")
        return 1

    # Load all dbs
    print("\n[SETUP] Loading precomputed embeddings...")
    video_db = load_embeddings(video_path)
    audio_db = load_embeddings(args.audio_embeds)
    caption_db = load_embeddings(args.caption_embeds)

    print("\n[DATA] Loading test split...")
    with open(METADATA) as f:
        meta = json.load(f)

    test_items = [m for m in meta if m["split"] == "test"]
    test_video_ids = set(m["video_id"] for m in test_items)

    # Intersection (same logic as run_query_routing / run_three_branch)
    common_vids = [vid for vid in sorted(video_db.keys())
                   if vid in audio_db and vid in caption_db and vid in test_video_ids]

    if args.max_videos:
        common_vids = common_vids[:args.max_videos]

    print(f"[FILTER] Common videos with all modalities + test queries: {len(common_vids)}")

    if len(common_vids) == 0:
        print("[ERROR] No overlapping videos.")
        return 1

    # Prepare video matrices
    video_matrix = torch.stack([torch.tensor(video_db[v]) for v in common_vids]).float()
    video_matrix = normalize(video_matrix)

    audio_matrix = torch.stack([torch.tensor(audio_db[v]) for v in common_vids]).float()
    audio_matrix = normalize(audio_matrix)

    # Build list of queries (one or more captions per video)
    texts = []
    text_video_ids = []
    for item in test_items:
        vid = item["video_id"]
        if vid in set(common_vids):
            texts.append(item["text"])
            text_video_ids.append(vid)

    print(f"[QUERIES] {len(texts)} test captions over {len(common_vids)} videos")

    # Encode text queries with CLIP
    print("\n[CLIP] Loading CLIP and encoding queries...")
    clip_model, _ = clip.load("ViT-B/32", device=DEVICE)
    clip_model.eval()

    text_embeds_list = []
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), args.batch_size), desc="Encoding text"):
            batch = texts[i:i + args.batch_size]
            toks = clip.tokenize(batch).to(DEVICE)
            emb = clip_model.encode_text(toks)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            text_embeds_list.append(emb.cpu())

    text_embeds = torch.cat(text_embeds_list, dim=0).float()
    del clip_model, text_embeds_list
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    # Move modality banks to device for sim computation (block-wise if needed, but test set is small)
    print("[SIM] Computing similarities...")
    text_dev = text_embeds.to(DEVICE).float()
    v_dev = video_matrix.to(DEVICE).float()
    a_dev = audio_matrix.to(DEVICE).float()

    sim_v = (text_dev @ v_dev.T).cpu()
    sim_a = (text_dev @ a_dev.T).cpu()

    # sim_t = MAX over 20 captions per video
    print("[CAPTIONS] Computing sim_t (max over 20 captions)...")
    sim_t_list = []
    for vid in tqdm(common_vids, desc="sim_t"):
        caps = caption_db[vid].to(DEVICE).float()   # [20, 512]
        sims = text_dev @ caps.T                    # [Nq, 20]
        sim_t_list.append(sims.max(dim=1).values.cpu())
    sim_t = torch.stack(sim_t_list, dim=1)          # [Nq, Nv]

    del text_dev, v_dev, a_dev
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    # Individual branches
    print("\n" + "-" * 70)
    print("BRANCH RESULTS (test split)")
    print("-" * 70)
    for name, sim in [("sim_v (video)", sim_v), ("sim_t (captions)", sim_t), ("sim_a (audio)", sim_a)]:
        res = evaluate_retrieval(sim, text_video_ids, common_vids)
        print(f"{name:20s}  R@1={res['R@1']:.4f}  R@5={res['R@5']:.4f}  R@10={res['R@10']:.4f}")

    # Equal fusion
    print("\n" + "-" * 70)
    print("EQUAL-WEIGHT FUSION")
    print("-" * 70)
    sim_equal = (sim_v + sim_t + sim_a) / 3.0
    res = evaluate_retrieval(sim_equal, text_video_ids, common_vids)
    print(f"Equal fusion          R@1={res['R@1']:.4f}  R@5={res['R@5']:.4f}  R@10={res['R@10']:.4f}")

    # Gated fusion (if requested and weights exist)
    if args.use_gate:
        if not os.path.exists(args.gate_weights):
            print(f"\n[SKIP] Gate weights not found: {args.gate_weights}")
        else:
            print("\n" + "-" * 70)
            print("GATED FUSION (learned weights from query)")
            print("-" * 70)
            from run_query_routing import GatingNetwork   # reuse the exact class

            gate = GatingNetwork().to(DEVICE)
            gate.load_state_dict(torch.load(args.gate_weights, map_location=DEVICE))
            gate.eval()

            # We need per-query weights
            # Re-encode queries with CLIP just for the gate input (or reuse text_embeds)
            # The gate was trained on normalized CLIP text embeds.
            gate_input = text_embeds.to(DEVICE).float()

            with torch.no_grad():
                weights = gate(gate_input)   # [N_queries, 3]

            # For efficiency we compute gated score in blocks
            # Because Nv is ~2-3k here, we can do it directly
            w_v = weights[:, 0:1].cpu()
            w_t = weights[:, 1:2].cpu()
            w_a = weights[:, 2:3].cpu()

            sim_gated = w_v * sim_v + w_t * sim_t + w_a * sim_a
            res = evaluate_retrieval(sim_gated, text_video_ids, common_vids)
            print(f"Gated fusion          R@1={res['R@1']:.4f}  R@5={res['R@5']:.4f}  R@10={res['R@10']:.4f}")

            # Show average weights
            print("\nMean gate weights over queries:")
            print(f"  w_v (video) : {w_v.mean().item():.4f}")
            print(f"  w_t (text)  : {w_t.mean().item():.4f}")
            print(f"  w_a (audio) : {w_a.mean().item():.4f}")

    print("\n" + "=" * 70)
    print("DONE")
    print(f"Video source used: {video_path}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
