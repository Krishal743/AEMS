#!/usr/bin/env python3
"""
Eval the Temporal Transformer video embeddings on the official MSR-VTT test split.

This is the direct analogue of run_clip_baseline.py, but using the embeddings
produced by train_temporal_transformer.py (video_embeddings_transformer.pt).

It answers: "Is the temporal transformer better than mean-pooling for pure visual retrieval?"
"""

import json
import os
import sys
import torch
import clip
from tqdm import tqdm

# Make sure we can import from the same dir
from src.evaluation.evaluate_retrieval import evaluate_retrieval

# ================= CONFIG =================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
METADATA = "data/processed/metadata/msrvtt_metadata.json"
TEXT_BATCH_SIZE = 256

TRANSFORMER_EMBEDS = "embeddings/video_embeddings_transformer.pt"
OLD_EMBEDS = "embeddings/video_embeddings.pt"
# ==========================================

import argparse


def load_video_matrix(embeds_path):
    """Load all embeddings, return (video_matrix, video_ids)."""
    print(f"\n{'─'*60}")
    print(f"Loading: {embeds_path}")
    print(f"{'─'*60}")
    video_db = torch.load(embeds_path, weights_only=False)
    print(f"  Embeddings loaded: {len(video_db)} videos")

    video_ids_unique = sorted(video_db.keys())
    video_matrix = torch.stack([video_db[v] for v in video_ids_unique]).float()
    video_matrix = video_matrix / video_matrix.norm(dim=-1, keepdim=True)

    return video_matrix, video_ids_unique


def evaluate(embeds_path, clip_model, texts, text_video_ids, tag=""):
    """Run retrieval evaluation for a given embedding file. Returns metrics dict."""
    video_matrix, video_ids = load_video_matrix(embeds_path)
    if video_matrix is None:
        print(f"  [SKIP] No overlap — skipping {embeds_path}")
        return None

    print(f"  Encoding {len(texts)} test captions...")
    all_text_embeds = []
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), TEXT_BATCH_SIZE), desc="CLIP text"):
            batch = texts[i:i + TEXT_BATCH_SIZE]
            tokens = clip.tokenize(batch).to(DEVICE)
            embeds = clip_model.encode_text(tokens)
            embeds = embeds / embeds.norm(dim=-1, keepdim=True)
            all_text_embeds.append(embeds.cpu())

    text_embeds = torch.cat(all_text_embeds, dim=0).float()

    print("  Computing similarity...")
    sim = text_embeds @ video_matrix.T

    print("  Computing Recall@K...")
    metrics = evaluate_retrieval(sim, text_video_ids, video_ids, ks=[1, 5, 10])

    name = embeds_path.split("/")[-1].replace(".pt", "")
    print(f"\n  Results [{name}]:")
    print(f"    R@1: {metrics['R@1']:.4f}  R@5: {metrics['R@5']:.4f}  R@10: {metrics['R@10']:.4f}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate temporal transformer video embeddings")
    parser.add_argument("--compare", action="store_true",
                        help="Also evaluate on old mean-pool embeddings and show comparison")
    parser.add_argument("--video-embeds", default=TRANSFORMER_EMBEDS,
                        help="Path to video embeddings (default: transformer output)")
    args = parser.parse_args()

    print("=" * 60)
    print("VIDEO RETRIEVAL EVALUATION (official test split)")
    print("=" * 60)
    print(f"Device: {DEVICE}")

    if args.compare:
        if not os.path.exists(TRANSFORMER_EMBEDS) or not os.path.exists(OLD_EMBEDS):
            print("[ERROR] Both embedding files must exist for --compare mode.")
            print(f"  Transformer: {TRANSFORMER_EMBEDS}  (exists: {os.path.exists(TRANSFORMER_EMBEDS)})")
            print(f"  Old (mean-pool): {OLD_EMBEDS}  (exists: {os.path.exists(OLD_EMBEDS)})")
            return 1
    else:
        if not os.path.exists(args.video_embeds):
            print(f"\n[ERROR] {args.video_embeds} does not exist yet.")
            print("Wait for train_temporal_transformer.py to finish (it exports at the very end).")
            return 1

    print("[INIT] Loading CLIP model...")
    clip_model, _ = clip.load("ViT-B/32", device=DEVICE)
    clip_model.eval()

    print("[DATA] Loading metadata...")
    with open(METADATA) as f:
        data = json.load(f)

    test_data = [d for d in data if d["split"] == "test"]
    texts = [d["text"] for d in test_data]
    text_video_ids = [d["video_id"] for d in test_data]
    print(f"[DATA] Test captions: {len(texts)} (from {len(set(text_video_ids))} unique videos)")

    if args.compare:
        print("\n" + "=" * 60)
        print("COMPARISON MODE: Transformer vs Mean-Pool")
        print("=" * 60)

        m_old = evaluate(OLD_EMBEDS, clip_model, texts, text_video_ids, tag="old")
        m_new = evaluate(TRANSFORMER_EMBEDS, clip_model, texts, text_video_ids, tag="transformer")

        if m_old is None or m_new is None:
            return 1

        print("\n" + "=" * 60)
        print("COMPARISON SUMMARY")
        print("=" * 60)
        print(f"{'Metric':>8}  {'Mean-Pool':>10}  {'Transformer':>12}  {'Δ':>10}")
        print(f"{'───':>8}  {'──────────':>10}  {'────────────':>12}  {'──':>10}")
        for k in [1, 5, 10]:
            old_v = m_old[f"R@{k}"]
            new_v = m_new[f"R@{k}"]
            delta = new_v - old_v
            sign = "+" if delta > 0 else ""
            print(f"R@{k:<5}  {old_v:>10.4f}  {new_v:>12.4f}  {sign}{delta:>+.4f}")
        print("=" * 60)
        if m_new["R@1"] > m_old["R@1"]:
            print("✓ Transformer improves over mean-pool baseline")
        else:
            print("✗ Transformer did NOT improve over mean-pool baseline")
    else:
        metrics = evaluate(args.video_embeds, clip_model, texts, text_video_ids)
        if metrics is None:
            return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())
