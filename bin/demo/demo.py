"""Demonstration pipeline with full explainability output."""

import argparse
import torch
from src.explainability.explain_retrieval import (
    explain_modality_contributions,
    explain_gating_decision,
    format_explanation,
)
from src.routing.query_router import (
    add_index_args, apply_rerank, load_search_index, load_fusion_gate, load_clip,
    encode_text_query, search,
)
from src.config import AEMS_MANIFEST_PATH
from src.data.metadata import load_metadata

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    parser = argparse.ArgumentParser(description="AEMS Demo Pipeline")
    parser.add_argument("--query", required=True, help="Text query")
    add_index_args(parser)
    args = parser.parse_args()

    print("=" * 70)
    print("  AEMS — Adaptive Explainable Multimodal Search")
    print("=" * 70)
    print(f"  Query: \"{args.query}\"")
    print("  Query type: text")
    print()

    print("[LOAD] Search index...", end=" ", flush=True)
    index = load_search_index(args.video_embeds, args.audio_embeds, args.caption_embeds,
                              args.chunk_embeds)
    print(f"{len(index.video_ids)} candidates")

    gate = load_fusion_gate(args, DEVICE)
    print(f"[FUSION] {'gating network' if gate is not None else 'fixed weights'}")

    print("[ENCODE] Query (CLIP)...", end=" ", flush=True)
    clip_query = encode_text_query(load_clip(DEVICE), args.query, DEVICE)
    audio_query = clip_query  # the audio branch lives in CLIP text space
    print("done")

    print("[SEARCH] Scoring and fusing...", end=" ", flush=True)
    w, *sims = search(index, clip_query=clip_query, audio_query=audio_query, gate=gate)
    if args.rerank != "none":
        records, chunk_db = None, None
        if args.rerank == "cross":
            records = {r["video_id"]: r for r in load_metadata(AEMS_MANIFEST_PATH)}
            chunk_db = torch.load(args.chunk_embeds, weights_only=False)
        ranking = apply_rerank(args, w, sims, index, clip_query=clip_query.cpu(),
                               query_text=args.query, records=records, chunk_db=chunk_db,
                               device=DEVICE)
        order = torch.argsort(ranking, descending=True)[:args.top_k]
        print(f"\nReranked top-{args.top_k} (--rerank {args.rerank}):")
        for rank, idx in enumerate(order.tolist(), 1):
            print(f"  {rank}. {index.video_ids[idx]}  score={ranking[idx]:.4f}")
    print("done")

    print("[EXPLAIN] Generating explanations...")
    contributions = explain_modality_contributions(w, sims, index.video_ids, top_k=args.top_k)
    print()
    print(format_explanation(contributions, explain_gating_decision(w), top_k=args.top_k))

    print()
    print("=" * 70)
    print("  Demo complete. Results shown above.")
    print("=" * 70)


if __name__ == "__main__":
    main()
