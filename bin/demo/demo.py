"""Demonstration pipeline with full explainability output."""

import argparse
import torch
from src.encoders.clap_encode import CLAPEncoder
from src.explainability.explain_retrieval import (
    explain_modality_contributions,
    explain_gating_decision,
    format_explanation,
)
from src.routing.query_router import (
    add_index_args, load_search_index, load_gate, load_clip,
    encode_text_query, encode_clap_text_query, search,
)

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
    index = load_search_index(args.video_embeds, args.audio_embeds, args.caption_embeds)
    print(f"{len(index[0])} candidates")

    print("[LOAD] Gating network...", end=" ", flush=True)
    gate = load_gate(args.gate_weights, DEVICE)
    print("done")

    print("[ENCODE] Query (CLIP + CLAP)...", end=" ", flush=True)
    clip_query = encode_text_query(load_clip(DEVICE), args.query, DEVICE)
    clap_query = encode_clap_text_query(CLAPEncoder(device=DEVICE), args.query, DEVICE)
    print("done")

    print("[SEARCH] Gating and scoring...", end=" ", flush=True)
    w, sim_v, sim_t, sim_a = search(index, gate, clip_query=clip_query, clap_query=clap_query)
    print("done")

    print("[EXPLAIN] Generating explanations...")
    contributions = explain_modality_contributions(w, sim_v, sim_t, sim_a, index[0], top_k=args.top_k)
    print()
    print(format_explanation(contributions, explain_gating_decision(w), top_k=args.top_k))

    print()
    print("=" * 70)
    print("  Demo complete. Results shown above.")
    print("=" * 70)


if __name__ == "__main__":
    main()
