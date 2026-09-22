"""Text-to-video retrieval with explainability."""

import argparse
import torch
from src.explainability.explain_retrieval import (
    explain_modality_contributions,
    explain_gating_decision,
    format_explanation,
)
from src.routing.query_router import (
    add_index_args, load_search_index, load_fusion_gate, load_clip,
    encode_text_query, search,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    parser = argparse.ArgumentParser(description="Text query retrieval")
    parser.add_argument("--query", required=True)
    add_index_args(parser)
    args = parser.parse_args()

    index = load_search_index(args.video_embeds, args.audio_embeds, args.caption_embeds)
    print(f"Candidates: {len(index[0])} videos")
    gate = load_fusion_gate(args, DEVICE)

    clip_query = encode_text_query(load_clip(DEVICE), args.query, DEVICE)
    audio_query = clip_query  # the audio branch lives in CLIP text space

    w, sim_v, sim_t, sim_a = search(index, clip_query=clip_query, audio_query=audio_query, gate=gate)
    contributions = explain_modality_contributions(w, sim_v, sim_t, sim_a, index[0], top_k=args.top_k)
    print(format_explanation(contributions, explain_gating_decision(w), top_k=args.top_k))


if __name__ == "__main__":
    main()
