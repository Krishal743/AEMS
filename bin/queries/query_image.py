"""Image-to-video retrieval with explainability.

An image has no CLAP encoding, so the audio branch is excluded.
"""

import argparse
import torch
import clip
from PIL import Image
from src.explainability.explain_retrieval import (
    explain_modality_contributions,
    explain_gating_decision,
    format_explanation,
)
from src.routing.query_router import (
    add_index_args, load_search_index, load_gate, encode_image_query, search,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    parser = argparse.ArgumentParser(description="Image query retrieval")
    parser.add_argument("--image", required=True)
    add_index_args(parser)
    args = parser.parse_args()

    index = load_search_index(args.video_embeds, args.audio_embeds, args.caption_embeds)
    print(f"Candidates: {len(index[0])} videos")
    gate = load_gate(args.gate_weights, DEVICE)

    clip_model, preprocess = clip.load("ViT-B/32", device=DEVICE)
    clip_model.eval()
    image = preprocess(Image.open(args.image)).unsqueeze(0)
    clip_query = encode_image_query(clip_model, image, DEVICE)

    w, sim_v, sim_t, sim_a = search(index, gate, clip_query=clip_query)
    contributions = explain_modality_contributions(w, sim_v, sim_t, sim_a, index[0], top_k=args.top_k)
    print(format_explanation(contributions, explain_gating_decision(w), top_k=args.top_k))


if __name__ == "__main__":
    main()
