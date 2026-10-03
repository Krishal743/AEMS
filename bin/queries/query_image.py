"""Image-to-video retrieval with explainability.

The audio branch lives in CLIP text space, which image vectors are not aligned
with, so it is excluded.
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
    add_index_args, check_rerank_supported, rerank_and_report,
    load_index_from_args, load_fusion_gate, encode_image_query, search,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    parser = argparse.ArgumentParser(description="Image query retrieval")
    parser.add_argument("--image", required=True)
    add_index_args(parser)
    args = parser.parse_args()
    check_rerank_supported(parser, args, query_text=None)

    index = load_index_from_args(args)
    print(f"Candidates: {len(index.video_ids)} videos")
    gate = load_fusion_gate(args, DEVICE)

    clip_model, preprocess = clip.load("ViT-B/32", device=DEVICE)
    clip_model.eval()
    image = preprocess(Image.open(args.image)).unsqueeze(0)
    clip_query = encode_image_query(clip_model, image, DEVICE)

    w, *sims = search(index, clip_query=clip_query, gate=gate)
    rerank_and_report(args, w, sims, index, clip_query, device=DEVICE)
    contributions = explain_modality_contributions(w, sims, index.video_ids, top_k=args.top_k)
    print(format_explanation(contributions, explain_gating_decision(w), top_k=args.top_k))


if __name__ == "__main__":
    main()
