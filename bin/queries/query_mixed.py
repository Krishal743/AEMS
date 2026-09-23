"""Mixed-modality query retrieval with explainability.

Usage:
  python3 bin/queries/query_mixed.py --text "a red car" --image query.jpg

Text and image are averaged in CLIP space for the visual/caption branches.
The audio branch lives in CLIP *text* space, so it is scored with the text part
only and is used only when --text is given.
"""

import argparse
import torch
import torch.nn.functional as F
import clip
from PIL import Image
from src.explainability.explain_retrieval import (
    explain_modality_contributions,
    explain_gating_decision,
    format_explanation,
)
from src.routing.query_router import (
    add_index_args, load_search_index, load_fusion_gate,
    encode_text_query, encode_image_query, search,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    parser = argparse.ArgumentParser(description="Mixed query retrieval")
    parser.add_argument("--text", default=None)
    parser.add_argument("--image", default=None)
    add_index_args(parser)
    args = parser.parse_args()

    if not args.text and not args.image:
        parser.error("provide at least one of --text or --image")

    index = load_search_index(args.video_embeds, args.audio_embeds, args.caption_embeds,
                              args.chunk_embeds)
    print(f"Candidates: {len(index.video_ids)} videos")
    gate = load_fusion_gate(args, DEVICE)

    clip_model, preprocess = clip.load("ViT-B/32", device=DEVICE)
    clip_model.eval()
    parts = []
    text_query = None
    if args.text:
        text_query = encode_text_query(clip_model, args.text, DEVICE)
        parts.append(text_query)
    if args.image:
        parts.append(encode_image_query(clip_model, preprocess(Image.open(args.image)).unsqueeze(0), DEVICE))
    clip_query = F.normalize(torch.cat(parts).mean(dim=0, keepdim=True), dim=1)

    audio_query = text_query

    w, *sims = search(index, clip_query=clip_query, audio_query=audio_query, gate=gate)
    contributions = explain_modality_contributions(w, sims, index.video_ids, top_k=args.top_k)
    print(format_explanation(contributions, explain_gating_decision(w), top_k=args.top_k))


if __name__ == "__main__":
    main()
