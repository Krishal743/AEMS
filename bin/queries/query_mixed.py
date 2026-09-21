"""Mixed-modality query retrieval with explainability.

Usage:
  python3 bin/queries/query_mixed.py --text "a red car" --image query.jpg

Text and image are averaged in CLIP space for the visual/caption branches.
Only the text part has a CLAP encoding, so the audio branch is used only when
--text is given.
"""

import argparse
import torch
import torch.nn.functional as F
import clip
from PIL import Image
from src.encoders.clap_encode import CLAPEncoder
from src.explainability.explain_retrieval import (
    explain_modality_contributions,
    explain_gating_decision,
    format_explanation,
)
from src.routing.query_router import (
    add_index_args, load_search_index, load_gate,
    encode_text_query, encode_image_query, encode_clap_text_query, search,
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

    index = load_search_index(args.video_embeds, args.audio_embeds, args.caption_embeds)
    print(f"Candidates: {len(index[0])} videos")
    gate = load_gate(args.gate_weights, DEVICE)

    clip_model, preprocess = clip.load("ViT-B/32", device=DEVICE)
    clip_model.eval()
    parts = []
    if args.text:
        parts.append(encode_text_query(clip_model, args.text, DEVICE))
    if args.image:
        parts.append(encode_image_query(clip_model, preprocess(Image.open(args.image)).unsqueeze(0), DEVICE))
    clip_query = F.normalize(torch.cat(parts).mean(dim=0, keepdim=True), dim=1)

    clap_query = None
    if args.text:
        clap_query = encode_clap_text_query(CLAPEncoder(device=DEVICE), args.text, DEVICE)

    w, sim_v, sim_t, sim_a = search(index, gate, clip_query=clip_query, clap_query=clap_query)
    contributions = explain_modality_contributions(w, sim_v, sim_t, sim_a, index[0], top_k=args.top_k)
    print(format_explanation(contributions, explain_gating_decision(w), top_k=args.top_k))


if __name__ == "__main__":
    main()
