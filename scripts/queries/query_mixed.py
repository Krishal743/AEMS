"""Mixed-modality query retrieval with explainability.

Usage:
  python3 scripts/queries/query_mixed.py \
      --text "a red car" \
      --image query.jpg \
      --video-embeds embeddings/video_embeddings.pt ...
"""

import sys, json, torch, argparse
from PIL import Image
import clip
from src.models.gating_network import GatingNetwork
from src.encoders.clap_encode import CLAPEncoder
from src.explainability.explain_retrieval import (
    explain_modality_contributions,
    explain_gating_decision,
    format_explanation,
)
from src.routing.query_router import (
    load_clip, encode_text_query, encode_image_query,
    compute_modal_similarities,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def encode_mixed_query(clip_model, preprocess_fn, text, image_path, device):
    embs = []
    if text:
        embs.append(encode_text_query(clip_model, text, device))
    if image_path:
        image = preprocess_fn(Image.open(image_path)).unsqueeze(0)
        embs.append(encode_image_query(clip_model, image, device))
    if not embs:
        raise ValueError("At least one of --text or --image required")
    fused = torch.stack(embs).mean(dim=0).float()
    return fused / fused.norm(dim=1, keepdim=True)


def main():
    parser = argparse.ArgumentParser(description="Mixed query retrieval")
    parser.add_argument("--text", default=None)
    parser.add_argument("--image", default=None)
    parser.add_argument("--video-embeds", default="embeddings/video_embeddings.pt")
    parser.add_argument("--audio-embeds", default="embeddings/audio_embeddings.pt")
    parser.add_argument("--caption-embeds", default="embeddings/caption_embeddings_test.pt")
    parser.add_argument("--gate-weights", default="models/gating_weights_meanpool.pth")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    if not args.text and not args.image:
        print("Provide at least one of --text or --image")
        return 1

    video_db = torch.load(args.video_embeds, weights_only=False)
    audio_db = torch.load(args.audio_embeds, weights_only=False)
    caption_db = torch.load(args.caption_embeds, weights_only=False)
    common = [v for v in video_db if v in audio_db and v in caption_db]
    print(f"Candidates: {len(common)} videos")

    clip_model, preprocess_fn = clip.load("ViT-B/32", device=DEVICE)
    clip_model.eval()
    query_emb = encode_mixed_query(clip_model, preprocess_fn, args.text, args.image, DEVICE)

    vid_m = torch.stack([torch.as_tensor(video_db[v]).float() for v in common])
    aud_m = torch.stack([torch.as_tensor(audio_db[v]).float() for v in common])
    cap_m = torch.stack([torch.as_tensor(caption_db[v]).float().max(dim=0)[0] for v in common])
    vid_m = torch.nn.functional.normalize(vid_m, p=2, dim=1)
    aud_m = torch.nn.functional.normalize(aud_m, p=2, dim=1)
    cap_m = torch.nn.functional.normalize(cap_m, p=2, dim=1)

    sim_v, sim_t, sim_a = compute_modal_similarities(query_emb, vid_m, aud_m, cap_m)

    gate = GatingNetwork(input_dim=512, hidden_dim=128).to(DEVICE)
    gate.load_state_dict(torch.load(args.gate_weights, map_location=DEVICE), strict=False)
    gate.eval()
    with torch.no_grad():
        w = gate(query_emb.to(DEVICE)).cpu().squeeze(0)

    gating_info = explain_gating_decision(w)
    contributions = explain_modality_contributions(w, sim_v, sim_t, sim_a, common, top_k=args.top_k)

    output = format_explanation(contributions, gating_info, top_k=args.top_k)
    print(output)


if __name__ == "__main__":
    main()
