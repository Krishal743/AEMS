"""Demonstration pipeline with full explainability output."""

import sys, json, torch, argparse, os
import clip
from src.models.gating_network import GatingNetwork
from src.explainability.explain_retrieval import (
    explain_modality_contributions,
    explain_gating_decision,
    format_explanation,
)
from src.routing.query_router import load_clip, encode_text_query, compute_modal_similarities

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    parser = argparse.ArgumentParser(description="AEMS Demo Pipeline")
    parser.add_argument("--query", required=True, help="Text query")
    parser.add_argument("--video-embeds", default="embeddings/video_embeddings.pt")
    parser.add_argument("--audio-embeds", default="embeddings/audio_embeddings.pt")
    parser.add_argument("--caption-embeds", default="embeddings/caption_embeddings_test.pt")
    parser.add_argument("--gate-weights", default="models/gating_weights_meanpool.pth")
    parser.add_argument("--metadata", default="data/processed/metadata/msrvtt_metadata.json")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    print("=" * 70)
    print("  AEMS — Adaptive Explainable Multimodal Search")
    print("=" * 70)
    print(f"  Query: \"{args.query}\"")
    print(f"  Query type: text")
    print()

    print("[LOAD] Embeddings...", end=" ", flush=True)
    video_db = torch.load(args.video_embeds, weights_only=False)
    audio_db = torch.load(args.audio_embeds, weights_only=False)
    caption_db = torch.load(args.caption_embeds, weights_only=False)
    common = sorted([v for v in video_db if v in audio_db and v in caption_db])
    print(f"{len(common)} candidates")

    print("[LOAD] CLIP model...", end=" ", flush=True)
    clip_model = load_clip(DEVICE)
    print("done")

    print("[ENCODE] Query...", end=" ", flush=True)
    query_emb = encode_text_query(clip_model, args.query, DEVICE)
    print("done")

    print("[BUILD] Modality matrices...", end=" ", flush=True)
    vid_m = torch.stack([torch.as_tensor(video_db[v]).float() for v in common])
    aud_m = torch.stack([torch.as_tensor(audio_db[v]).float() for v in common])
    cap_m = torch.stack([torch.as_tensor(caption_db[v]).float().max(dim=0)[0] for v in common])
    vid_m = torch.nn.functional.normalize(vid_m, p=2, dim=1)
    aud_m = torch.nn.functional.normalize(aud_m, p=2, dim=1)
    cap_m = torch.nn.functional.normalize(cap_m, p=2, dim=1)
    print("done")

    print("[SEARCH] Computing similarities...", end=" ", flush=True)
    sim_v, sim_t, sim_a = compute_modal_similarities(query_emb, vid_m, aud_m, cap_m)
    print("done")

    print("[GATE] Loading gating network...", end=" ", flush=True)
    gate = GatingNetwork(input_dim=512, hidden_dim=128).to(DEVICE)
    gate.load_state_dict(torch.load(args.gate_weights, map_location=DEVICE), strict=False)
    gate.eval()
    with torch.no_grad():
        w = gate(query_emb.to(DEVICE)).cpu().squeeze(0)
    print("done")

    print("[EXPLAIN] Generating explanations...")
    gating_info = explain_gating_decision(w)
    contributions = explain_modality_contributions(w, sim_v, sim_t, sim_a, common, top_k=args.top_k)
    print()

    output = format_explanation(contributions, gating_info, top_k=args.top_k)
    print(output)

    print()
    print("=" * 70)
    print("  Demo complete. Results shown above.")
    print("=" * 70)


if __name__ == "__main__":
    main()
