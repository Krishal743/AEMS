"""Video-to-video retrieval with explainability."""

import sys, json, torch, argparse, os
import clip
from PIL import Image
from src.models.gating_network import GatingNetwork
from src.explainability.explain_retrieval import (
    explain_modality_contributions,
    explain_gating_decision,
    format_explanation,
)
from src.routing.query_router import encode_video_query, compute_modal_similarities

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_FRAMES = 16


def load_video_frames(video_path, preprocess):
    import cv2
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = [int(i * total / NUM_FRAMES) for i in range(NUM_FRAMES)]
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frames.append(preprocess(pil))
        else:
            frames.append(torch.zeros(3, 224, 224))
    cap.release()
    return torch.stack(frames)


def main():
    parser = argparse.ArgumentParser(description="Video query retrieval")
    parser.add_argument("--video", required=True)
    parser.add_argument("--video-embeds", default="embeddings/video_embeddings.pt")
    parser.add_argument("--audio-embeds", default="embeddings/audio_embeddings.pt")
    parser.add_argument("--caption-embeds", default="embeddings/caption_embeddings_test.pt")
    parser.add_argument("--gate-weights", default="models/gating_weights_meanpool.pth")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    video_db = torch.load(args.video_embeds, weights_only=False)
    audio_db = torch.load(args.audio_embeds, weights_only=False)
    caption_db = torch.load(args.caption_embeds, weights_only=False)
    common = [v for v in video_db if v in audio_db and v in caption_db]
    print(f"Candidates: {len(common)} videos")

    clip_model, preprocess = clip.load("ViT-B/32", device=DEVICE)
    clip_model.eval()
    frames = load_video_frames(args.video, preprocess)
    query_emb = encode_video_query(clip_model, frames, DEVICE)

    vid_m = torch.stack([torch.as_tensor(video_db[v]).float() for v in common])
    aud_m = torch.stack([torch.as_tensor(audio_db[v]).float() for v in common])
    cap_m = torch.stack([torch.as_tensor(caption_db[v]).float().max(dim=0)[0] for v in common])
    vid_m = torch.nn.functional.normalize(vid_m, p=2, dim=1)
    aud_m = torch.nn.functional.normalize(aud_m, p=2, dim=1)
    cap_m = torch.nn.functional.normalize(cap_m, p=2, dim=1)

    sim_v, sim_t, sim_a = compute_modal_similarities(query_emb, vid_m, aud_m, cap_m)

    gate = GatingNetwork(text_dim=512, hidden_dim=128).to(DEVICE)
    gate.load_state_dict(torch.load(args.gate_weights, map_location=DEVICE))
    gate.eval()
    with torch.no_grad():
        w = gate(query_emb.to(DEVICE)).cpu().squeeze(0)

    gating_info = explain_gating_decision(w)
    contributions = explain_modality_contributions(w, sim_v, sim_t, sim_a, common, top_k=args.top_k)

    output = format_explanation(contributions, gating_info, top_k=args.top_k)
    print(output)


if __name__ == "__main__":
    main()
