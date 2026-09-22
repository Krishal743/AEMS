"""Video-to-video retrieval with explainability.

The query video is encoded from 16 uniform frames with CLIP; its soundtrack is
not encoded, so the audio branch is excluded.
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
    add_index_args, load_search_index, load_fusion_gate, encode_video_query, search,
)

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
            frames.append(preprocess(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))))
    cap.release()
    if not frames:
        raise ValueError(f"could not decode any frames from {video_path}")
    return torch.stack(frames)


def main():
    parser = argparse.ArgumentParser(description="Video query retrieval")
    parser.add_argument("--video", required=True)
    add_index_args(parser)
    args = parser.parse_args()

    index = load_search_index(args.video_embeds, args.audio_embeds, args.caption_embeds)
    print(f"Candidates: {len(index[0])} videos")
    gate = load_fusion_gate(args, DEVICE)

    clip_model, preprocess = clip.load("ViT-B/32", device=DEVICE)
    clip_model.eval()
    clip_query = encode_video_query(clip_model, load_video_frames(args.video, preprocess), DEVICE)

    w, sim_v, sim_t, sim_a = search(index, clip_query=clip_query, gate=gate)
    contributions = explain_modality_contributions(w, sim_v, sim_t, sim_a, index[0], top_k=args.top_k)
    print(format_explanation(contributions, explain_gating_decision(w), top_k=args.top_k))


if __name__ == "__main__":
    main()
