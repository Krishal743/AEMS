import json
import os
import torch
import clip
from PIL import Image
from tqdm import tqdm

# ================= CONFIG =================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32        # safe for RTX 3060 (12GB)
METADATA = "data/processed/metadata/msrvtt_metadata.json"
OUT_PATH = "data/processed/video/video_embeddings.pt"
# =========================================

print("Using device:", DEVICE)

model, preprocess = clip.load("ViT-B/32", device=DEVICE)
model.eval()

with open(METADATA) as f:
    data = json.load(f)

# Collect unique videos
videos = {}
for d in data:
    videos[d["video_id"]] = d["frames_dir"]

print(f"Total unique videos: {len(videos)}")

video_embeddings = {}

with torch.no_grad():
    for vid, frames_dir in tqdm(videos.items(), desc="Encoding videos"):
        frame_files = sorted(os.listdir(frames_dir))
        frame_embeds = []

        for i in range(0, len(frame_files), BATCH_SIZE):
            batch_files = frame_files[i:i+BATCH_SIZE]

            images = []
            for f in batch_files:
                img = Image.open(os.path.join(frames_dir, f)).convert("RGB")
                images.append(preprocess(img))

            images = torch.stack(images).to(DEVICE)

            embeds = model.encode_image(images)
            embeds = embeds / embeds.norm(dim=-1, keepdim=True)

            frame_embeds.append(embeds.cpu())

        frame_embeds = torch.cat(frame_embeds, dim=0)
        video_embed = frame_embeds.mean(dim=0)
        video_embed = video_embed / video_embed.norm()

        video_embeddings[vid] = video_embed

torch.save(video_embeddings, OUT_PATH)
print(f"✅ Saved {len(video_embeddings)} video embeddings to {OUT_PATH}")
