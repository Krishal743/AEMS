import argparse, os, gc
import torch
import torch.nn.functional as F
import clip
from PIL import Image
from tqdm import tqdm
from src.models.temporal_transformer import TemporalTransformer
from src.config import (AEMS_MANIFEST_PATH, AEMS_FRAMES_DIR, AEMS_TRANSFORMER_BEST_PATH,
                         AEMS_VIDEO_EMBEDDINGS_TRANSFORMER_PATH, NUM_FRAMES, DEVICE, set_seeds)
from src.data.metadata import load_metadata

parser = argparse.ArgumentParser(description="Export AEMS transformer embeddings")
parser.add_argument("--manifest", type=str, default=AEMS_MANIFEST_PATH)
parser.add_argument("--frames-dir", type=str, default=AEMS_FRAMES_DIR)
parser.add_argument("--checkpoint", type=str, default=AEMS_TRANSFORMER_BEST_PATH)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

set_seeds(args.seed)
os.makedirs("embeddings", exist_ok=True)

print(f"[INIT] Loading CLIP on {DEVICE}")
clip_model, preprocess = clip.load("ViT-B/32", device=DEVICE)
clip_model.eval()
for p in clip_model.parameters():
    p.requires_grad_(False)

print(f"[MODEL] Loading TemporalTransformer from {args.checkpoint}")
model = TemporalTransformer().to(DEVICE)
model.load_state_dict(torch.load(args.checkpoint, map_location=DEVICE, weights_only=False))
model.eval()

print(f"[DATA] Loading manifest: {args.manifest}")
records = load_metadata(args.manifest)
unique_video_ids = sorted(set(item["video_id"] for item in records))
print(f"[DATA] {len(unique_video_ids)} unique videos")


def encode_clip_frames(frame_tensors):
    with torch.no_grad(), torch.amp.autocast("cuda"):
        flat = frame_tensors.flatten(0, 1)
        embeds = clip_model.encode_image(flat)
        embeds = embeds.view(1, NUM_FRAMES, 512).float()
        embeds = embeds / embeds.norm(dim=-1, keepdim=True)
    return embeds


def get_video_frames(video_id):
    frame_dir = os.path.join(args.frames_dir, video_id)
    if not os.path.exists(frame_dir):
        return None
    frames = []
    for i in range(NUM_FRAMES):
        frame_path = os.path.join(frame_dir, "frame_%04d.jpg" % i)
        if not os.path.exists(frame_path):
            return None
        img = Image.open(frame_path).convert("RGB")
        img = preprocess(img)
        frames.append(img)
    if len(frames) != NUM_FRAMES:
        return None
    return torch.stack(frames).unsqueeze(0).to(DEVICE)


all_video_embeds = {}
for video_id in tqdm(unique_video_ids, desc="Exporting embeddings"):
    frames = get_video_frames(video_id)
    if frames is None:
        continue
    with torch.no_grad():
        frame_embeds = encode_clip_frames(frames)
        video_embed = model(frame_embeds)
    all_video_embeds[video_id] = video_embed.squeeze(0).cpu()
    if len(all_video_embeds) % 1000 == 0:
        torch.save(all_video_embeds, AEMS_VIDEO_EMBEDDINGS_TRANSFORMER_PATH)
    del frames, frame_embeds, video_embed
    gc.collect()
    torch.cuda.empty_cache()

torch.save(all_video_embeds, AEMS_VIDEO_EMBEDDINGS_TRANSFORMER_PATH)
print(f"\n[DONE] Saved {len(all_video_embeds)} embeddings to {AEMS_VIDEO_EMBEDDINGS_TRANSFORMER_PATH}")
