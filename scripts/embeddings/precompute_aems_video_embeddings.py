import argparse, os, glob, gc
import torch
import torch.nn.functional as F
import clip
from PIL import Image
from tqdm import tqdm
from src.config import AEMS_MANIFEST_PATH, AEMS_FRAMES_DIR, AEMS_VID_EMBEDDINGS_PATH, NUM_FRAMES, DEVICE, set_seeds
from src.data.metadata import load_metadata

parser = argparse.ArgumentParser(description="Precompute AEMS video embeddings (CLIP meanpool)")
parser.add_argument("--manifest", type=str, default=AEMS_MANIFEST_PATH)
parser.add_argument("--frames-dir", type=str, default=AEMS_FRAMES_DIR)
parser.add_argument("--batch-size", type=int, default=64)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

set_seeds(args.seed)
os.makedirs("embeddings", exist_ok=True)

print(f"[INIT] Using device: {DEVICE}")
model, preprocess = clip.load("ViT-B/32", device=DEVICE)
model.eval()

print(f"[DATA] Loading manifest: {args.manifest}")
records = load_metadata(args.manifest)
videos = {}
for item in records:
    vid = item["video_id"]
    if vid not in videos:
        videos[vid] = os.path.join(args.frames_dir, vid)

out = {}
skipped = 0
skip_reasons = {}

with torch.no_grad():
    for vid, frames_dir in tqdm(videos.items(), desc="Encoding videos"):
        frame_files = sorted(glob.glob(os.path.join(frames_dir, "frame_*.jpg")))
        if len(frame_files) != NUM_FRAMES:
            skipped += 1
            skip_reasons[vid] = f"expected {NUM_FRAMES} frames, got {len(frame_files)}"
            continue
        frame_embeds = []
        for i in range(0, len(frame_files), args.batch_size):
            batch_files = frame_files[i:i + args.batch_size]
            images = []
            for f in batch_files:
                img = Image.open(f).convert("RGB")
                images.append(preprocess(img))
            images = torch.stack(images).to(DEVICE)
            embeds = model.encode_image(images)
            embeds = F.normalize(embeds, dim=-1)
            frame_embeds.append(embeds.cpu())
            del images, embeds
            gc.collect()
            torch.cuda.empty_cache()
        frame_embeds = torch.cat(frame_embeds, dim=0)
        video_embed = frame_embeds.mean(dim=0)
        video_embed = F.normalize(video_embed, dim=0)
        out[vid] = video_embed

torch.save(out, AEMS_VID_EMBEDDINGS_PATH)
print(f"\n[DONE] Saved {len(out)} embeddings to {AEMS_VID_EMBEDDINGS_PATH}")
print(f"       Skipped: {skipped}")
if skip_reasons:
    failed_path = os.path.join(os.path.dirname(AEMS_MANIFEST_PATH), "_video_embed_failed.txt")
    with open(failed_path, "w") as f:
        for vid, reason in skip_reasons.items():
            f.write(f"{vid},{reason}\n")
    print(f"       Failed list: {failed_path}")
