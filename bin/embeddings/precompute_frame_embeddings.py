"""Precompute per-frame CLIP embeddings.

The visual branch mean-pools 16 frames into one vector, which is the same
information loss the caption branch had before the passage branch was added:
a single matching moment is diluted by fifteen others. Keeping the frames
separate enables late interaction over frames (score a video by its
best-matching frame) and lets an evaluation hold one frame out of the index
and use it as a query.

Saves {video_id: (n_frames, 512)}, mirroring precompute_text_chunks.py.
"""

import argparse, glob, os
import torch
import torch.nn.functional as F
import clip
from PIL import Image
from tqdm import tqdm
from src.config import (AEMS_MANIFEST_PATH, AEMS_FRAMES_DIR, AEMS_FRAME_EMBEDDINGS_PATH,
                        NUM_FRAMES, DEVICE, set_seeds)
from src.data.metadata import load_metadata

parser = argparse.ArgumentParser(description="Precompute AEMS per-frame CLIP embeddings")
parser.add_argument("--manifest", default=AEMS_MANIFEST_PATH)
parser.add_argument("--frames-dir", default=AEMS_FRAMES_DIR)
parser.add_argument("--output", default=AEMS_FRAME_EMBEDDINGS_PATH)
parser.add_argument("--batch-size", type=int, default=256)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()
set_seeds(args.seed)
os.makedirs(os.path.dirname(args.output), exist_ok=True)

print(f"[INIT] Loading CLIP on {DEVICE}")
model, preprocess = clip.load("ViT-B/32", device=DEVICE)
model.eval()

videos = sorted({r["video_id"] for r in load_metadata(args.manifest)})
print(f"[DATA] {len(videos)} videos")


@torch.no_grad()
def encode(images):
    out = []
    for i in range(0, len(images), args.batch_size):
        batch = torch.stack(images[i:i + args.batch_size]).to(DEVICE)
        out.append(F.normalize(model.encode_image(batch).float(), dim=1).cpu())
    return torch.cat(out)


out, skipped, total = {}, 0, 0
for vid in tqdm(videos, desc="Encoding frames"):
    paths = sorted(glob.glob(os.path.join(args.frames_dir, vid, "frame_*.jpg")))
    if len(paths) != NUM_FRAMES:
        skipped += 1
        continue
    try:
        images = [preprocess(Image.open(p)) for p in paths]
    except Exception:
        skipped += 1
        continue
    out[vid] = encode(images)
    total += len(images)

torch.save(out, args.output)
print(f"\n[DONE] {len(out)} videos, {total} frames -> {args.output}")
print(f"       Skipped: {skipped}")
