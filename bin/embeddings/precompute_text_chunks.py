"""Precompute per-chunk CLIP text embeddings for the late-interaction branch.

Saves {video_id: (n_chunks, 512)} so a query can be scored against a video's
best-matching passage instead of an average over the whole transcript.
"""

import argparse, os
import torch
import torch.nn.functional as F
import clip
from tqdm import tqdm
from src.config import AEMS_MANIFEST_PATH, AEMS_TEXT_CHUNKS_PATH_TEMPLATE, DEVICE, set_seeds
from src.data.metadata import load_metadata, filter_by_split
from src.data.text_chunks import video_chunks

parser = argparse.ArgumentParser(description="Precompute AEMS per-chunk CLIP text embeddings")
parser.add_argument("--split", required=True, choices=["train", "test"])
parser.add_argument("--manifest", default=AEMS_MANIFEST_PATH)
parser.add_argument("--output", default=None)
parser.add_argument("--batch-size", type=int, default=256)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()
set_seeds(args.seed)

out_path = args.output or AEMS_TEXT_CHUNKS_PATH_TEMPLATE.format(split=args.split)
os.makedirs(os.path.dirname(out_path), exist_ok=True)

print(f"[INIT] Loading CLIP on {DEVICE}")
model, _ = clip.load("ViT-B/32", device=DEVICE)
model.eval()

records = filter_by_split(load_metadata(args.manifest), split=args.split)
print(f"[DATA] {len(records)} records")


@torch.no_grad()
def encode(texts):
    out = []
    for i in range(0, len(texts), args.batch_size):
        tokens = clip.tokenize(texts[i:i + args.batch_size], truncate=True).to(DEVICE)
        out.append(F.normalize(model.encode_text(tokens).float(), dim=1).cpu())
    return torch.cat(out)


out, skipped, total_chunks = {}, 0, 0
for record in tqdm(records, desc=f"Encoding chunks {args.split}"):
    chunks = video_chunks(record)
    if not chunks:
        skipped += 1
        continue
    out[record["video_id"]] = encode(chunks)
    total_chunks += len(chunks)

torch.save(out, out_path)
print(f"\n[DONE] {len(out)} videos, {total_chunks} chunks "
      f"({total_chunks / max(len(out), 1):.1f} per video) -> {out_path}")
print(f"       Skipped: {skipped}")
