"""Encode frames at a fixed time interval, for temporal grounding.

The deployed index samples 16 frames per video. On a median AEMS video that is
one frame every ~20 s, while a derived moment lasts ~9.5 s, so **40% of ground
truth spans contain no frame at all** and the visual method scores 0.002 on
them against 0.081 where a frame exists. Any conclusion about visual evidence
drawn from those features would be a conclusion about the sampling rate.

This samples on a time grid instead (default every 3 s), so a median moment
contains ~3 frames. Frames are decoded straight to CLIP and never written to
disk; timestamps are stored alongside the embeddings because localisation needs
to know *when* each frame was.
"""

import argparse, json, os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from src.config import AEMS_MANIFEST_PATH, DEVICE, set_seeds

parser = argparse.ArgumentParser(description="Encode frames on a fixed time grid")
parser.add_argument("--manifest", default=AEMS_MANIFEST_PATH)
parser.add_argument("--videos-from", default=None,
                    help="moment-benchmark JSON; only its videos are encoded")
parser.add_argument("--split", default="test")
parser.add_argument("--interval", type=float, default=3.0, help="seconds between frames")
parser.add_argument("--max-frames", type=int, default=400,
                    help="cap per video, so one long video cannot dominate the cost")
parser.add_argument("--batch-size", type=int, default=256)
parser.add_argument("--output", default="embeddings/aems_dense_frames_{split}.pt")
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()
set_seeds(args.seed)

import cv2
cv2.setNumThreads(4)

records = {r["video_id"]: r for r in json.load(open(args.manifest))
           if r.get("split") == args.split}
if args.videos_from:
    wanted = sorted({m["video_id"] for m in json.load(open(args.videos_from))["moments"]})
else:
    wanted = sorted(records)
wanted = [v for v in wanted if v in records]
print(f"[DATA] {len(wanted)} videos, one frame every {args.interval:g}s "
      f"(cap {args.max_frames})", flush=True)

import clip
model, preprocess = clip.load("ViT-B/32", device=DEVICE)
model.eval()


@torch.no_grad()
def encode(images):
    batch = torch.stack([preprocess(im) for im in images]).to(DEVICE)
    return F.normalize(model.encode_image(batch).float(), dim=1).cpu()


embeddings, timestamps, failed = {}, {}, []
pending_images, pending_owner = [], []


def flush():
    if not pending_images:
        return
    vecs = encode(pending_images)
    for owner, vec in zip(pending_owner, vecs):
        embeddings.setdefault(owner, []).append(vec)
    pending_images.clear()
    pending_owner.clear()


for video in tqdm(wanted, desc="Encoding"):
    record = records[video]
    duration = float(record.get("duration_seconds") or 0.0)
    path = record.get("video_path")
    if duration <= 0 or not path or not os.path.exists(path):
        failed.append(video)
        continue
    times = np.arange(args.interval / 2, duration, args.interval)[:args.max_frames]
    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        failed.append(video)
        continue
    kept = []
    for t in times:
        capture.set(cv2.CAP_PROP_POS_MSEC, float(t) * 1000.0)
        ok, frame = capture.read()
        if not ok:
            continue
        pending_images.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        pending_owner.append(video)
        kept.append(float(t))
        if len(pending_images) >= args.batch_size:
            flush()
    capture.release()
    if kept:
        timestamps[video] = torch.tensor(kept)
    else:
        failed.append(video)
flush()

embeddings = {v: torch.stack(rows) for v, rows in embeddings.items()}
for video in list(embeddings):
    n = min(len(embeddings[video]), len(timestamps.get(video, [])))
    embeddings[video] = embeddings[video][:n]
    timestamps[video] = timestamps[video][:n]

counts = np.array([len(v) for v in embeddings.values()]) if embeddings else np.array([0])
print(f"\n[DONE] {len(embeddings)} videos, {int(counts.sum()):,} frames "
      f"(median {int(np.median(counts))}/video); {len(failed)} failed")
out_path = args.output.format(split=args.split)
os.makedirs(os.path.dirname(out_path), exist_ok=True)
torch.save({"embeddings": embeddings, "timestamps": timestamps,
            "interval": args.interval}, out_path)
print(f"[SAVE] {out_path}")
if failed:
    print(f"[WARN] failed: {failed[:10]}{' ...' if len(failed) > 10 else ''}")
