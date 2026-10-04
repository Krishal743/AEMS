"""Temporal grounding: given a question and its video, predict when the answer is.

Whole-video retrieval is solvable by matching a question against a transcript,
which is what makes the aggregate benchmark an artifact. Localisation is not:
every candidate window belongs to the same video and shares its vocabulary, so
a method must discriminate *within* a transcript rather than between documents.
The prior-only baselines bear this out — predicting the whole video scores
exactly 0 at IoU 0.5.

Labels come from locating the **answer** in the timecoded transcript; a method
sees only the **question** (`src/data/timecodes.py`).

Frame sampling decides whether the visual method can be judged at all. At 16
uniform frames per video there are ~20 s between frames against a ~9.5 s
moment, so 40% of spans contain no frame and the visual score there is 0.002.
Pass `--dense-frames` to score against a 3 s grid instead.
"""

import argparse, json, os
import numpy as np
import torch

from src.config import AEMS_MANIFEST_PATH, AEMS_FRAME_EMBEDDINGS_PATH, DEVICE, set_seeds
from src.data.timecodes import MAX_SEGMENTS
from src.evaluation.evaluate_retrieval import bootstrap_ci
from src.evaluation.moments import METHODS, encode_side, load_frames, score_moments

parser = argparse.ArgumentParser(description="Evaluate temporal grounding")
parser.add_argument("--manifest", default=AEMS_MANIFEST_PATH)
parser.add_argument("--moments",
                    default="data/processed/aems/metadata/aems_moments_{split}.json")
parser.add_argument("--split", default="test", choices=["train", "test"])
parser.add_argument("--max-segments", type=int, default=MAX_SEGMENTS)
parser.add_argument("--frame-embeddings", default=AEMS_FRAME_EMBEDDINGS_PATH)
parser.add_argument("--dense-frames", default=None)
parser.add_argument("--n-frames", type=int, default=16)
parser.add_argument("--visual-weight", type=float, default=None,
                    help="defaults to the value tuned on validation, if present")
parser.add_argument("--tuning", default="outputs/aems/moment_fusion_tuning.json")
parser.add_argument("--methods", nargs="+", default=list(METHODS))
parser.add_argument("--bootstrap-iters", type=int, default=2000)
parser.add_argument("--output", default="outputs/aems/moment_retrieval.json")
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()
set_seeds(args.seed)

visual_weight = args.visual_weight
if visual_weight is None and os.path.exists(args.tuning):
    visual_weight = json.load(open(args.tuning))["best_visual_weight"]
    print(f"[TUNE] using validation-tuned visual weight {visual_weight:g}", flush=True)
elif visual_weight is None:
    visual_weight = 0.5
    print("[TUNE] no tuning file; falling back to visual weight 0.5", flush=True)

bench = json.load(open(args.moments.format(split=args.split)))
moments = bench["moments"]
records = {r["video_id"]: r for r in json.load(open(args.manifest))
           if r.get("split") == args.split}
moments = [m for m in moments if m["video_id"] in records]
print(f"[DATA] {len(moments)} moments over "
      f"{len({m['video_id'] for m in moments})} {args.split} videos", flush=True)

frame_db, frame_time_db, interval = load_frames(args.dense_frames, args.frame_embeddings,
                                                args.n_frames)
print(f"[FRAMES] {'dense grid every %gs' % interval if interval else '%d uniform frames' % args.n_frames}"
      f" over {len(frame_db)} videos", flush=True)

q_clip, q_dense, seg_emb, seg_index, video_segs = encode_side(moments, records, DEVICE)
print(f"[ENC] {len(moments)} questions, {seg_emb.shape[0]} transcript segments", flush=True)

results, frame_hit, _ = score_moments(
    moments, records, seg_emb, seg_index, video_segs, q_clip, q_dense,
    frame_db, frame_time_db, DEVICE, max_segments=args.max_segments,
    visual_weight=visual_weight, n_frames=args.n_frames, methods=args.methods)

print(f"\n{'method':<14} {'mIoU':>7}  {'IoU>=0.3':>9} {'IoU>=0.5':>9} {'IoU>=0.7':>9}  "
      f"{'IoU>=0.5 CI':>18}")
print("-" * 76)
out = {}
for name in args.methods:
    vals = results.get(name)
    if vals is None or not len(vals):
        continue
    hits = (vals >= 0.5).astype(float)
    lo, hi = bootstrap_ci(torch.as_tensor(hits), iters=args.bootstrap_iters, seed=args.seed)
    out[name] = {"n": int(len(vals)), "mIoU": float(vals.mean()),
                 "IoU@0.3": float((vals >= 0.3).mean()), "IoU@0.5": float(hits.mean()),
                 "IoU@0.7": float((vals >= 0.7).mean()), "IoU@0.5_CI": [lo, hi]}
    print(f"{name:<14} {vals.mean():>7.4f}  {(vals >= 0.3).mean():>9.4f} "
          f"{hits.mean():>9.4f} {(vals >= 0.7).mean():>9.4f}  [{lo:.4f}, {hi:.4f}]")

if "visual" in results and len(frame_hit) == len(results["visual"]):
    v = results["visual"]
    print(f"\nvisual, by whether any frame lands in the ground-truth span "
          f"({frame_hit.mean():.1%} do):")
    for label, mask in (("frame inside", frame_hit), ("no frame", ~frame_hit)):
        if mask.sum():
            print(f"  {label:<13} n={int(mask.sum()):>5}  mIoU={v[mask].mean():.4f}  "
                  f"IoU>=0.5={(v[mask] >= 0.5).mean():.4f}")
    out["visual_by_frame_coverage"] = {"frac_with_frame": float(frame_hit.mean())}

os.makedirs(os.path.dirname(args.output), exist_ok=True)
with open(args.output, "w") as f:
    json.dump({"split": args.split, "n_moments": len(moments), "methods": out,
               "visual_weight": visual_weight,
               "frames": "dense" if args.dense_frames else "uniform"}, f, indent=2)
print(f"\n[SAVE] {args.output}")
